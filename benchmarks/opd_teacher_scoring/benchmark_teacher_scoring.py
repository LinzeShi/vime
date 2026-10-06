"""Compare OPD scoring paths against an already running vLLM teacher.

Use a V2 teacher with raw_logprobs and per-row candidate scoring support.
This measures HTTP scoring, not training throughput. No teacher is launched here.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx
import torch
from transformers import AutoTokenizer

from vime.rollout.on_policy_distillation import post_process_rewards, reward_func
from vime.utils import http_utils
from vime.utils.types import Sample


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt_ids = tokenizer.encode("Explain teacher scoring in on-policy distillation. ", add_special_tokens=False)
    response_ids = tokenizer.encode(
        "The teacher scores each answer token using the full preceding context. ", add_special_tokens=False
    )
    modes = ["prompt-logprobs", "actual-only", "per-position"]
    rng = random.Random(0)
    records = []
    parity = []
    config = argparse.Namespace(
        rm_url=args.teacher_url, rollout_temperature=1.0, opd_teacher_model=args.model, reward_key=None
    )

    async def score(tokens, length, mode):
        start = time.perf_counter()
        call_args = argparse.Namespace(**vars(config), opd_teacher_scoring=mode)
        sample = Sample(tokens=tokens, response_length=length)
        if mode == "actual-only":
            # Control arm: change only the old interface's top-candidate count.
            payload = {
                "token_ids": tokens,
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "skip_special_tokens": False,
                    "prompt_logprobs": 0,
                },
            }
            if args.model:
                payload["model"] = args.model
            sample.reward = await http_utils.post(args.teacher_url, payload, max_retries=1)
        else:
            sample.reward = await reward_func(call_args, sample)
        post_process_rewards(call_args, [sample])
        return sample.teacher_log_probs, (time.perf_counter() - start) * 1000

    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        http_utils._http_client = client
        try:
            for lengths in args.lengths:
                p, r = (int(x) for x in lengths.split(":"))
                if p < 1 or r < 1:
                    raise ValueError("Both prompt and response lengths must be positive")
                tokens = (prompt_ids * ((p + len(prompt_ids) - 1) // len(prompt_ids)))[:p]
                tokens += (response_ids * ((r + len(response_ids) - 1) // len(response_ids)))[:r]

                reference, _ = await score(tokens, r, "prompt-logprobs")
                for mode in modes[1:]:
                    scores, _ = await score(tokens, r, mode)
                    torch.testing.assert_close(scores, reference, atol=1e-3, rtol=1e-4)
                    parity.append(
                        {"lengths": lengths, "mode": mode, "max_abs": (scores - reference).abs().max().item()}
                    )

                # Compare both formats from the SAME forward, avoiding bf16
                # differences caused by different concurrent batch shapes.
                async def paired_check(tokens, p, r):
                    payload = {
                        "token_ids": tokens,
                        "sampling_params": {
                            "max_tokens": 1,
                            "prompt_logprobs": 1,
                            "prompt_logprob_start": p - 1,
                            "prompt_logprob_token_ids": [[token] for token in tokens[p:]],
                        },
                    }
                    if args.model:
                        payload["model"] = args.model
                    reward = await http_utils.post(args.teacher_url, payload, max_retries=1)
                    old = Sample(tokens=tokens, response_length=r, reward=reward)
                    new = Sample(tokens=tokens, response_length=r, reward=reward)
                    post_process_rewards(config, [old])
                    post_process_rewards(argparse.Namespace(**vars(config), opd_teacher_scoring="per-position"), [new])
                    torch.testing.assert_close(new.teacher_log_probs, old.teacher_log_probs, atol=1e-3, rtol=1e-4)
                    return (new.teacher_log_probs - old.teacher_log_probs).abs().max().item()

                pair_errors = await asyncio.gather(*(paired_check(tokens, p, r) for _ in range(args.concurrency)))
                parity.append({"lengths": lengths, "mode": "same-request", "max_abs": max(pair_errors)})
                for mode in modes:
                    await asyncio.gather(*(score(tokens, r, mode) for _ in range(args.concurrency)))
                for repetition in range(args.repeats):
                    order = modes.copy()
                    rng.shuffle(order)
                    for mode in order:
                        wave = await asyncio.gather(*(score(tokens, r, mode) for _ in range(args.concurrency)))
                        for index, (scores, latency) in enumerate(wave):
                            records.append(
                                {
                                    "lengths": lengths,
                                    "concurrency": args.concurrency,
                                    "round": repetition,
                                    "index": index,
                                    "mode": mode,
                                    "latency_ms": latency,
                                    "max_abs_vs_single": (scores - reference).abs().max().item(),
                                }
                            )
        finally:
            http_utils._http_client = None
            args.output.write_text(
                json.dumps(
                    {"config": {**vars(args), "output": str(args.output)}, "parity": parity, "requests": records},
                    indent=2,
                )
            )
    for lengths in args.lengths:
        for mode in modes:
            times = [row["latency_ms"] for row in records if row["lengths"] == lengths and row["mode"] == mode]
            print(f"{lengths} concurrency={args.concurrency} {mode}: median {statistics.median(times):.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-url", required=True, help="Teacher /inference/v1/generate URL")
    parser.add_argument("--tokenizer", required=True, help="Teacher-compatible tokenizer name or local path")
    parser.add_argument("--model", help="Optional served teacher model name")
    parser.add_argument("--lengths", nargs="+", default=["64:16", "512:128", "2048:512"])
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.repeats < 1:
        parser.error("concurrency and repeats must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    asyncio.run(run(args))
