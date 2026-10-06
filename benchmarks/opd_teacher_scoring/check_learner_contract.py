"""Real teacher HTTP scores -> existing VIME learner math and CPU backward.

This is a loss-contract test, not a distributed Megatron training run.
Only the unused Megatron parallel-state import is stubbed (single rank).
"""
import argparse
import asyncio
import json
import sys
import types
from pathlib import Path

import httpx
import torch
from transformers import AutoTokenizer

from vime.rollout.on_policy_distillation import post_process_rewards, reward_func
from vime.utils import http_utils
from vime.utils.types import Sample

core = types.ModuleType("megatron.core")
core.mpu = types.SimpleNamespace(get_context_parallel_world_size=lambda: 1)
sys.modules["megatron"] = types.ModuleType("megatron")
sys.modules["megatron.core"] = core
from vime.backends.megatron_utils.loss import apply_opd_kl_to_advantages
from vime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from vime.utils.ppo_utils import compute_policy_loss


def learner_step(teacher_scores):
    torch.manual_seed(123)
    lengths = [t.numel() for t in teacher_scores]
    total = sum(lengths)
    logits = torch.nn.Parameter(torch.randn(total, 8))
    targets = torch.arange(total) % 8
    initial = logits.detach().clone()
    old_logp = initial.log_softmax(-1)[torch.arange(total), targets]
    old_parts = list(old_logp.split(lengths))
    base_advantages = [torch.linspace(-0.3, 0.4, n) for n in lengths]
    data = {"teacher_log_probs": teacher_scores}
    apply_opd_kl_to_advantages(
        argparse.Namespace(opd_kl_coef=0.7, opd_type="vllm"),
        data, base_advantages, old_parts,
    )
    masks = [torch.ones(n) for n in lengths]
    masks[0][1] = 0
    reduce = get_sum_of_sample_mean([n + 3 for n in lengths], lengths, masks)
    current = logits.log_softmax(-1)[torch.arange(total), targets]
    # Execute the repository function eagerly; kernel compilation is not under test.
    eager_policy_loss = getattr(compute_policy_loss, "_torchdynamo_orig_callable", compute_policy_loss)
    per_token, _ = eager_policy_loss(old_logp - current, torch.cat(base_advantages), 0.2, 0.2)
    loss = reduce(per_token) / len(lengths)
    loss.backward()
    gradient = logits.grad.clone()
    assert gradient.abs().max() > 0
    assert torch.count_nonzero(gradient[1]) == 0
    optimizer = torch.optim.SGD([logits], lr=0.05)
    optimizer.step()
    assert not torch.equal(logits, initial)
    return {
        "loss": loss.detach(), "gradient": gradient, "updated_parameters": logits.detach().clone(),
        "advantages": torch.cat(base_advantages), "reverse_kl": torch.cat(data["opd_reverse_kl"]),
    }


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    examples = [
        ("What is on-policy distillation?", " The student writes an answer and the teacher scores its tokens."),
        ("Why does token order matter?", " Each score depends on its prefix."),
    ]
    scores = {"prompt-logprobs": [], "per-position": []}
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        http_utils._http_client = client
        try:
            for prompt, answer in examples:
                p = tokenizer.encode(prompt, add_special_tokens=False)
                r = tokenizer.encode(answer, add_special_tokens=False)
                for mode in scores:
                    config = argparse.Namespace(
                        rm_url=args.url + "/inference/v1/generate", rollout_temperature=0.7,
                        opd_teacher_model="opd-teacher", reward_key=None, opd_teacher_scoring=mode,
                    )
                    sample = Sample(tokens=p + r, response_length=len(r))
                    sample.reward = await reward_func(config, sample)
                    post_process_rewards(config, [sample])
                    scores[mode].append(sample.teacher_log_probs)
        finally:
            http_utils._http_client = None
    old = learner_step(scores["prompt-logprobs"])
    new = learner_step(scores["per-position"])
    differences = {}
    for key in old:
        torch.testing.assert_close(old[key], new[key], atol=1e-5, rtol=1e-5)
        differences[key] = (old[key] - new[key]).abs().max().item()
    # Negative control: a wrong response alignment must change the gradient.
    shifted = learner_step([s.roll(1) for s in scores["per-position"]])
    control_delta = (shifted["gradient"] - new["gradient"]).abs().max().item()
    assert control_delta > 1e-5
    result = {
        "scope": "real GPU teacher, existing VIME learner math, CPU backward and optimizer; not full training",
        "response_lengths": [s.numel() for s in scores["per-position"]],
        "teacher_scores": {k: [s.tolist() for s in v] for k, v in scores.items()},
        "max_abs_differences": differences, "wrong_alignment_gradient_delta": control_delta,
        "loss_mask_test": "PASS", "status": "PASS",
    }
    (Path(__file__).parent / "live-score-gradient.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "teacher_scores"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    asyncio.run(main(parser.parse_args()))
