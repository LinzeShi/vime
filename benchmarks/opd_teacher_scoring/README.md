# OPD teacher scoring benchmark

Reproduction materials for [VIME #466](https://github.com/vllm-project/vime/issues/466).
These files are archived on the author's `evidence/opd-teacher-scoring` branch in the fork, outside the upstream PR's merge diff.
The artifact commit includes the VIME implementation needed to run the comparison.

The benchmark and learner-check scripts are the scripts used for the original measurements.
The reported numbers are from those measurements, not a new performance run of this packaging.
`metadata.json` records the software, model revision, teacher options and hashes of the measured runtime files.
Machine-local tokenizer and output paths were normalized in the exported JSON configuration; all timing and parity records are unchanged.

## Recompute the table

From this artifact commit's repository root, no GPU or third-party Python packages are needed:

```bash
python benchmarks/opd_teacher_scoring/summarize.py
```

The script validates request counts, duplicates, measurement values and the archived parity results before printing the table.
It includes all five input lengths and both concurrency levels, including cases with no speedup.
The files contain 180 timed requests at concurrency 1 and 720 at concurrency 4, across all three scoring paths.
Warmups and correctness probes are excluded from the timing records.

## Repeat the measurements

Use vLLM commit `84bcbc62644356270aaaa5e2d0237d03adc9bb3a` with matching compiled binaries.
The measured environment used Python 3.12.4, PyTorch 2.13.0+cu130, Transformers 5.17.0, NumPy 2.3.5 and httpx 0.28.1 on WSL2 with one RTX 4070 Laptop GPU (8188 MiB).
Install VIME from this artifact checkout in that compatible environment:

```bash
python -m pip install -e . --no-deps
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir ./opd-teacher-model
```

Start the teacher in one terminal from the repository root:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 OMP_NUM_THREADS=4 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false \
  vllm serve ./opd-teacher-model \
  --host 127.0.0.1 --port 8000 --served-model-name opd-teacher \
  --enable-scale-out --dtype bfloat16 --logprobs-mode raw_logprobs \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.6 --enforce-eager --no-enable-prefix-caching
```

After the teacher is healthy, run the two comparisons from a second terminal with the same Python environment:

```bash
OPD_RESULTS="$(mktemp -d)"
for concurrency in 1 4; do
  PYTHONPATH="$PWD" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    python benchmarks/opd_teacher_scoring/benchmark_teacher_scoring.py \
    --teacher-url http://127.0.0.1:8000/inference/v1/generate \
    --model opd-teacher --tokenizer ./opd-teacher-model \
    --lengths 1:1 64:16 512:128 2048:512 512:2048 \
    --concurrency "$concurrency" --repeats 12 \
    --output "$OPD_RESULTS/concurrency-$concurrency.json"
done
python benchmarks/opd_teacher_scoring/summarize.py --results "$OPD_RESULTS"
```

`summarize.py` checks the zero-error parity claim recorded in this archive; it deliberately rejects nonzero differences rather than silently treating a new run as identical to the recorded one.
The live benchmark itself checks scores at `atol=1e-3, rtol=1e-4` and retains observed differences.
Other hardware or kernel versions can produce nonzero differences within that tolerance; inspect those results independently rather than changing the archived assertion to make a repeat pass.

The optional learner-contract check uses the same running teacher:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  python benchmarks/opd_teacher_scoring/check_learner_contract.py \
  --url http://127.0.0.1:8000 --model ./opd-teacher-model
```

This writes a new `live-score-gradient.json` beside the script, separate from the retained result in `results/learner-contract.json`.
It feeds real teacher scores through the existing OPD advantage and policy-loss functions, masking, CPU backward and one SGD step.
It uses controlled student tensors and a single-rank import stub, so it is not a Megatron model training run.

## Measurement scope

- The paths are current top-1 prompt scores, `prompt_logprobs=0` (actual-token-only control), and per-position response candidates. They send the same full input and use `max_tokens=1`.
- The timer surrounds the VIME request and response-processing calls. It includes HTTP and score parsing, but excludes model loading and tokenization.
- Inputs repeat fixed text to reach the requested lengths; they are not sampled training trajectories. Each shape receives one warmup wave per mode, then 12 measured rounds with deterministically shuffled mode order.
- Concurrency means simultaneous requests for the same input. Correctness probes compare both output formats from a single forward as well as separate single-request executions.
- The retained runs have exact parity in those probes. Some concurrent requests differ from the single-request reference by up to 0.23923 in every mode, including the old path, due to different bf16 batch execution. These observations remain in the JSON.
- Short inputs have no consistent gain; the 1:1 concurrent case is slower with per-position scoring. HTTP scoring improvements do not establish training-throughput or model-quality improvements.
- The original script keeps VIME's transport retry policy in the production paths and uses one attempt in the actual-only control. Do not use runs with transport errors or retries for timing comparisons. An interrupted run can leave partial JSON; the summarizer rejects incomplete record sets.
- Full multi-GPU training and real image-model inference were not run. Image routing and alignment have CPU regression coverage in the implementation PR.
