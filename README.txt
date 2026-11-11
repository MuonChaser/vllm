vLLM #66896 DeepSeek-V4.1 image delimiter reproducer

The scripts generate their own two-layer BF16 checkpoint and tokenizer.
They require a CUDA vLLM development installation. Use the vLLM CUDA
installation guide at docs/getting_started/installation/gpu.cuda.inc.md.
The end-to-end runs used H100, Python 3.12, torch 2.14.0+cu130,
Transformers 5.22.2, CUDA 13.0 and GCC 13.4. The cached precompiled CUDA
extensions were from 7691ff6e5b70930b80d09ec14fed84d1a2ea612f.
The tested Python source base was f460a5ed01599c549f24751bdf6ee8026c590f14.
To select the same extension build in an editable install, set
VLLM_USE_PRECOMPILED=1 and VLLM_PRECOMPILED_WHEEL_COMMIT to the full
extension commit above when running the documented uv pip install command.

Download all three scripts into one directory. From an activated CUDA
development environment, set repro_dir to that directory and run from
the vLLM source checkout:

    export PYTHONPATH="$PWD"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python "$repro_dir/remote_e2e.py" text

Use a single idle H100 (select it via CUDA_VISIBLE_DEVICES). The test
invokes the real LLM entry point. Baseline initialization raises a
missing-weight error for image_start, image_end and image_newline.
The patched checkout starts and generates four tokens with finite
logprobs. The image case uses a full checkpoint and exercises a PIL
image request as well as a text request:

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python "$repro_dir/remote_e2e.py" image

The serialization setting is for this local test's trusted worker callback.
On H100, generated token IDs, sampled-token logprobs and first-stage
image span matched between baseline and patch in the image case.

The loader-level fixture uses two Gloo pipeline ranks and one CUDA device.
It also runs missing-weight checks with a single rank.

    VLLM_DISABLE_PYNCCL=1 python "$repro_dir/reproduce.py" pipeline-disabled
    VLLM_DISABLE_PYNCCL=1 python "$repro_dir/reproduce.py" pipeline
    VLLM_DISABLE_PYNCCL=1 python "$repro_dir/reproduce.py" single

These are reduced synthetic checkpoints with deterministic weights. The
end-to-end cases use one GPU, and the pipeline cases test the real loader.
