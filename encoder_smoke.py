"""Start LLM in encoder-only mode and evaluate one real vision patch."""

import tempfile
from pathlib import Path

from reproduce import check_image, generate
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm import LLM
from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_PLACEHOLDER


def check_worker(worker):
    model = worker.model_runner.get_model()
    check_image(model, worker.vllm_config)
    return True


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        generate(root)
        (root / "full.safetensors").rename(root / "model.safetensors")
        vocab = ["<bos>", "<eos>", "<pad>", IMAGE_PLACEHOLDER, "<unk>"]
        vocab.extend(f"token{i}" for i in range(len(vocab), 256))
        tokenizer = Tokenizer(
            WordLevel(dict(zip(vocab, range(256), strict=True)), unk_token="<unk>")
        )
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
        ).save_pretrained(root)
        llm = LLM(
            model=str(root),
            enforce_eager=True,
            dtype="bfloat16",
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=1,
            gpu_memory_utilization=0.25,
            limit_mm_per_prompt={"image": 1},
            mm_encoder_only=True,
            mm_encoder_attn_backend="TORCH_SDPA",
            skip_mm_profiling=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            model_loader_extra_config={"enable_weights_track": True},
        )
        assert llm.collective_rpc(check_worker) == [True]
        print(
            "PASS: LLM encoder-only startup and first-stage image embedding", flush=True
        )


if __name__ == "__main__":
    main()
