"""Load reduced DeepSeek-V4.1 checkpoints via the normal safetensors loader."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors.torch import save_file

from vllm.config import (
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
)
from vllm.utils.torch_utils import set_default_torch_dtype

DELIMITERS = {"image_start": 1.0, "image_newline": 2.0, "image_end": 3.0}


def write_config(root):
    root.mkdir(exist_ok=True)
    config = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": 3,
        "text_config": {
            "vocab_size": 256,
            "hidden_size": 512,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 64,
            "o_lora_rank": 64,
            "o_groups": 1,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "rope_scaling": {"rope_type": "default"},
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "sliding_window": 128,
            "compress_ratios": [0, 0],
            "compress_rope_theta": 160000,
            # Retain the next source layer ID from the full config while
            # keeping only its first two pure sliding-window decoder layers.
            "kv_source_layer_ids": [2],
            "index_source_layer_ids": [2],
            "index_topk": 1,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
        },
        "vision_config": {
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "num_attention_heads": 1,
            "intermediate_size": 32,
            "patch_size": 2,
            "downsample_ratio": 1,
            "min_pixels": 4,
            "max_image_tokens": 16,
        },
    }
    (root / "config.json").write_text(json.dumps(config))


def config_for(root, image_limit, pp_size, device="cpu"):
    return VllmConfig(
        model_config=ModelConfig(
            model=str(root),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=128,
            enforce_eager=True,
            limit_mm_per_prompt={"image": image_limit},
            mm_encoder_attn_backend="TORCH_SDPA",
        ),
        device_config=DeviceConfig(device),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=pp_size,
            distributed_executor_backend="mp" if pp_size > 1 else None,
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=1,
        ),
        load_config=LoadConfig(
            load_format="safetensors",
            use_tqdm_on_load=False,
            model_loader_extra_config={"enable_weights_track": True},
        ),
    )


def checkpoint(model):
    """Serialize real parameter shapes with the original checkpoint names."""
    weights = {}
    for name, p in model.named_parameters():
        name = name.removeprefix("image_delimiters.")
        name = name.removeprefix("language_model.model.")
        name = {
            "embed_tokens.weight": "embed.weight",
            "language_model.lm_head.weight": "head.weight",
        }.get(name, name)
        data = torch.full_like(p, DELIMITERS.get(name, 0.25), device="cpu")
        if ".experts.routed_experts." in name:
            prefix, leaf = name.split(".experts.routed_experts.")
            if leaf == "w13_weight":
                for expert, pair in enumerate(data):
                    for proj, w in zip(("w1", "w3"), pair.chunk(2), strict=True):
                        weights[f"{prefix}.experts.{expert}.{proj}.weight"] = w.clone()
            elif leaf == "w2_weight":
                for expert, w in enumerate(data):
                    weights[f"{prefix}.experts.{expert}.w2.weight"] = w.clone()
            else:
                raise AssertionError(name)
        else:
            weights[name] = data
    return weights


def load(root, image_limit, pp_size, weights, device="cpu"):
    save_file(weights, root / "model.safetensors")
    config = config_for(root, image_limit, pp_size, device)
    with set_current_vllm_config(config):
        model = DefaultModelLoader(config.load_config).load_model(
            config, config.model_config
        )
    return model, config


def check_image(model, config):
    kwargs = {
        "patches": torch.ones(1, 3, 2, 2, device="cuda"),
        "vit_grid": torch.tensor([[1, 1]]),
        "llm_grid": torch.tensor([[1, 1]]),
        "types": torch.tensor([IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END]),
    }
    with torch.inference_mode(), set_current_vllm_config(config):
        (span,) = model.embed_multimodal(**kwargs)
        hidden_size = model.config.hidden_size
        assert span.shape == (4, hidden_size)
        torch.testing.assert_close(
            span[[0, 2, 3]],
            span.new_tensor([1.0, 2.0, 3.0])[:, None].expand(3, hidden_size),
            rtol=0,
            atol=0,
        )
        assert torch.isfinite(span[1]).all() and span[1].abs().sum() > 0
        dest = [None]
        model.postprocess_encoder_output(
            {"default": span[1:2]}, [0], [4], dest, batch_mm_kwargs=kwargs
        )
        torch.testing.assert_close(dest[0], span, rtol=0, atol=0)
    digest = hashlib.sha256(span.float().cpu().numpy().tobytes()).hexdigest()
    print(
        f"PASS: first-stage eager image encoder and graph postprocessing use exact legacy vectors; span SHA256={digest}",
        flush=True,
    )


def worker(rank, pp_size, store, root, case):
    # Use real PP groups, with Gloo so both ranks can load CPU tensors on one host.
    root = root / f"stage-{rank}"
    write_config(root)
    config = config_for(root, 1, pp_size)
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=pp_size,
            rank=rank,
            local_rank=0,
            distributed_init_method=store,
            backend="gloo",
        )
        initialize_model_parallel(1, pp_size, backend="gloo")
    try:
        assert get_pp_group().is_first_rank == (rank == 0)
        # Build the full fixture on each worker's independent single-stage run
        # beforehand; actual workers receive only their own checkpoint tensors.
        from safetensors.torch import load_file

        weights = load_file(root.parent / "full.safetensors")
        if case == "single":
            text = {
                n: p
                for n, p in weights.items()
                if not n.startswith(("vision.", "aligner.", "image_"))
            }
            model, _ = load(root, 0, 1, text)
            torch.testing.assert_close(
                model.language_model.lm_head.weight, text["head.weight"]
            )
            print("PASS: disabled images load without any vision weights", flush=True)
            model, _ = load(root, 0, 1, weights)
            torch.testing.assert_close(
                model.language_model.lm_head.weight, text["head.weight"]
            )
            print(
                "PASS: disabled images accept the same full legacy checkpoint",
                flush=True,
            )
            for image_limit, key, missing in (
                (
                    0,
                    "layers.0.hc_attn_base",
                    "language_model.model.layers.0.hc_attn_base",
                ),
                (1, "image_newline", "image_newline"),
            ):
                incomplete = dict(text if image_limit == 0 else weights)
                incomplete.pop(key)
                try:
                    load(root, image_limit, 1, incomplete)
                except ValueError as exc:
                    assert (
                        "Following weights were not initialized from checkpoint"
                        in str(exc)
                        and missing in str(exc)
                    ), exc
                else:
                    raise AssertionError(f"missing {key} was not detected")
                print(f"PASS: strict check detects missing {key}", flush=True)
        else:
            stage = {
                n: p
                for n, p in weights.items()
                if n.startswith(f"layers.{rank}.")
                or (
                    rank == 0
                    and n.startswith(("embed.", "vision.", "aligner.", "image_"))
                )
                or (rank == 1 and n.startswith(("norm.", "head.")))
            }
            image_limit = int(case != "pipeline-disabled")
            if not image_limit:
                stage = {
                    n: p
                    for n, p in stage.items()
                    if not n.startswith(("vision.", "aligner.", "image_"))
                }
            model, cfg = load(
                root,
                image_limit,
                pp_size,
                stage,
                "cuda" if rank == 0 and image_limit else "cpu",
            )
            torch.testing.assert_close(
                model.language_model.model.layers[rank].attn_norm.weight.cpu(),
                stage[f"layers.{rank}.attn_norm.weight"],
            )
            if not image_limit:
                print(
                    f"PASS: PP rank {rank} loads its own shard with image limit zero",
                    flush=True,
                )
            elif rank == 0:
                check_image(model, cfg)
                # Full legacy checkpoint on rank 0, with the new child forced
                # through the non-UVA tower offloader's forward wrapper.
                if os.getenv("TEST_DELIMITER_OFFLOAD") == "1":
                    from vllm.model_executor.offloader import set_offloader
                    from vllm.model_executor.offloader.uva import UVAOffloader

                    offloader = UVAOffloader(1 << 20, {"image_delimiters"})
                    assert not offloader.uva_offloading
                    set_offloader(offloader)
                    try:
                        offloaded, cfg = load(root, 1, pp_size, weights, "cuda")
                        assert all(
                            p.device.type == "cpu"
                            for p in offloaded.image_delimiters.parameters()
                        )
                        check_image(offloaded, cfg)
                        print(
                            "PASS: first-stage full checkpoint and non-UVA delimiter offloading",
                            flush=True,
                        )
                    finally:
                        set_offloader(None)
            else:
                assert not any(
                    n.startswith(("vision.", "aligner.", "image_"))
                    for n, _ in model.named_parameters()
                )
                print(
                    "PASS: PP rank 1 loads its own shard with images enabled and no tower tensors",
                    flush=True,
                )
                model, _ = load(root, 1, pp_size, weights)
                torch.testing.assert_close(
                    model.language_model.lm_head.weight, weights["head.weight"]
                )
                print(
                    "PASS: PP rank 1 also accepts the full legacy checkpoint",
                    flush=True,
                )
    finally:
        cleanup_dist_env_and_memory()


def generate(root):
    write_config(root)
    cfg = config_for(root, 1, 1)
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{root / 'prepare-store'}",
            backend="gloo",
        )
        initialize_model_parallel(1, 1, backend="gloo")
        with torch.device("cpu"), set_default_torch_dtype(torch.bfloat16):
            model = initialize_model(cfg)
        save_file(checkpoint(model), root / "full.safetensors")
    cleanup_dist_env_and_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("single", "pipeline", "pipeline-disabled"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        generate(root)
        if args.case == "single":
            worker(0, 1, f"file://{root / 'single-store'}", root, args.case)
        else:
            mp.spawn(
                worker,
                args=(2, f"file://{root / 'pp-store'}", root, args.case),
                nprocs=2,
            )


if __name__ == "__main__":
    main()
