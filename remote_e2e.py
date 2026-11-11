"""Real LLM startup/generation with reduced, on-disk DeepSeek-V4.1 weights."""
import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from reproduce import checkpoint, config_for, write_config
from encoder_smoke import check_worker
from vllm import LLM, SamplingParams
from vllm.config import set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory, init_distributed_environment, initialize_model_parallel
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_PLACEHOLDER, IMAGE_SENTINEL_BASE_ID
from vllm.utils.torch_utils import set_default_torch_dtype


def prepare(root, case):
    write_config(root)
    data = json.loads((root / 'config.json').read_text())
    data['text_config']['vocab_size'] = 129280
    data['image_token_id'] = IMAGE_SENTINEL_BASE_ID
    (root / 'config.json').write_text(json.dumps(data))
    vocab = [f'token{i}' for i in range(129280)]
    vocab[:4] = ['<bos>', '<eos>', '<pad>', '<unk>']
    vocab[IMAGE_SENTINEL_BASE_ID] = IMAGE_PLACEHOLDER
    tokenizer = Tokenizer(WordLevel(dict(zip(vocab, range(len(vocab)), strict=True)), unk_token='<unk>'))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token='<bos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>', additional_special_tokens=[IMAGE_PLACEHOLDER]).save_pretrained(root)
    cfg = config_for(root, 1, 1)
    with set_current_vllm_config(cfg):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method=f'file://{root / "store"}', backend='gloo')
        initialize_model_parallel(1, 1, backend='gloo')
        with torch.device('cpu'), set_default_torch_dtype(torch.bfloat16):
            model = initialize_model(cfg)
        weights = checkpoint(model)
        if case in ('text', 'missing-text'):
            weights = {n:p for n,p in weights.items() if not n.startswith(('vision.', 'aligner.', 'image_'))}
        if case == 'missing-text':
            weights.pop('layers.0.hc_attn_base')
        if case == 'missing-image':
            weights.pop('image_newline')
        save_file(weights, root / 'model.safetensors')
    cleanup_dist_env_and_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['text', 'full', 'image', 'missing-text', 'missing-image'])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prepare(root, args.case)
        image_limit = int(args.case in ('image', 'missing-image'))
        llm = LLM(
            model=str(root), enforce_eager=True, dtype='bfloat16',
            max_model_len=128, max_num_batched_tokens=128, max_num_seqs=1,
            gpu_memory_utilization=0.1, kv_cache_memory_bytes=32*1024**2,
            limit_mm_per_prompt={'image': image_limit}, skip_mm_profiling=True,
            mm_encoder_attn_backend='TORCH_SDPA', enable_prefix_caching=False,
            enable_chunked_prefill=False, model_loader_extra_config={'enable_weights_track':True},
        )
        params = SamplingParams(max_tokens=4, temperature=0, logprobs=5, detokenize=False, ignore_eos=True)
        requests = [('text', {'prompt_token_ids':[7, 8]})]
        if image_limit:
            assert llm.collective_rpc(check_worker) == [True]
            requests.append(('image', {'prompt_token_ids':[IMAGE_SENTINEL_BASE_ID], 'multi_modal_data':{'image':Image.new('RGB',(2,2),(120,130,140))}}))
        for name, prompt in requests:
            output = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]
            assert len(output.token_ids) == 4
            assert output.logprobs is not None
            logprobs = [step[token].logprob for step,token in zip(output.logprobs,output.token_ids,strict=True)]
            assert all(math.isfinite(p) for p in logprobs)
            print('RESULT '+json.dumps({'case':args.case,'request':name,'token_ids':list(output.token_ids),'logprobs':logprobs}),flush=True)
        print('PASS: real LLM startup and generation completed',flush=True)
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
