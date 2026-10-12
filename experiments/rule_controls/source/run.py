"""Offline BF16 inference with a transaction per completed sample."""
import argparse
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

from common import atomic_json, connect_db, digest, now, parse_probability, require_metadata, after_reasoning, validate_frozen


def effective_generation_limit(config, model_spec):
    limit = model_spec.get('max_new_tokens', config['max_new_tokens'])
    if type(limit) is not int or limit < 1:
        raise ValueError('Effective max_new_tokens must be a positive integer')
    return limit


def validate_context_budget(input_tokens, output_limit, context_limit):
    """Reject context extrapolation before generation, including padded inputs."""
    for name, value in (('input_tokens', input_tokens), ('output_limit', output_limit),
                        ('max_position_embeddings', context_limit)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be an explicit positive integer')
    if input_tokens + output_limit > context_limit:
        raise RuntimeError(f'Input plus output budget exceeds model context: '
                           f'{input_tokens}+{output_limit}>{context_limit}')


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    run_dir = config_path.parent
    validate_frozen(run_dir)
    lock = (run_dir / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = run_dir / config['manifest']
    rows = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s]
    model_spec = next(m for m in config['models'] if m['name'] == args.model)
    effective_max_new_tokens = effective_generation_limit(config, model_spec)
    db = connect_db(run_dir / 'results.sqlite')
    require_metadata(db, 'config_sha256', digest(config_path))
    require_metadata(db, 'manifest_sha256', digest(manifest))
    require_metadata(db, 'source_hashes', {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    done = {r[0] for r in db.execute('SELECT item_id FROM results WHERE model=?', (args.model,))}
    remaining = [r for r in rows if r['id'] not in done]
    if not remaining:
        print(f'{args.model}: all {len(rows)} records already saved.', flush=True)
        return
    random.Random(config['seed']).shuffle(remaining)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config.get('cpu_threads', 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    env = {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
           'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
           'accelerate': importlib.metadata.version('accelerate')}
    require_metadata(db, 'environment', env)
    atomic_json(run_dir / 'progress.json', {'state': 'loading', 'model': args.model, 'saved': len(done),
        'total': len(rows), 'updated_at': now(), 'pid': os.getpid()})
    model_path = Path(model_spec['path'])
    provenance = {}
    for name in ('config.json', 'generation_config.json', 'tokenizer_config.json',
                 'model.safetensors.index.json', 'MODEL_PROVENANCE.json', 'download_manifest.json'):
        if (model_path / name).is_file():
            provenance[name] = {'sha256': digest(model_path / name)}
    provenance['weight_files'] = [{ 'name': p.name, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns }
        for p in sorted(model_path.glob('*.safetensors'))]
    require_metadata(db, f'model_provenance:{args.model}', provenance)
    print(json.dumps({'event': 'loading', 'model': args.model, 'environment': env}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    model.eval()
    atomic_json(run_dir / 'progress.json', {'state': 'generating', 'model': args.model,
        'saved': len(done), 'total': len(rows), 'updated_at': now(), 'pid': os.getpid()})
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=effective_max_new_tokens,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    started = time.monotonic()
    initial_saved = len(done)
    batch_size = config['batch_size']
    offset = 0
    while offset < len(remaining):
        batch = remaining[offset:offset + batch_size]
        messages = [[{'role': 'user', 'content': r['prompt']}] for r in batch]
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                    **model_spec.get('chat_template_kwargs', {})) for m in messages]
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=False).to('cuda')
        if inputs.input_ids.shape[1] > config['max_input_tokens']:
            raise RuntimeError(f'Prompt exceeds frozen token budget: {inputs.input_ids.shape[1]}')
        validate_context_budget(inputs.input_ids.shape[1], effective_max_new_tokens,
                                getattr(model.config, 'max_position_embeddings', None))
        began = time.monotonic()
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=generation)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(json.dumps({'event': 'oom_reduce_batch', 'batch_size': batch_size, 'time': now()}), flush=True)
            continue
        duration = time.monotonic() - began
        for item, input_ids, mask, output_ids, prompt in zip(batch, inputs.input_ids,
                inputs.attention_mask, generated, prompts):
            token_ids = output_ids[input_ids.shape[0]:].tolist()
            stop_at = next((i for i, t in enumerate(token_ids) if t in eos_set), None)
            terminated = stop_at is not None
            if terminated:
                token_ids = token_ids[:stop_at + 1]
            raw = tokenizer.decode(token_ids, skip_special_tokens=True)
            thinking = model_spec.get('chat_template_kwargs', {}).get('enable_thinking', False)
            # Read the close marker from the unfiltered decoding too, so tokenizer special-token
            # handling cannot silently turn unfinished reasoning into a final answer.
            full_raw = tokenizer.decode(token_ids, skip_special_tokens=False)
            if thinking:
                close_ids = tokenizer.encode('</think>', add_special_tokens=False)
                final_tokens = after_reasoning(token_ids, close_ids)
                final = tokenizer.decode(final_tokens, skip_special_tokens=True).strip() if final_tokens is not None else None
                raw = full_raw
            else:
                final = raw
            probability = parse_probability(final) if terminated and final is not None else None
            status = 'truncated' if not terminated else ('ok' if probability is not None else 'invalid')
            metadata = dict(item, rendered_prompt=prompt, actual_batch_size=len(batch),
                effective_max_new_tokens=effective_max_new_tokens,
                model_mode=model_spec['mode'], finish_reason='eos' if terminated else 'length',
                final_output=final, thinking_segment_closed=(final_tokens is not None if thinking else None),
                reasoning_nonempty=(bool(full_raw.split('</think>', 1)[0].replace('<think>', '').strip())
                                    if thinking and final_tokens is not None else None),
                reasoning_tokens=(len(token_ids) - len(final_tokens) - len(close_ids)
                                  if thinking and final_tokens is not None else None),
                rendered_prompt_sha256=__import__('hashlib').sha256(prompt.encode('utf-8')).hexdigest())
            # INSERT, never replace: a saved terminal result is immutable, even invalid output.
            with db:
                db.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (args.model + ':' + item['id'], args.model, item['id'], status, probability, raw,
                     int(mask.sum().item()), len(token_ids), duration / len(batch), now(),
                     json.dumps(metadata, ensure_ascii=False)))
            done.add(item['id'])
        offset += len(batch)
        elapsed = time.monotonic() - started
        completed = len(done) - initial_saved
        atomic_json(run_dir / 'progress.json', {'state': 'running', 'model': args.model,
            'saved': len(done), 'total': len(rows), 'updated_at': now(), 'pid': os.getpid(),
            'elapsed_sec': elapsed, 'samples_per_sec': completed / elapsed,
            'eta_model_sec': (len(rows) - len(done)) * elapsed / completed,
            'actual_batch_size': batch_size, 'peak_gpu_allocated_gb': torch.cuda.max_memory_allocated()/1e9})
        print(json.dumps({'model': args.model, 'saved': len(done), 'total': len(rows),
            'elapsed_sec': round(elapsed, 2), 'time': now()}), flush=True)
        if completed % 128 < len(batch) or len(done) == len(rows):
            import sqlite3
            temp_backup = run_dir / 'results.backup.pending.sqlite'
            backup = sqlite3.connect(temp_backup)
            try:
                db.backup(backup)
            finally:
                backup.close()
            os.replace(temp_backup, run_dir / 'results.backup.sqlite')
        del generated, inputs
    atomic_json(run_dir / 'progress.json', {'state': 'model_complete', 'model': args.model,
        'saved': len(done), 'total': len(rows), 'updated_at': now(), 'pid': os.getpid()})


if __name__ == '__main__':
    main()
