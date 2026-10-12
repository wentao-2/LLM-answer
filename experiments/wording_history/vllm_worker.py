"""Uniform R6 vLLM runner, retaining the original strict numerical readout."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib.metadata,json,os,random,sys,time
import native_worker as w

def run(config_path,model_name):
    config_path=Path(config_path).resolve(); root=config_path.parent
    with w.worker_lock(root/'worker.lock'):
        config,frozen=w.validate_frozen(config_path,running_source=__file__)
        specs=[s for s in config['models'] if s['name']==model_name]; assert len(specs)==1
        spec=specs[0]
        rows=w.select_rows([json.loads(s) for s in (root/config['manifest']).read_text().splitlines()],model_name)
        assert config['dtype']=='bfloat16' and config['seed']==1601
        assert config['backend']=='vllm-0.8.5' and config['temperature']==0
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            VLLM_USAGE_STATS_ENABLED='0',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',OMP_NUM_THREADS='4',
            VLLM_CACHE_ROOT=str(root/'runtime-cache/vllm'),TRITON_CACHE_DIR=str(root/'runtime-cache/triton'),
            TORCHINDUCTOR_CACHE_DIR=str(root/'runtime-cache/torchinductor'))
        from transformers import AutoTokenizer,GenerationConfig
        import torch,vllm
        from vllm import LLM,SamplingParams
        assert vllm.__version__=='0.8.5'
        tokenizer=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True,trust_remote_code=False)
        assert tokenizer.chat_template
        generation=GenerationConfig.from_pretrained(spec['path'],local_files_only=True)
        eos=generation.eos_token_id
        eos_set=set(eos if isinstance(eos,list) else [eos])
        assert eos_set and all(type(n)is int for n in eos_set)
        profiles={};rendered={};tokens={}
        for item in rows:
            profile=w.effective_profile(config,spec,item); profiles[item['id']]=profile
            prompt=tokenizer.apply_chat_template(w.item_messages(item),tokenize=False,add_generation_prompt=True,
                                                 **profile.chat_template_kwargs)
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            assert len(ids)<=profile.max_input_tokens
            w.validate_context_budget(len(ids),profile.max_new_tokens,config['max_model_len'])
            rendered[item['id']]=prompt;tokens[item['id']]=ids
        provenance=w.model_provenance(Path(spec['path']),spec)
        engine=None
        db=w.connect_db(root/'results.sqlite')
        try:
            for key in ('config_sha256','manifest_sha256','source_hashes','protocol_hashes'):
                w.require_metadata(db,key,frozen[key])
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','vllm','tokenizers','huggingface-hub')}
            env=dict(python=sys.version,packages=packages,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
            w.require_metadata(db,'environment',env)
            w.require_metadata(db,'runner_schema','review-vllm-r6-v1')
            w.require_metadata(db,'effective_model_readout',dict(eos_token_ids=sorted(eos_set),
                chat_template_sha256=w.text_digest(w.canonical_json(tokenizer.chat_template))))
            w.require_metadata(db,'model_provenance:'+model_name,provenance)
            remaining,done=w.remaining_rows(db,model_name,rows)
            if not remaining:return
            progress=dict(model=model_name,saved=len(done),total=len(rows),pid=os.getpid(),state='loading')
            w.atomic_json(root/'progress.json',progress)
            engine=LLM(model=spec['path'],tokenizer=spec['path'],dtype='bfloat16',seed=1601,
                tensor_parallel_size=1,trust_remote_code=False,generation_config='vllm',
                max_model_len=config['max_model_len'],max_num_seqs=32,max_num_batched_tokens=8192,
                gpu_memory_utilization=.88,enable_prefix_caching=False,disable_log_stats=True,
                disable_custom_all_reduce=True,enforce_eager=False,kv_cache_dtype='auto')
            random.Random(1601).shuffle(remaining)
            start=time.monotonic();initial=len(done);out_tokens=0
            for offset in range(0,len(remaining),config['chunk_size']):
                batch=remaining[offset:offset+config['chunk_size']]
                params=[]
                for item in batch:
                    p=profiles[item['id']]
                    params.append(SamplingParams(n=1,temperature=0,top_p=1,top_k=-1,min_p=0,
                        repetition_penalty=1,presence_penalty=0,frequency_penalty=0,max_tokens=p.max_new_tokens,
                        stop_token_ids=sorted(eos_set),ignore_eos=False,skip_special_tokens=False,
                        seed=1601,include_stop_str_in_output=True))
                began=time.monotonic()
                outputs=engine.generate([dict(prompt_token_ids=tokens[x['id']]) for x in batch],params,use_tqdm=False)
                duration=time.monotonic()-began
                assert len(outputs)==len(batch)
                for item,result in zip(batch,outputs):
                    assert result.finished and len(result.outputs)==1
                    assert list(result.prompt_token_ids)==tokens[item['id']]
                    output=result.outputs[0];token_ids=list(output.token_ids)
                    profile=profiles[item['id']]
                    status,probability,raw,retained,metadata=w.classify_output(token_ids,tokenizer,eos_set,profile)
                    # No stop strings, guided grammar or synthetic EOS tokens are used.
                    # An unexpected stop without an actual EOS is a technical failure.
                    if output.finish_reason=='stop' and not any(n in eos_set for n in token_ids):
                        raise RuntimeError('Engine stop omitted actual EOS token; readout requires investigation')
                    assert output.finish_reason in ('stop','length')
                    meta=dict(item);meta.update(metadata)
                    meta.update(rendered_prompt=rendered[item['id']],rendered_prompt_sha256=w.text_digest(rendered[item['id']]),
                        generated_token_ids=retained,actual_batch_size=len(batch),backend='vllm-0.8.5',
                        engine_finish_reason=output.finish_reason,engine_stop_reason=output.stop_reason,
                        effective_max_new_tokens=profile.max_new_tokens,effective_chat_template_kwargs=profile.chat_template_kwargs,
                        require_think_close=profile.require_think_close,thinking_enabled=profile.chat_template_kwargs.get('enable_thinking'),
                        readout=profile.readout,readout_version=w.READOUT_VERSIONS[profile.readout],
                        generation_profile=profile.metadata(),model_provenance=provenance)
                    w.insert_result(db,(model_name+':'+item['id'],model_name,item['id'],status,probability,raw,
                        len(tokens[item['id']]),len(retained),duration/len(batch),w.now(),w.canonical_json(meta)))
                    done.add(item['id']);out_tokens+=len(retained)
                elapsed=time.monotonic()-start
                progress.update(saved=len(done),state='running',elapsed_sec=elapsed,output_tokens=out_tokens,
                    samples_per_sec=(len(done)-initial)/elapsed,output_tokens_per_sec=out_tokens/elapsed,updated_at=w.now())
                w.atomic_json(root/'progress.json',progress)
                w.backup_db(db,root)
                print(json.dumps(progress),flush=True)
            w.atomic_json(root/'progress.json',dict(progress,state='model_complete'))
        finally:
            db.close()
            if engine is not None:
                engine.llm_engine.engine_core.shutdown()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--model',required=True)
    args=parser.parse_args();run(args.config,args.model)
