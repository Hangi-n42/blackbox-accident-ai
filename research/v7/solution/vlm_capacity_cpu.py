"""Contingency 8B loader: CPU-resident FP16 token lookup, original ask().

No root GPU device-map entry: avoid root safetensor GPU staging and recursive
root placement of the CPU embedding. This module performs no work on import.
"""
from pathlib import Path
import hashlib,importlib.metadata,importlib.util,json,time
from .vlm import LocalVLM

REPOSITORY='Qwen/Qwen3-VL-8B-Instruct'
REVISION='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
EMBEDDING='model.language_model.embed_tokens'
DEVICE_MAP={'model.visual':0,EMBEDDING:'cpu','model.language_model.layers':0,
            'model.language_model.norm':0,'model.language_model.rotary_emb':0,'lm_head':0}
EXPECTED_VERSIONS={'torch':'2.8.0+cu128','transformers':'4.57.6','accelerate':'1.9.0','bitsandbytes':'0.48.1'}

def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def quantization_guard(q):
    require(q.get('quant_method')=='bitsandbytes' and q.get('load_in_4bit',q.get('_load_in_4bit')) is True,'Stored bitsandbytes4bit required')
    require(q.get('bnb_4bit_quant_type')=='nf4' and q.get('bnb_4bit_compute_dtype')=='float16','Stored NF4/FP16 required')
    require(q.get('bnb_4bit_use_double_quant') is False,'Double quantization is outside this policy')
    require(q.get('llm_int8_skip_modules') in (None,[]),'Stored skipped Linear modules not supported')
    require(q.get('bnb_4bit_quant_storage','uint8')=='uint8','Stored quantized storage differs')
    # Transformers4.57.6 ignores BitsAndBytes overrides for an already quantized
    # config. Do not pretend a runtime config safely enables this missing flag.
    require(q.get('llm_int8_enable_fp32_cpu_offload') is True,
            'Saved checkpoint lacks explicit CPU-offload flag. This loader will not silently edit or override its quantization config; reload a derivative saved under this policy.')
def config_guard(config):
    t=config.get('text_config',{})
    require(config.get('model_type')=='qwen3_vl' and config.get('tie_word_embeddings') is False,'Untied Qwen3VL required')
    require(t.get('vocab_size')==151936 and t.get('hidden_size')==4096,'Pinned8B architecture differs')
    if config.get('quantization_config'):quantization_guard(config['quantization_config'])
def coverage(keys):
    counts={p:0 for p in DEVICE_MAP}
    for key in keys:
        matches=[p for p in DEVICE_MAP if key==p or key.startswith(p+'.')]
        require(len(matches)==1,'Uncovered/overlapping state key: '+key);counts[matches[0]]+=1
    require(counts[EMBEDDING]==1,'Expected only the input embedding weight on CPU')
    return counts
def provenance(path,stored):
    manifest_path=path/('EXPORT_MANIFEST.json' if stored else 'download_manifest.json')
    manifest=read(manifest_path)
    require(manifest.get('repository')==REPOSITORY and manifest.get('revision')==REVISION,'Pinned official8B provenance required')
    require(manifest.get('status')==('saved_unverified' if stored else 'complete'),'Incomplete source/checkpoint')
    require(not (path/'EXPORT_INCOMPLETE.json').exists(),'Incomplete derivative')
    files=manifest.get('files',[]);require(bool(files),'Empty provenance files')
    seen=set()
    # Caller performs full weight hashes; constructor checks bound metadata and
    # complete file sizes without a second 17.5GB scan during a memory trial.
    for row in files:
        name=row['name'];p=(path/name).resolve()
        require(p.is_relative_to(path) and name not in seen,'Unsafe/duplicate manifest member');seen.add(name)
        require(p.is_file() and p.stat().st_size==row['bytes'],'Manifest payload size differs: '+name)
        if name=='config.json':require(sha(p)==row['sha256'],'Manifest config differs')
        if not stored and name.endswith('.safetensors'):
            require(row.get('expected_lfs_sha256')==row['sha256'],'Unverified source shard')
    index=read(path/'model.safetensors.index.json') if (path/'model.safetensors.index.json').exists() else None
    if index:require(set(index['weight_map'].values())<=seen,'Unmanifested shard reference')
    else:require(stored and 'model.safetensors' in seen,'No serialized model weights')
    return manifest_path,index

class CandidateVLM(LocalVLM):
    def __init__(self,model_path,*,pixel_budget=1_200_000,precision='nf4',gpu_memory=None,cpu_memory=None):
        require(precision=='nf4' and pixel_budget==1_200_000,'Fixed NF4 and1.2MP policy required')
        self.model=None;self.processor=None
        self.model_path=Path(model_path).resolve();saved=read(self.model_path/'config.json');config_guard(saved)
        stored=bool(saved.get('quantization_config'));manifest,index=provenance(self.model_path,stored)
        versions={k:importlib.metadata.version(k) for k in EXPECTED_VERSIONS}
        require(versions==EXPECTED_VERSIONS,'Contingency loader was reviewed only for fixed runtime versions')
        import torch
        import bitsandbytes as bnb
        from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
        from accelerate.hooks import remove_hook_from_module
        require(torch.cuda.is_available(),'CUDA required for vision/decoder/lm_head')
        torch.set_num_threads(2);self.torch=torch;self.device='cuda';self.pixel_budget=pixel_budget
        kwargs=dict(local_files_only=True,trust_remote_code=False,dtype=torch.float16,
                    attn_implementation='sdpa',low_cpu_mem_usage=True,device_map=dict(DEVICE_MAP))
        if not stored:
            kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=False,llm_int8_enable_fp32_cpu_offload=True)
        # Saved NF4 configuration is reused verbatim; no fresh config is passed.
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        try:
            self.processor=AutoProcessor.from_pretrained(str(self.model_path),local_files_only=True,trust_remote_code=False)
            self.model=Qwen3VLForConditionalGeneration.from_pretrained(str(self.model_path),**kwargs).eval()
            torch.cuda.synchronize();dispatch_seconds=time.perf_counter()-started
            dispatch_peak_allocated=torch.cuda.max_memory_allocated();dispatch_peak_reserved=torch.cuda.max_memory_reserved()
            config_guard(self.model.config.to_dict());state_coverage=coverage(self.model.state_dict().keys())
            original=self.model.get_input_embeddings();require(isinstance(original,torch.nn.Embedding),'Unexpected input embedding class')
            hook=getattr(original,'_hf_hook',None);require(hook is not None,'Expected embedding offload hook absent')
            hooks=getattr(hook,'hooks',(hook,));offload_hooks=[h for h in hooks if getattr(h,'offload',False)]
            require(len(offload_hooks)==1,'Ambiguous embedding offload hook')
            h=offload_hooks[0]
            require(set(h.original_devices)=={'weight'} and str(h.original_devices['weight'])=='cpu','Hook would restore embedding outside CPU')
            require(str(h.weights_map['weight'].dtype)=='torch.float16','CPU backing embedding is not FP16')
            hook_record={'type':type(h).__name__,'execution_device':str(h.execution_device),'original_devices':{k:str(v) for k,v in h.original_devices.items()}}
            remove_hook_from_module(original,recurse=False)
            require(original.weight.device.type=='cpu' and original.weight.dtype==torch.float16 and not hasattr(original,'_hf_hook'),'Embedding hook did not restore real CPU FP16 weight')
            helper=Path(__file__).with_name('cpu_embedding.py');spec=importlib.util.spec_from_file_location('capacity_cpu_embedding_policy',helper)
            policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
            keys_before=set(self.model.state_dict());values_before=original.weight.detach()
            policy_metadata=policy.install(self.model)
            embedding=self.model.get_input_embeddings()
            require(set(self.model.state_dict())==keys_before and torch.equal(values_before,embedding.weight),'Policy changed state keys/embedding values')
            del values_before,original,hook,h,offload_hooks,hooks
            parameters=[];linear_inventory=[];plain_linear=[];bad=[];non_fp16=[]
            for name,param in self.model.named_parameters():
                expected='cpu' if name==EMBEDDING+'.weight' else 'cuda'
                if param.device.type!=expected:bad.append({'name':name,'device':str(param.device)})
                if param.is_floating_point() and param.dtype!=torch.float16:non_fp16.append({'name':name,'dtype':str(param.dtype)})
                parameters.append({'name':name,'shape':list(param.shape),'dtype':str(param.dtype),'device':str(param.device)})
            require(not bad and not non_fp16,'Unexpected parameter placement/dtype: '+repr((bad,non_fp16)))
            bad_buffers=[(n,str(b.device)) for n,b in self.model.named_buffers() if b.device.type!='cuda']
            require(not bad_buffers,'Unexpected CPU/meta registered buffer: '+repr(bad_buffers))
            for name,module in self.model.named_modules():
                if isinstance(module,bnb.nn.Linear4bit):
                    require(module.compute_dtype==torch.float16 and module.weight.quant_type=='nf4' and not module.weight.compress_statistics,'Linear4bit precision changed: '+name)
                    require(module.weight.quant_state is not None and not module.weight.quant_state.nested,'Missing/double quantization state: '+name)
                    linear_inventory.append({'name':name,'in_features':module.in_features,'out_features':module.out_features,
                        'packed_shape':list(module.weight.shape),'packed_dtype':str(module.weight.dtype),'device':str(module.weight.device)})
                elif isinstance(module,torch.nn.Linear):plain_linear.append(name)
            require(linear_inventory and plain_linear==['lm_head'],'Unintended unquantized Linear modules: '+repr(plain_linear))
            head=self.model.get_output_embeddings();require(head.weight.device.type=='cuda' and head.weight.dtype==torch.float16,'Output head placement/dtype differs')
            require(embedding.weight.device.type=='cpu' and embedding.weight.dtype==torch.float16,'Input embedding policy differs')
            torch.cuda.synchronize()
            self.candidate_metadata={'precision':'nf4','compute_dtype':'float16','attention':'sdpa','double_quant':False,
                'prequantized_checkpoint':stored,'quantization_config_source':'saved_verbatim' if stored else 'explicit_nf4_fp16_cpu_embedding_offload',
                'cpu_offload_permission':True,'initial_device_map':dict(DEVICE_MAP),'device_map':{k:str(v) for k,v in self.model.hf_device_map.items()},
                'embedding_policy':policy_metadata,'removed_embedding_hook':hook_record,'parameter_inventory':parameters,
                'linear4bit_count':len(linear_inventory),'linear4bit_inventory':linear_inventory,'plain_linear_names':plain_linear,
                'all_non_embedding_parameters_cuda':True,'all_floating_parameters_fp16':True,'all_registered_buffers_cuda':True,
                'state_coverage':state_coverage,'source_tensor_coverage':None if index is None else coverage(index['weight_map']),
                'dispatch_seconds':dispatch_seconds,'model_load_seconds':time.perf_counter()-started,
                'dispatch_peak_allocated_bytes':dispatch_peak_allocated,'dispatch_peak_reserved_bytes':dispatch_peak_reserved,
                'load_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'load_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'loaded_allocated_bytes':torch.cuda.memory_allocated(),'loaded_reserved_bytes':torch.cuda.memory_reserved(),
                'model_memory_footprint_bytes':self.model.get_memory_footprint(),'versions':versions,
                'repository':REPOSITORY,'revision':REVISION,'manifest_sha256':sha(manifest),'cpu_policy_sha256':sha(helper),
                'loader_sha256':sha(Path(__file__)),'ask_implementation':'inherited unchanged from solution.vlm.LocalVLM.ask',
                'limitations':'No accuracy guarantee; caller must verify source hashes, raw equivalence, full-budget memory/time and fresh saved reload.'}
        except BaseException:
            self.close();raise
