"""Fixed Qwen native-video input; synthetic playback is never physical time."""
import hashlib,math,re,time
import numpy as np
from PIL import Image

PREFIX=('Frames are ordered chronologically. The playback timestamps are synthetic and do not represent physical capture time; '
        'answer using the original frame numbers printed on the images.\n')
PIXEL_BUDGET=1_200_000
SYNTHETIC_FPS=2.0

def require(ok,msg):
    if not ok:raise ValueError(msg)
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def prepare_native_inputs(vlm,labeled_tiles,prompt):
    """CPU processor path exposed for contract tests; never invokes a model."""
    start=time.perf_counter();torch=vlm.torch
    require(vlm.pixel_budget==PIXEL_BUDGET,'Frozen1.2MP input budget required')
    require(isinstance(prompt,str) and bool(prompt),'Nonempty frozen task prompt required')
    tiles=list(labeled_tiles);K=len(tiles)
    require(1<=K<=18,'Expected1..18 offered labeled tiles')
    require(all(isinstance(t,Image.Image) and t.size==(384,256) for t in tiles),'Every offered tile must be original384x256 sheet tile, including its printed frame label')
    vp=vlm.processor.video_processor
    require(vp.patch_size==16 and vp.merge_size==2 and vp.temporal_patch_size==2,'Frozen Qwen video patch geometry differs')
    encoded_K=2*((K+1)//2);budget=max(32*32,PIXEL_BUDGET//encoded_K)
    bounded=[];tile_hashes=[]
    for image in tiles:
        image=image.convert('RGB');tile_hashes.append(array_hash(np.asarray(image)))
        scale=min(1.0,math.sqrt(budget/(image.width*image.height)))
        w=max(32,int(image.width*scale)//32*32);h=max(32,int(image.height*scale)//32*32)
        bounded.append(image.resize((w,h),Image.Resampling.BICUBIC))
    require(len({im.size for im in bounded})==1,'Video tensor requires identical spatial shapes')
    w,h=bounded[0].size;require(encoded_K*w*h<=PIXEL_BUDGET,'Padding-inclusive pixel budget exceeded')
    array=np.stack([np.asarray(im) for im in bounded]);tensor=torch.from_numpy(array.transpose(0,3,1,2).copy())
    require(tensor.dtype==torch.uint8 and list(tensor.shape)==[K,3,h,w],'Expected CPU uint8 TCHW tensor')
    # _calculate_timestamps extends odd indices in place; preserve this independent
    # diagnostic snapshot and hand the processor a fresh list.
    original_metadata={'fps':SYNTHETIC_FPS,'frames_indices':list(range(K)),'total_num_frames':K}
    metadata={**original_metadata,'frames_indices':list(range(K))}
    text=vlm.processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':PREFIX+prompt}]}],
                                         tokenize=False,add_generation_prompt=True)
    inputs=vlm.processor(text=[text],videos=[tensor],return_tensors='pt',
        videos_kwargs={'video_metadata':[metadata],'do_sample_frames':False,'do_resize':False,
                       'input_data_format':'channels_first','device':'cpu'})
    require('pixel_values_videos' in inputs and 'video_grid_thw' in inputs and 'pixel_values' not in inputs,'Actual video processor path required')
    grid=inputs['video_grid_thw'].tolist();expected_grid=[[encoded_K//2,h//16,w//16]]
    require(grid==expected_grid,'Processor sampled/resized or temporal patch count differs')
    pixels=inputs['pixel_values_videos'];require(list(pixels.shape)==[math.prod(grid[0]),1536],'Unexpected video patch tensor shape')
    video_id=vlm.processor.tokenizer.convert_tokens_to_ids(vlm.processor.video_token)
    video_tokens=int((inputs['input_ids']==video_id).sum().item());require(video_tokens==math.prod(grid[0])//4,'Video placeholder token count mismatch')
    decoded=vlm.processor.tokenizer.decode(inputs['input_ids'][0],skip_special_tokens=False)
    timestamps=re.findall(r'<([0-9]+(?:\.[0-9]+)?) seconds>',decoded)
    padded_indices=list(range(K))+([K-1] if K%2 else [])
    expected_timestamps=[f'{(padded_indices[i]+padded_indices[i+1])/(2*SYNTHETIC_FPS):.1f}' for i in range(0,encoded_K,2)]
    require(timestamps==expected_timestamps,'Synthetic timestamp encoding differs')
    diagnostics={'input_policy':'native video TCHW; synthetic2fps playback, not source/capture FPS','synthetic_prefix':PREFIX,
        'source_FPS_or_GT_argument_used':False,'original_candidate_count':K,'encoded_frame_count':encoded_K,'temporal_padding_frames':encoded_K-K,
        'temporal_padding_policy':'processor repeats last frame once for oddK; no offered candidate discarded',
        'metadata_before_processor':original_metadata,'processor_synthetic_timestamp_strings':timestamps,
        'original_tile_size':[384,256],'bounded_tile_size':[w,h],'budget_divisor':encoded_K,'pixel_budget':PIXEL_BUDGET,
        'original_candidate_pixels':K*w*h,'encoded_pixels_including_padding':encoded_K*w*h,'video_tensor_shape':list(tensor.shape),
        'video_tensor_dtype':str(tensor.dtype),'video_grid_thw':grid,'pixel_values_videos_shape':list(pixels.shape),
        'pixel_values_videos_dtype':str(pixels.dtype),'input_tokens':int(inputs['input_ids'].shape[1]),'video_placeholder_tokens':video_tokens,
        'input_ids_sha256':array_hash(inputs['input_ids'].numpy()),'input_tile_RGB_sha256':tile_hashes,
        'bounded_tile_RGB_sha256':[array_hash(np.asarray(im)) for im in bounded],
        'processor_output_tensor_bytes':sum(v.numel()*v.element_size() for v in inputs.values() if hasattr(v,'numel')),
        'processor_seconds':time.perf_counter()-start,
        'comparison_limit':'Spatial resize, pixel/token counts, synthetic timestamps and temporal patching differ from earlier sheet/multi-image. Not a pure packing comparison.'}
    return inputs,diagnostics

def ask_native_video(vlm,labeled_tiles,prompt,max_new_tokens=48):
    """Return (raw, diagnostics); original frame numbers remain labeled pixels."""
    require(type(max_new_tokens) is int and max_new_tokens==48,'Frozen48 generation-token budget required')
    start=time.perf_counter();inputs,diag=prepare_native_inputs(vlm,labeled_tiles,prompt);torch=vlm.torch
    cuda=str(vlm.device).startswith('cuda')
    if cuda:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        diag['cuda_allocated_before_input_bytes']=torch.cuda.memory_allocated()
    inputs=inputs.to(vlm.device);generation_start=time.perf_counter()
    with torch.inference_mode():
        generated=vlm.model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True)
    if cuda:torch.cuda.synchronize()
    new_tokens=generated[:,inputs['input_ids'].shape[1]:]
    raw=vlm.processor.batch_decode(new_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
    diag.update(generation_seconds=time.perf_counter()-generation_start,generated_token_count=int(new_tokens.shape[1]),
        generated_token_ids=new_tokens[0].detach().cpu().tolist(),max_new_tokens=max_new_tokens,
        total_seconds=time.perf_counter()-start,do_sample=False,use_cache=True,original_ids_recovered_from_synthetic_time=False)
    if cuda:
        diag.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cuda_allocated_after_generation_bytes=torch.cuda.memory_allocated(),peak_scope='Peak counters reset for this call after CPU processing; includes resident model. Aggregate all call peaks for run peak.')
    del generated,new_tokens,inputs
    return raw,diag
