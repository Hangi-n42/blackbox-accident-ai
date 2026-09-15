"""Frozen V5 two-development-video baseline trace; never reads user GT.

Run with .venv/Scripts/python.exe -I -B <this file> prepare|run --output <new directory>.
Only the parent runs GPU inference. No validation IDs or candidate models supported.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
IDS = ("00000", "00003")
PACKAGE = ROOT / "artifacts/submissions/verify_v5"
SOURCE = ROOT / "research/v6/nexar_review_candidates"
SELECTION = ROOT / "artifacts/submissions/v5_selection_frozen.json"
POLICY = dict(ids=list(IDS), role="development_baseline_error_diagnosis_only", gt_read=False,
    sampling="nearest native PTS to firstPTS + k/10 seconds through lastPTS; earlier tie; unique indices; include first and last",
    sampling_hz=10, image_format="PNG lossless RGB compress_level=1", original_decoded_index_in_filename=True,
    use_entire_source=True, max_calls_per_file=4, max_calls_total=8,
    tokens=[64,48,40,40], cpu_threads=2, model="actual extracted V5 saved NF4", candidate=False)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def sampled_indices(times):
    require(times and all(math.isfinite(t) for t in times), "Native PTS missing/nonfinite")
    require(all(b>a for a,b in zip(times,times[1:])), "Non-monotonic PTS")
    indices={0,len(times)-1}
    count=math.floor((times[-1]-times[0])*10+1e-9)
    for k in range(count+1):
        target=times[0]+k/10
        right=bisect_left(times,target)
        choices=[i for i in (right-1,right) if 0<=i<len(times)]
        # Floating-point near ties select earlier rather than drift by rounding.
        best=choices[0]
        for i in choices[1:]:
            a,b=abs(times[i]-target),abs(times[best]-target)
            if a < b and not math.isclose(a,b,rel_tol=0,abs_tol=1e-12):best=i
        indices.add(best)
    return sorted(indices)


def package_binding():
    selected=read(SELECTION)
    require(selected["stage2_module"]=="stage2_motion_collision", "Wrong frozen V5 Stage2 policy")
    expected={name:digest for name,digest in selected["expected_archive_sha256"].items()
              if name.startswith("model/stage2/") or name=="inference.py"}
    actual={name:sha(PACKAGE/name) for name in expected}
    require(actual==expected,"Actual V5 extracted source/weights differ from frozen selection")
    return dict(package=str(PACKAGE),selection_sha256=sha(SELECTION),files=actual,
                runner_sha256=sha(__file__))


def prepare(output, protocol):
    import av
    from PIL import Image
    require(not output.exists(),"Output exists; preparation refuses overwrite")
    binding=package_binding()
    output.mkdir(parents=True)
    records=[]
    for ID in IDS:
        source=SOURCE/f"{ID}.mp4"  # No directory enumeration or GT/annotation reads.
        source_sha=sha(source)
        native=[]
        with av.open(str(source)) as container:
            container.streams.video[0].thread_count=2
            for index,frame in enumerate(container.decode(video=0)):
                require(frame.pts is not None and frame.time_base is not None,"Source lacks native timestamp")
                native.append(dict(frame=index,pts_seconds=float(frame.pts*frame.time_base),native_pts=frame.pts,
                    time_base=[frame.time_base.numerator,frame.time_base.denominator]))
        times=[row["pts_seconds"] for row in native]
        indices=sampled_indices(times); chosen=set(indices)
        folder=output/"inputs/images"/ID
        folder.mkdir(parents=True)
        images=[]
        with av.open(str(source)) as container:
            container.streams.video[0].thread_count=2
            for index,frame in enumerate(container.decode(video=0)):
                require(float(frame.pts*frame.time_base)==times[index],"Decode PTS changed between passes")
                if index not in chosen:continue
                image=frame.to_image().convert("RGB")
                path=folder/f"frame_{index:06d}.png"
                image.save(path,compress_level=1)
                images.append(dict(frame=index,pts_seconds=times[index],path=path.relative_to(output).as_posix(),
                    size=list(image.size),file_sha256=sha(path),rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        require([x["frame"] for x in images]==indices,"Incomplete sampled image export")
        require(sha(source)==source_sha,"Source changed during decode")
        row=dict(ID=ID,source_path=str(source),source_sha256=source_sha,source_frame_pts=native,
            input_images=images,selected_frame_pts=[dict(frame=x["frame"],pts_seconds=x["pts_seconds"]) for x in images],
            source_frame_count=len(native),sampled_count=len(images),duration_seconds=times[-1]-times[0],
            max_sample_gap_seconds=max((b["pts_seconds"]-a["pts_seconds"] for a,b in zip(images,images[1:])),default=0))
        row["input_manifest_sha256"]=hashlib.sha256(json.dumps(images,sort_keys=True).encode()).hexdigest()
        records.append(row)
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),policy=POLICY,binding=binding,videos=records,
        protocol_artifact=None if protocol is None else dict(path=str(protocol.resolve()),sha256=sha(protocol)),
        versions={name:importlib.metadata.version(name) for name in ("av","pillow","numpy","torch","transformers","bitsandbytes")})
    # Optional root protocol is hashed only, never parsed for GT or validation labels.
    write_new(output/"freeze.json",freeze)
    print(json.dumps({"status":"prepared_no_inference","freeze_sha256":sha(output/"freeze.json"),
        "ids":list(IDS),"sampled_counts":[r["sampled_count"] for r in records]}),flush=True)


def verify_freeze(output):
    frozen=read(output/"freeze.json")
    require(frozen["policy"]==POLICY,"Policy changed after input freeze")
    require([row["ID"] for row in frozen["videos"]]==list(IDS),"Only two declared development IDs allowed")
    require(frozen["binding"]==package_binding(),"Package or runner changed after freeze")
    for video in frozen["videos"]:
        require(sha(video["source_path"])==video["source_sha256"],"Source bytes changed")
        for image in video["input_images"]:
            require(sha(output/image["path"])==image["file_sha256"],"Frozen image changed")
    if frozen["protocol_artifact"]:
        p=frozen["protocol_artifact"]
        require(sha(p["path"])==p["sha256"],"Root protocol changed")
    return frozen


class Recorder:
    def __init__(self,model,folder,ledger):
        self.model,self.folder,self.ledger=model,folder,ledger
        self.calls=[];self.overhead=0.

    def ask(self,images,prompt,max_new_tokens=128):
        slot=len(self.calls)
        require(slot<4 and self.ledger["call_count"]<8,"Call budget exceeded")
        require(max_new_tokens==POLICY["tokens"][slot],"Frozen V5 token policy differs")
        start=time.perf_counter()
        folder=self.folder/f"call_{slot+1}";folder.mkdir()
        row=dict(number=slot+1,prompt=prompt,max_new_tokens=max_new_tokens,status="pending",input_images=[])
        budget=max(1024,self.model.pixel_budget//len(images))
        for i,image in enumerate(images):
            rgb=image.convert("RGB")
            rgb.save(folder/f"input_{i}.png")
            scale=min(1.,math.sqrt(budget/(rgb.width*rgb.height)))
            width=max(32,int(rgb.width*scale)//32*32);height=max(32,int(rgb.height*scale)//32*32)
            row["input_images"].append(dict(size=list(rgb.size),bounded_size=[width,height],
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        candidates=re.findall(r"(?:Available frames|Allowed frames|Lane entry candidates):\s*(\[[^\]]*\])",prompt)
        row["offered_original_numbers"]=json.loads(candidates[0]) if candidates else None
        row["bounded_pixel_total"]=sum(x["bounded_size"][0]*x["bounded_size"][1] for x in row["input_images"])
        self.model.torch.cuda.synchronize();self.model.torch.cuda.reset_peak_memory_stats()
        self.overhead+=time.perf_counter()-start
        self.calls.append(row);self.ledger["call_count"]+=1
        start=time.perf_counter()
        try:
            answer=self.model.ask(images,prompt,max_new_tokens=max_new_tokens)
            self.model.torch.cuda.synchronize()
            row.update(status="complete",raw_output=answer)
            return answer
        except BaseException as error:
            row.update(status="failed",error=repr(error));raise
        finally:
            row.update(seconds=time.perf_counter()-start,peak_allocated_bytes=self.model.torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=self.model.torch.cuda.max_memory_reserved())
            start=time.perf_counter();write_new(folder/"record.json",row)
            self.overhead+=time.perf_counter()-start
            print(json.dumps({"call":slot+1,"folder":str(self.folder),"raw":row.get("raw_output"),"seconds":row["seconds"]}),flush=True)


def run(output):
    require(sys.flags.isolated,"Use Python -I for actual package import isolation")
    frozen=verify_freeze(output)
    report_path=output/"report.json"
    require(not report_path.exists() and not (output/"traces").exists(),"Prior run exists; no overwrite/retry")
    code=PACKAGE/"model/stage2/code"
    require(not any(n=="solution" or n.startswith("solution.") for n in sys.modules),"Workspace solution already imported")
    sys.path.insert(0,str(code))
    report=dict(status="running",created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Two development sources, baseline-only error diagnosis; no accuracy/adoption claim",
        freeze_sha256=sha(output/"freeze.json"),call_count=0,network_attempts=0,videos=[])
    def deny(*args,**kwargs):
        report["network_attempts"]+=1
        raise RuntimeError("Network blocked in frozen V5 inference")
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    try:
        import cv2
        cv2.setNumThreads(2)
        from solution import stage2_motion_collision as baseline
        for name,module in list(sys.modules.items()):
            if name=="solution" or name.startswith("solution."):
                require(Path(module.__file__).resolve().is_relative_to(code.resolve()),f"Non-package import: {name}")
        report["imported_solution"]={name:str(module.__file__) for name,module in sys.modules.items() if name=="solution" or name.startswith("solution.")}
        with baseline.CandidateVLM(PACKAGE/"model/stage2/vlm",precision="nf4") as model:
            model.torch.set_num_threads(2)
            report["runtime"]=model.candidate_metadata
            for video in frozen["videos"]:
                ID=video["ID"]
                paths=[output/x["path"] for x in video["input_images"]]
                start=time.perf_counter();valid,scores,_=baseline.base._motion_scan(paths)
                scan_seconds=time.perf_counter()-start
                folder=output/"traces"/ID;folder.mkdir(parents=True)
                rec=Recorder(model,folder,report)
                start=time.perf_counter();prediction,diagnostics=baseline._predict_file(valid,scores,rec)
                elapsed=time.perf_counter()-start
                require(len(rec.calls)==4,"Frozen V5 four-call contract violated")
                numbers=[baseline.base._frame_number(p) for p in valid]
                internal=diagnostics["collision_replacement"]["base_collision_frame"]
                center=numbers.index(internal)
                rec.calls[3]["offered_original_numbers"]=[numbers[i] for i in sorted(set([max(0,center-2),center,min(len(numbers)-1,center+2)]))]
                write_new(folder/"call_4/offered_context.json",dict(
                    offered_original_numbers=rec.calls[3]["offered_original_numbers"],
                    source="Reconstructed from actual frozen base internal collision and valid path indices after all four asks",
                    internal_collision_frame=internal))
                row=dict(ID=ID,prediction=prediction,diagnostics=diagnostics,calls=rec.calls,
                    input_manifest_sha256=video["input_manifest_sha256"],frame_numbers=numbers,
                    original_frame_numbers=[x["frame"] for x in video["input_images"]],
                    selected_frame_pts=video["selected_frame_pts"],motion_scores=scores.tolist(),
                    internal_collision_frame=internal,final_motion_collision_frame=prediction["collision_frame"],
                    entry_prefix=numbers[:center+1],scan_seconds=scan_seconds,predict_elapsed_seconds=elapsed,
                    logger_overhead_seconds=rec.overhead,predict_compute_seconds=elapsed-rec.overhead)
                write_new(folder/"trace.json",row);report["videos"].append(row)
        require(report["call_count"]==8 and report["network_attempts"]==0,"Offline/call budget contract failed")
        verify_freeze(output)
        report["status"]="complete"
    except BaseException as error:
        report.update(status="failed",error=repr(error));raise
    finally:
        write_new(report_path,report)
    print(json.dumps({"status":report["status"],"call_count":report["call_count"],"ids":list(IDS)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,help="Optional root split/protocol artifact, hash only; no GT read")
    args=parser.parse_args()
    if args.command=="prepare":prepare(args.output.resolve(),args.protocol)
    else:run(args.output.resolve())


if __name__=="__main__":main()
