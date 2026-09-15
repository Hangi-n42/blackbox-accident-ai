"""Frozen V5 reserved 00005/00006/00007 CPU preparation and verification ONLY; never reads GT.

Run with .venv/Scripts/python.exe -I -B <this file> prepare|verify --output <new directory>.
This module cannot run inference. The parent uses a separate gated paired evaluator.
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
IDS = ("00005", "00006", "00007")
PACKAGE = ROOT / "artifacts/submissions/verify_v5"
SOURCE = ROOT / "research/v6/nexar_review_candidates"
SELECTION = ROOT / "artifacts/submissions/v5_selection_frozen.json"
POLICY = dict(ids=list(IDS), role="reserved_baseline_after_candidate_frozen", gt_read=False,
    sampling="nearest native PTS to firstPTS + k/10 seconds through lastPTS; earlier tie; unique indices; include first and last",
    sampling_hz=10, image_format="PNG lossless RGB compress_level=1", original_decoded_index_in_filename=True,
    use_entire_source=True, max_calls_per_file=4, max_calls_total=12,
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
    require([row["ID"] for row in frozen["videos"]]==list(IDS),"Only predeclared reserved IDs allowed")
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
        require(slot<4 and self.ledger["call_count"]<12,"Call budget exceeded")
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
    raise RuntimeError("Inference disabled: reserved baseline/candidate execution belongs to the separate parent evaluator after frozen candidate and development gate")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","verify"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,help="Optional root split/protocol artifact, hash only; no GT read")
    args=parser.parse_args()
    if args.command=="prepare":prepare(args.output.resolve(),args.protocol)
    else:
        frozen=verify_freeze(args.output.resolve())
        print(json.dumps({"status":"verified_no_inference","ids":[row["ID"] for row in frozen["videos"]]}))


if __name__=="__main__":main()
