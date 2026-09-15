"""Frozen V3/V5 paired representation audit; five source groups, not 15 samples."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import cv2
import numpy as np
from PIL import Image
from solution import stage2_motion_collision as baseline
from solution import stage2_v5 as candidate

SETS = {"canonical": "research/v5_inputs/canonical/images",
        "jpeg": "artifacts/public_eval/stage2/images",
        "mirror": "research/v5_inputs/mirror/images"}
MODEL = ROOT/"artifacts/candidates/qwen3_vl_4b_nf4"
FREEZE = ROOT/"research/v5_stage2/gpu_freeze.json"
OUTPUT = ROOT/"research/v5_stage2/paired_run"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def input_manifest():
    manifests = {}
    for name, relative in SETS.items():
        rows = []
        for folder in sorted((ROOT/relative).iterdir()):
            if not folder.is_dir(): continue
            for path in sorted(folder.iterdir(), key=candidate.base._frame_number):
                if path.suffix.lower() not in candidate.base.IMAGE_EXTENSIONS: continue
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    rows.append(dict(path=path.relative_to(ROOT).as_posix(), ID=folder.name,
                        original_number=candidate.base._frame_number(path), size=list(rgb.size),
                        file_sha256=sha(path), rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        manifests[name] = rows
    return manifests


def source_manifest():
    names = ["research/v5_stage2/design.md", "research/v5_stage2/cpu_contract.json",
             "scripts/evaluate_stage2_v5.py", "scripts/test_stage2_v5.py", "solution/vlm.py",
             "solution/vlm_candidate.py"] + [p.relative_to(ROOT).as_posix() for p in (ROOT/"solution").glob("stage2*.py")]
    return {name: sha(ROOT/name) for name in names}


def freeze():
    if FREEZE.exists(): raise FileExistsError("GPU freeze already exists")
    contract = json.loads((ROOT/"research/v5_stage2/cpu_contract.json").read_text())
    assert contract["passed"] and contract["candidate_sha256"] == sha(ROOT/"solution/stage2_v5.py")
    export = json.loads((MODEL/"EXPORT_MANIFEST.json").read_text())
    weights = {item["name"]: sha(MODEL/item["name"]) for item in export["files"]}
    assert all(weights[item["name"]] == item["sha256"] for item in export["files"])
    frozen = dict(created_utc=datetime.now(timezone.utc).isoformat(), gpu_results_seen=False,
        sources=source_manifest(), input_sets=input_manifest(), model=weights,
        model_manifest_sha256=sha(MODEL/"EXPORT_MANIFEST.json"), model_path=MODEL.as_posix(),
        order="canonical,jpeg,mirror; sorted source; V3 then V5 per identical input",
        budget=dict(max_clips=30,max_calls=120,threads=2,pixel_budget=1200000,
                    v5_tokens=list(candidate.TOKENS),tracking_steps=candidate.TRACK_STEP_BUDGET),
        gates=dict(per_representation_predictor_ratio_max=1.25,max_vram_gib=7.5,
            known_regression="No regression on already documented unambiguous source-group observations; no unofficial label presented as official GT",
            timing_scope="predict_file elapsed minus measured logger image/hash/file overhead, includes render/tracking/VLM; no 60-minute guarantee"),
        no_external_inference=True, source_groups=5, derived_sets_not_independent=True)
    write(FREEZE,frozen)
    print(json.dumps({"frozen":str(FREEZE),"sha256":sha(FREEZE)}))


class Recorder:
    def __init__(self, model, path, report):
        self.model,self.path,self.report = model,path,report
        self.calls=[]
        self.overhead=0.

    def ask(self, images, prompt, max_new_tokens=128):
        if self.report["call_count"] >= 120 or len(self.calls) >= 4:
            raise RuntimeError("Predeclared call budget exhausted")
        start = time.perf_counter()
        folder=self.path/f"call_{len(self.calls)+1}"
        folder.mkdir(parents=True,exist_ok=False)
        row=dict(prompt=prompt,max_new_tokens=max_new_tokens,input_images=[],status="running")
        budget=self.model.pixel_budget//len(images)
        for index,img in enumerate(images):
            rgb=img.convert("RGB")
            rgb.save(folder/f"input_{index}.png")
            scale=min(1.,math.sqrt(budget/(rgb.width*rgb.height)))
            width=max(32,int(rgb.width*scale)//32*32)
            height=max(32,int(rgb.height*scale)//32*32)
            row["input_images"].append(dict(size=list(rgb.size),bounded_size=[width,height],
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        row["bounded_pixel_total"]=sum(x["bounded_size"][0]*x["bounded_size"][1] for x in row["input_images"])
        assert row["bounded_pixel_total"] <= self.model.pixel_budget
        self.model.torch.cuda.synchronize()
        self.model.torch.cuda.reset_peak_memory_stats()
        write(folder/"result.json",row)
        self.overhead+=time.perf_counter()-start
        self.calls.append(row)
        self.report["call_count"]+=1
        start=time.perf_counter()
        try:
            answer=self.model.ask(images,prompt,max_new_tokens=max_new_tokens)
            self.model.torch.cuda.synchronize()
            row.update(raw_output=answer,status="complete")
            return answer
        finally:
            row.update(seconds=time.perf_counter()-start,
                peak_allocated_bytes=self.model.torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=self.model.torch.cuda.max_memory_reserved())
            start=time.perf_counter()
            write(folder/"result.json",row)
            print(json.dumps({"path":self.path.as_posix(),"call":len(self.calls),"seconds":row["seconds"],"raw":row.get("raw_output")}),flush=True)
            self.overhead+=time.perf_counter()-start


def summarize(report, frozen):
    # Only after all inference: official PTS/contact GT, no GT in predictors.
    import av
    import pandas as pd
    labels=pd.read_csv(ROOT/"Baseline/data/stage2/labels.csv")
    time_maps={}
    gt={}
    for row in labels.itertuples():
        with av.open(str(ROOT/"Baseline/data/stage2"/row.path)) as container:
            time_maps[row.ID]=[float(f.pts*f.time_base) for f in container.decode(video=0)]
        gt[row.ID]=int(row.t_collision)
    summaries={}
    for rep,arms in report["sets"].items():
        summaries[rep]={}
        for arm,rows in arms.items():
            correct=0
            for row in rows:
                times=time_maps[row["ID"]]; truth=gt[row["ID"]]
                pred=row["prediction"]["collision_frame"]
                error=abs(times[pred]-times[truth])
                correct+=error <= .3+1e-9
                row["official_collision"]={"provided_frame":truth,"error_seconds":error,"within_0_3":error <= .3+1e-9}
                d=row["diagnostics"]
                pools={"final_motion_singleton":[pred]}
                if arm=="V3": pools.update(collision_vlm_candidates=d["collision_candidates"])
                else: pools.update(identity_candidates=[x["frame"] for x in d["render"][0]["tiles"]])
                row["collision_oracle_coverage"]={k:dict(nearest_seconds=min(abs(times[v]-times[truth]) for v in values),
                    within_0_3=any(abs(times[v]-times[truth]) <= .3+1e-9 for v in values)) for k,values in pools.items()}
                row["official_entry_side_space_gt_available"]=False
            summaries[rep][arm]=dict(clips=len(rows),collision_correct=int(correct),
                predictor_compute_seconds=sum(x["predict_compute_seconds"] for x in rows),
                scan_seconds=sum(x["scan_seconds"] for x in rows),
                peak_vram_bytes=max(c["peak_reserved_bytes"] for r in rows for c in r["calls"]))
        ratio=summaries[rep]["V5"]["predictor_compute_seconds"]/summaries[rep]["V3"]["predictor_compute_seconds"]
        summaries[rep]["timing_ratio"]=ratio
        summaries[rep]["timing_gate_pass"]=ratio <= frozen["gates"]["per_representation_predictor_ratio_max"]
        for a,b in zip(arms["V3"],arms["V5"]):
            assert a["input_manifest_sha256"]==b["input_manifest_sha256"]
            assert a["prediction"]["collision_frame"]==b["prediction"]["collision_frame"]
    report["summary"]=summaries
    report["source_pts_seconds"]=time_maps
    report["mirror_diagnostics"]={}
    for arm in ("V3","V5"):
        rows=[]
        for a,b in zip(report["sets"]["canonical"][arm],report["sets"]["mirror"][arm]):
            p,q=a["prediction"],b["prediction"]
            rows.append(dict(ID=a["ID"],temporal_and_space_same=all(p[k]==q[k] for k in ("collision_frame","entry_frame","evasion_space")),
                             side_reversed=p["entry_side"]!=q["entry_side"],canonical=p,mirror=q))
        report["mirror_diagnostics"][arm]=rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true")
    args=parser.parse_args()
    cv2.setNumThreads(2)
    if args.freeze:
        freeze();return
    frozen=json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen["sources"]==source_manifest()
    assert frozen["input_sets"]==input_manifest()
    assert all(sha(MODEL/n)==h for n,h in frozen["model"].items())
    if OUTPUT.exists(): raise FileExistsError("Refusing to overwrite paired experiment")
    OUTPUT.mkdir()
    (OUTPUT/"stage2_v5.py").write_bytes((ROOT/"solution/stage2_v5.py").read_bytes())
    (OUTPUT/"evaluate_stage2_v5.py").write_bytes(Path(__file__).read_bytes())
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status="running",freeze_sha256=sha(FREEZE),
                call_count=0,network_attempts=0,sets={name:{"V3":[],"V5":[]} for name in SETS})
    def deny(*args,**kwargs):
        report["network_attempts"]+=1
        raise RuntimeError("Offline Stage2 V5 network denied")
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    try:
        with candidate.CandidateVLM(MODEL,precision="nf4") as model:
            model.torch.set_num_threads(2)
            report["runtime"]=model.candidate_metadata
            for rep,relative in SETS.items():
                for folder in sorted((ROOT/relative).iterdir()):
                    if not folder.is_dir():continue
                    paths=sorted((p for p in folder.iterdir() if p.suffix.lower() in candidate.base.IMAGE_EXTENSIONS),key=candidate.base._frame_number)
                    start=time.perf_counter();paths,scores,_=candidate.base._motion_scan(paths)
                    scan=time.perf_counter()-start
                    group=[r for r in frozen["input_sets"][rep] if r["ID"]==folder.name]
                    digest=hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()
                    for arm,module in (("V3",baseline),("V5",candidate)):
                        rec=Recorder(model,OUTPUT/rep/arm/folder.name,report)
                        start=time.perf_counter();pred,diag=module._predict_file(paths,scores,rec)
                        elapsed=time.perf_counter()-start
                        report["sets"][rep][arm].append(dict(ID=folder.name,prediction=pred,diagnostics=diag,
                            input_manifest_sha256=digest,input_path=folder.as_posix(),frames=len(paths),
                            frame_numbers=[candidate.base._frame_number(p) for p in paths],motion_scores=scores.tolist(),
                            scan_seconds=scan,predict_elapsed_seconds=elapsed,predict_compute_seconds=elapsed-rec.overhead,
                            logger_overhead_seconds=rec.overhead,calls=rec.calls))
                        write(OUTPUT/"report.json",report)
        summarize(report,frozen)
        assert frozen["sources"]==source_manifest()
        assert frozen["input_sets"]==input_manifest()
        assert all(sha(MODEL/n)==h for n,h in frozen["model"].items())
        assert report["network_attempts"]==0
        report.update(status="complete",sources_model_inputs_unchanged=True)
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
        raise
    finally:
        write(OUTPUT/"report.json",report)
    print(json.dumps(report["summary"]),flush=True)


if __name__ == "__main__":main()
