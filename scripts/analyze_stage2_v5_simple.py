"""Post-inference simple candidate contracts and advisory-only diagnostics."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/"research/v5_stage2"
report=json.loads((folder/"paired_simple_run/report.json").read_text(encoding="utf-8"))
assert report["status"]=="complete"
observations=json.loads((ROOT/"research/v4_stage2_annotation/root_review.json").read_text(encoding="utf-8"))
obs={r["ID"]:r for r in observations["rows"]}
summary={};pairs=[]
for rep,arms in report["sets"].items():
    summary[rep]=dict(entry_origins=dict(Counter(r["diagnostics"]["entry_origin"] for r in arms["SIMPLE"])),
        coarse_invalid=sum(not r["diagnostics"]["coarse_valid"] for r in arms["SIMPLE"]),
        fine_invalid=sum(not r["diagnostics"]["fine_valid"] for r in arms["SIMPLE"]),
        valid_selected_first=sum(r["diagnostics"]["valid_model_selected_first"] for r in arms["SIMPLE"]))
    for a,b in zip(arms["V3"],arms["SIMPLE"]):
        ac,bc=a["calls"][0],b["calls"][0]
        first_equal=all(ac[k]==bc[k] for k in ("prompt","max_new_tokens","input_images","raw_output"))
        assert first_equal and a["prediction"]["entry_side"]==b["prediction"]["entry_side"]
        d=b["diagnostics"]
        row=dict(representation=rep,ID=b["ID"],V3=a["prediction"],SIMPLE=b["prediction"],
            first_call_same=first_equal,entry_origin=d["entry_origin"],coarse=d["coarse"],fine=d["fine"],
            coarse_candidates=d["coarse_candidates"],fine_candidates=d["fine_candidates"],
            space_same_input=a["calls"][3]["input_images"]==b["calls"][3]["input_images"],
            space_same_raw=a["calls"][3]["raw_output"]==b["calls"][3]["raw_output"],
            space_context=d["space_context"],entry_after_motion=d["entry_after_motion_collision"])
        observation=obs[b["ID"]]
        if observation.get("entry_interval_advisory"):
            low,high=observation["entry_interval_advisory"]
            row["AI_advisory_not_GT"]=dict(interval=[low,high],
                coarse_candidates_inside=[v for v in d["coarse_candidates"] if low<=v<=high],
                fine_candidates_inside=[v for v in d["fine_candidates"] if low<=v<=high],
                V3_inside=low<=a["prediction"]["entry_frame"]<=high,
                SIMPLE_inside=low<=b["prediction"]["entry_frame"]<=high)
        pairs.append(row)
result=dict(no_new_inference=True,independent_source_groups=5,derived_inputs=15,official_entry_side_space_GT=False,
    report_sha256=hashlib.sha256((folder/"paired_simple_run/report.json").read_bytes()).hexdigest(),
    summary=summary,pairs=pairs,timing=report["summary"],mirror=report["mirror_diagnostics"],
    no_fallback_or_valid_integer_is_semantic_accuracy=True)
out=folder/"simple_analysis.json"
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"summary":summary,"pairs":[{k:r[k] for k in ("representation","ID","V3","SIMPLE","entry_origin","space_same_input")} for r in pairs]},ensure_ascii=False))
