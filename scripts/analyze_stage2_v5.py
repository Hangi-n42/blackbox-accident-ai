"""Post-inference audit of frozen V5; no fitting or prediction changes."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/"research/v5_stage2"
report = json.loads((folder/"paired_layout_run/report.json").read_text(encoding="utf-8"))
assert report["status"] == "complete"
observations = json.loads((ROOT/"research/v4_stage2_annotation/root_review.json").read_text(encoding="utf-8"))
obs = {r["ID"]:r for r in observations["rows"]}
summary = {}
all_origins, all_statuses = collections.Counter(), collections.Counter()
comparisons=[]
for representation, arms in report["sets"].items():
    origins=collections.Counter(r["diagnostics"]["entry_origin"] for r in arms["V5"])
    all_origins.update(origins)
    statuses=collections.Counter(r["diagnostics"]["calls"][1]["parsed"].get("status") for r in arms["V5"])
    all_statuses.update(statuses)
    summary[representation] = dict(entry_origins=dict(origins),coarse_statuses=dict(statuses),
        identified_schema_valid=sum(r["diagnostics"]["reference"]["identified"] for r in arms["V5"]),
        identified_is_not_semantic_success=True,
        fine_reasons=dict(collections.Counter(r["diagnostics"]["fine"]["reason"] for r in arms["V5"])),
        tracking_seconds=sum(r["diagnostics"]["tracking"]["seconds"] for r in arms["V5"]))
    for a,b in zip(arms["V3"],arms["V5"]):
        d=b["diagnostics"]
        row=dict(representation=representation,ID=b["ID"],V3=a["prediction"],V5=b["prediction"],
            entry_origin=d["entry_origin"],reference=d["reference"],tracked_frames=len(d["tracking"]["tracked_frames"]),
            tracking_events=d["tracking"]["events"],coarse_candidates=d["coarse"]["candidates"],fine_candidates=d["fine"]["candidates"])
        annotation=obs[b["ID"]]
        if annotation.get("entry_interval_advisory"):
            low,high=annotation["entry_interval_advisory"]
            row["AI_advisory_only"]={"interval":[low,high],"official_GT":False,
                "coarse_offered_inside_interval":[n for n in d["coarse"]["candidates"] if low <= n <= high],
                "fine_offered_inside_interval":[n for n in d["fine"]["candidates"] if low <= n <= high],
                "V3_entry_inside_advisory":low <= a["prediction"]["entry_frame"] <= high,
                "V5_entry_inside_advisory":low <= b["prediction"]["entry_frame"] <= high}
        comparisons.append(row)
result=dict(scope="Five source groups with three representations; not 15 independent validation samples",
    no_official_entry_side_space_GT=True,report_sha256=hashlib.sha256((folder/"paired_layout_run/report.json").read_bytes()).hexdigest(),
    source_observation_sha256=hashlib.sha256((ROOT/"research/v4_stage2_annotation/root_review.json").read_bytes()).hexdigest(),
    entry_origins_total=dict(all_origins),coarse_statuses_total=dict(all_statuses),
    representation_summary=summary,comparisons=comparisons,
    time_gates={k:v["timing_gate_pass"] for k,v in report["summary"].items()},
    timing_ratios={k:v["timing_ratio"] for k,v in report["summary"].items()},
    max_reserved_vram_gib=max(c["peak_reserved_bytes"] for arms in report["sets"].values() for rows in arms.values() for r in rows for c in r["calls"])/1024**3,
    mirror_diagnostics=report["mirror_diagnostics"],
    interpretation="Schema/folder contracts are distinct from semantic success. A first-frame fallback coinciding with an already-in-lane observation is not an observed detection.")
target=folder/"results_analysis.json"
assert not target.exists()
target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:result[k] for k in ("entry_origins_total","coarse_statuses_total","representation_summary","timing_ratios","max_reserved_vram_gib")},ensure_ascii=False))
