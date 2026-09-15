"""CPU-only existing public-contact audit; never manufactures three missing GTs."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import audit


def main():
    import av  # Decode original PTS on CPU, never estimate timestamps from FPS.
    root = Path(__file__).resolve().parents[2]
    output = Path(__file__).parent
    trace_path = output / "archived_v3_jpeg_trace.json"
    trace = audit.read(trace_path)
    selection_path = root / "artifacts/submissions/v5_selection_frozen.json"
    selection = audit.read(selection_path)
    archive_map = selection["expected_archive_sha256"]
    sources = trace["provenance"]["recorded_sources"]
    code_binding = {}
    for name in ("stage2.py", "stage2_v2.py", "stage2_motion_collision.py", "vlm.py", "vlm_candidate.py"):
        packaged = f"model/stage2/code/solution/{name}"
        expected = sources[f"solution/{name}"]
        actual = audit.sha(root / "artifacts/submissions/verify_v5" / packaged)
        audit.require(actual == expected == archive_map[packaged], "Actual V5 code differs from archived policy")
        code_binding[packaged] = expected
    model_binding = {}
    for name, digest in trace["provenance"]["recorded_model"].items():
        packaged = f"model/stage2/vlm/{name}"
        audit.require(archive_map[packaged] == digest, "Archived/V5 model manifests differ")
        model_binding[packaged] = digest
    csv_path = root / "artifacts/submissions/verify_v5_results/stage2.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        actual_predictions = {r["ID"]: {k: (v if k == "entry_side" else int(v)) for k,v in r.items() if k != "ID"} for r in csv.DictReader(stream)}
    audit.require(all(actual_predictions[r["ID"]] == r["prediction"] for r in trace["videos"]), "Actual V5 public output differs from archived trace")
    labels_path = root / "Baseline/data/stage2/labels.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        labels = {r["ID"]: r for r in csv.DictReader(stream)}
    records = []
    stamp = datetime.now(timezone.utc).isoformat()
    for row in trace["videos"]:
        label = labels[row["ID"]]
        source = labels_path.parent / label["path"]
        with av.open(str(source)) as container:
            decoded = list(container.decode(video=0))
            audit.require(all(f.pts is not None and f.time_base is not None for f in decoded), "Native PTS unavailable")
            mapping = [dict(frame=i,pts_seconds=float(f.pts*f.time_base),native_pts=f.pts,
                        time_base_numerator=f.time_base.numerator,time_base_denominator=f.time_base.denominator) for i,f in enumerate(decoded)]
        audit.require(row["original_frame_numbers"] == list(range(len(mapping))), "Existing public exporter index mapping does not match source")
        pts = [{"frame":m["frame"],"pts_seconds":m["pts_seconds"]} for m in mapping]
        # This index binding is only for this existing verified public exporter.
        # General evaluator never treats arbitrary original numbers as indices.
        mapping_path = output / "public_pts" / f"{row['ID']}.json"
        origin = "Original container decoded presentation timestamp in seconds; no origin shift"
        audit.write_new(mapping_path,dict(frame_pts=pts,native_pts=mapping,
            source_video_sha256=audit.sha(source),time_origin=origin,
            exporter_evidence="scripts/evaluate_stage2_v5_simple.py and research/v5_inputs/manifest.json",
            source_path=source.relative_to(root).as_posix(),created_utc=stamp))
        contact_index = int(label["t_collision"])
        audit.require(contact_index in row["original_frame_numbers"], "Official provided contact index unavailable")
        contact = dict(value=pts[contact_index]["pts_seconds"],gt_type="official",provenance=dict(
            artifact_path=labels_path.relative_to(root).as_posix(),sha256=audit.sha(labels_path),
            source_ref=f"Provided Baseline/data/stage2/labels.csv ID={row['ID']}, t_collision={contact_index}; frame index converted using bound native PTS",
            created_at=stamp,annotators=[],adjudicator=None))
        records.append(dict(ID=row["ID"],source_group_id=f"public_original_{row['ID']}",
            source=dict(dataset="Provided competition Stage2 public examples",source_uri=label["path"],
                video_path=source.relative_to(root).as_posix(),video_sha256=audit.sha(source)),
            split="development",exposure=dict(predictions_seen=True,prior_model_development=True,
                annotation_blinded=False,notes="Repeatedly inspected public source; only provided collision label scored; no new independent GT"),
            target_vehicle_id=None,input_manifest_sha256=row["input_manifest_sha256"],frame_pts=pts,
            mapping_provenance=dict(artifact_path=mapping_path.relative_to(root).as_posix(),sha256=audit.sha(mapping_path),
                method="PyAV sequential original decode; integer PTS times rational time_base; original exporter numbers verified 0..N-1",
                time_origin=origin),labels=dict(collision_time_seconds=contact,entry_time_seconds=None,entry_side=None,evasion_space=None)))
    gt = dict(schema_version=1,cohort_id="existing_public5_partial_collision_only",videos=records)
    gt_path = output / "public_official_partial_gt.json"
    audit.write_new(gt_path,gt)
    result = audit.evaluate(gt,trace,root)
    result["input_binding"] = dict(gt_sha256=audit.sha(gt_path),trace_sha256=audit.sha(trace_path),
        evaluator_sha256=audit.sha(output/"audit.py"),preparer_sha256=audit.sha(__file__),
        actual_v5_selection_sha256=audit.sha(selection_path),actual_v5_csv_sha256=audit.sha(csv_path),
        actual_v5_code_sha256=code_binding,model_hashes_equal_in_frozen_manifests=model_binding,
        weights_rehashed_in_this_cpu_audit=False,actual_v5_predictions_equal=True,
        detailed_trace_is_archived_development_not_actual_v5_runtime=True)
    audit.write_new(output/"public_partial_audit.json",result)
    print(json.dumps({"complete_gt":result["complete_gt"],"S2":result["S2"],"components":result["components"]}))


if __name__ == "__main__":
    main()
