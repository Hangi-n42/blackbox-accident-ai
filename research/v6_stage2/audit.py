"""CPU-only Stage2 archived-trace audit. No model imports or inference entrypoint."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

FIELDS = ("collision_time_seconds", "entry_time_seconds", "entry_side", "evasion_space")
TOLERANCE_SECONDS = 0.3  # Official rule; not a model adoption threshold.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def artifact(item, root):
    require(isinstance(item, dict), "Evidence must be an object")
    require(nonempty(item.get("artifact_path")), "Missing evidence artifact_path")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))), "Invalid evidence SHA")
    path = Path(root) / item["artifact_path"]
    require(path.is_file() and sha(path) == item["sha256"], f"Evidence hash mismatch: {path}")


def validate_label(label, field, root):
    if label is None:
        return None
    require(isinstance(label, dict), f"{field}: label must be object/null")
    kind = label.get("gt_type")
    require(kind in ("official", "human_adjudicated"), f"{field}: unaccepted GT type {kind!r}")
    p = label.get("provenance")
    artifact(p, root)
    require(nonempty(p.get("source_ref")), "GT source_ref required")
    try:
        stamp = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
        require(stamp.tzinfo is not None, "GT created_at requires timezone")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("GT created_at must be timezone-aware ISO8601") from error
    names = p.get("annotators")
    require(isinstance(names, list) and all(nonempty(x) for x in names), "Invalid annotators")
    require("adjudicator" in p and (p["adjudicator"] is None or nonempty(p["adjudicator"])), "Invalid adjudicator")
    if kind == "human_adjudicated":
        require(len(set(names)) >= 2 and nonempty(p["adjudicator"]), "Two human reviewers and adjudicator required")
        reviews = p.get("independent_reviews")
        require(isinstance(reviews, list), "Independent review artifacts required")
        require(len({r.get("annotator") for r in reviews}) >= 2, "Two distinct review identities required")
        require(len({r.get("sha256") for r in reviews}) >= 2, "Two distinct review artifacts required")
        for review in reviews:
            require(review.get("annotator") in names, "Review identity absent from annotators")
            artifact(review, root)
        artifact(p.get("adjudication"), root)
        require(nonempty(p["adjudication"].get("resolution")), "Adjudication resolution required")
    value = label.get("value")
    if field.endswith("time_seconds"):
        require(finite_number(value) and value >= 0, f"{field}: finite nonnegative seconds required")
    elif field == "entry_side":
        require(value in ("LEFT", "RIGHT"), "Invalid GT entry_side")
    else:
        require(type(value) is int and value in (0, 1), "Invalid GT evasion_space")
    return value


def validate_gt(document, root):
    require(document.get("schema_version") == 1 and nonempty(document.get("cohort_id")), "Invalid GT header")
    videos = document.get("videos")
    require(isinstance(videos, list) and videos, "Nonempty declared GT cohort required")
    validated = {}
    for video in videos:
        ID = video.get("ID")
        require(nonempty(ID) and ID not in validated, "Empty/duplicate GT ID")
        require(nonempty(video.get("source_group_id")), f"{ID}: source group required")
        s = video.get("source", {})
        require(all(nonempty(s.get(k)) for k in ("dataset", "source_uri", "video_path")), "Source provenance missing")
        artifact({"artifact_path": s["video_path"], "sha256": s.get("video_sha256")}, root)
        require(video.get("split") in ("development", "validation", "test"), "Invalid split")
        e = video.get("exposure", {})
        require(all(type(e.get(k)) is bool for k in ("predictions_seen", "prior_model_development", "annotation_blinded")), "Explicit exposure booleans required")
        require(nonempty(e.get("notes")), "Exposure notes required")
        if video["split"] != "development":
            require(not e["predictions_seen"] and not e["prior_model_development"] and e["annotation_blinded"], "Exposed/nonblind data must use development split")
        require("target_vehicle_id" in video and (video["target_vehicle_id"] is None or nonempty(video["target_vehicle_id"])), "Explicit target identity/null required")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", str(video.get("input_manifest_sha256", "")))), "Input manifest hash required")
        mapping = video.get("mapping_provenance", {})
        artifact(mapping, root)
        require(nonempty(mapping.get("method")) and nonempty(mapping.get("time_origin")), "Native PTS method and origin required")
        frames = video.get("frame_pts")
        require(isinstance(frames, list) and frames, "Explicit frame to PTS mapping required")
        evidence = read(Path(root) / mapping["artifact_path"])
        require(evidence.get("frame_pts") == frames and evidence.get("source_video_sha256") == s["video_sha256"]
                and evidence.get("time_origin") == mapping["time_origin"], "PTS evidence content/source/origin mismatch")
        times = {}
        previous = -math.inf
        for item in frames:
            number, pts = item.get("frame"), item.get("pts_seconds")
            require(type(number) is int and number >= 0 and number not in times, "Invalid/duplicate original frame number")
            require(finite_number(pts) and pts >= 0 and pts > previous, "PTS must be finite, nonnegative and strictly chronological")
            times[number] = float(pts)
            previous = pts
        labels = video.get("labels", {})
        require(set(labels) == set(FIELDS), "All four label keys required; use null for missing")
        values = {field: validate_label(labels[field], field, root) for field in FIELDS}
        if any(label is not None and label["gt_type"] == "human_adjudicated" for label in labels.values()):
            require(nonempty(video["target_vehicle_id"]), "Human-adjudicated GT requires collision target identity")
        for field in FIELDS[:2]:
            if values[field] is not None:
                require(min(times.values()) <= values[field] <= max(times.values()), "GT time outside explicitly mapped clip")
        validated[ID] = {"metadata": video, "times": times, "labels": values}
    groups = {}
    source_groups = {}
    for entry in validated.values():
        video = entry["metadata"]
        key = video["source_group_id"]
        groups.setdefault(key, set()).add(video["split"])
        source_groups.setdefault(video["source"]["video_sha256"], set()).add(key)
    require(all(len(splits) == 1 for splits in groups.values()), "One source group cannot straddle development/validation/test")
    require(all(len(names) == 1 for names in source_groups.values()), "Identical source bytes cannot count as different source groups")
    return validated


def integer_like(value):
    # Match the frozen V5 permissive parser only to describe selection provenance.
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(?:[Ff][Rr][Aa][Mm][Ee][ _]*)?(\d+)\s*", value)
        if match:
            return int(match.group(1))
    return None


def selection_origin(raw, offered, selected, first):
    number = integer_like(raw)
    if number is None:
        return "invalid_fallback_first" if selected == first else "invalid_fallback_other"
    if number not in offered:
        return "numeric_snapped_to_offered"
    return "valid_first_selected_semantics_unverified" if selected == first else "valid_offered_selected"


def normalize_trace(row, original_numbers=None):
    """Lossless per-file trace adapter; never calls or imports a VLM."""
    d = row["diagnostics"]
    replacement = d["collision_replacement"]
    valid = list(row["frame_numbers"])
    original = list(original_numbers if original_numbers is not None else valid)
    require(original and len(set(original)) == len(original), "Invalid original frame list")
    require(valid and len(set(valid)) == len(valid) and set(valid) <= set(original), "Invalid decoded frame list")
    require([n for n in original if n in set(valid)] == valid, "Decoded frame chronology changed")
    internal = replacement["base_collision_frame"]
    require(internal in valid, "Internal collision absent from decoded original numbers")
    collision_candidates = d["collision_candidates"]
    require(internal in collision_candidates and set(collision_candidates) <= set(valid), "Internal collision candidates mismatch")
    require(replacement["collision_frame"] in valid, "Motion collision absent from valid original numbers")
    prefix = valid[:valid.index(internal) + 1]
    candidates = d["entry_candidates"]
    require(candidates and len(set(candidates)) == len(candidates) and set(candidates) <= set(prefix), "Entry candidates violate actual V5 collision prefix")
    prediction = dict(row["prediction"])
    require(prediction.get("collision_frame") == replacement["collision_frame"], "Final collision replacement mismatch")
    require(prediction.get("entry_frame") in candidates, "Entry final outside recorded offered candidates")
    calls = row.get("calls", [])
    require(len(calls) == 4 and all(c.get("status") == "complete" for c in calls), "Four completed archived V5-policy asks required")
    return dict(ID=row["ID"], input_manifest_sha256=row["input_manifest_sha256"],
        original_frame_numbers=original, valid_frame_numbers=valid,
        internal_collision_frame=internal, motion_collision_frame=replacement["collision_frame"],
        entry_prefix=prefix, entry_candidates=candidates,
        prediction=prediction, diagnostics=d, calls=calls,
        selection_provenance=dict(entry=selection_origin(d.get("entry", {}).get("entry_frame"), candidates, prediction["entry_frame"], valid[0]),
            side="valid_class" if str(d.get("coarse", {}).get("entry_side", "")).upper().strip() in ("LEFT", "RIGHT") else "invalid_fallback_LEFT",
            space="valid_class" if integer_like(d.get("space", {}).get("evasion_space")) in (0, 1) else "invalid_fallback_zero"))


def error_seconds(frame, times, truth):
    if type(frame) is not int or frame not in times:
        return None
    return times[frame] - truth


def within(error):
    # abs_tol is solely floating-point equality at the official inclusive boundary.
    return error is not None and (abs(error) <= TOLERANCE_SECONDS or math.isclose(abs(error), TOLERANCE_SECONDS, rel_tol=0, abs_tol=1e-12))


def coverage(pool, times, truth):
    distances = [abs(times[frame] - truth) for frame in pool]
    return {"count": len(pool), "nearest_error_seconds": min(distances) if distances else None,
            "within_0_3_seconds": any(within(x) for x in distances)}


def macro_f1(truth, prediction, classes):
    """Fixed full class set; absent class denominator zero -> F1 zero."""
    scores = {}
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        denominator = 2 * tp + fp + fn
        scores[str(label)] = {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / denominator if denominator else 0.0}
    return {"value": sum(x["f1"] for x in scores.values()) / len(classes), "per_class": scores}


def evaluate(gt, trace_document, root):
    checked = validate_gt(gt, root)
    require(trace_document.get("schema_version") == 1, "Trace schema mismatch")
    rows = trace_document.get("videos", [])
    traces = {r["ID"]: r for r in rows}
    require(len(traces) == len(rows) and set(traces) == set(checked), "GT and trace cohorts must match exactly")
    audited = []
    truths, predictions = {k: [] for k in FIELDS}, {k: [] for k in FIELDS}
    for ID, g in checked.items():
        trace = traces[ID]
        require(trace["input_manifest_sha256"] == g["metadata"]["input_manifest_sha256"], f"{ID}: input representation binding mismatch")
        # Reconstruct from original archived fields, ignoring potentially stale derived audit fields.
        t = normalize_trace(dict(trace, frame_numbers=trace["valid_frame_numbers"]), trace["original_frame_numbers"])
        times, labels, pred = g["times"], g["labels"], t["prediction"]
        require(list(times) == t["original_frame_numbers"], f"{ID}: PTS mapping must exactly cover original ordered input")
        item = {"ID": ID, "source_group_id": g["metadata"]["source_group_id"], "split": g["metadata"]["split"],
                "selection_provenance": t["selection_provenance"], "entry_coverage": None, "collision_errors": None}
        for field in FIELDS:
            truth = labels[field]
            if truth is None:
                continue
            truths[field].append(truth)
            if field.endswith("time_seconds"):
                frame_key = field.replace("_time_seconds", "_frame")
                predictions[field].append(within(error_seconds(pred.get(frame_key), times, truth)))
            elif field == "evasion_space":
                value = pred.get(field)
                predictions[field].append(value if type(value) is int and value in (0, 1) else None)
            else:
                value = pred.get(field)
                predictions[field].append(value if value in ("LEFT", "RIGHT") else None)
        if labels["collision_time_seconds"] is not None:
            truth = labels["collision_time_seconds"]
            item["collision_errors"] = {name: {"signed_seconds": error_seconds(frame, times, truth),
                "correct_at_0_3_seconds": within(error_seconds(frame, times, truth))} for name, frame in
                (("final_motion", t["motion_collision_frame"]), ("internal_vlm", t["internal_collision_frame"]))}
            item["collision_candidate_coverage"] = {name: coverage(pool, times, truth) for name, pool in
                (("all_original", t["original_frame_numbers"]), ("vlm_offered", t["diagnostics"]["collision_candidates"]),
                 ("motion_singleton", [t["motion_collision_frame"]]))}
        if labels["entry_time_seconds"] is not None:
            truth = labels["entry_time_seconds"]
            pools = [("all_original", t["original_frame_numbers"]), ("decoded_valid", t["valid_frame_numbers"]),
                     ("internal_collision_prefix", t["entry_prefix"]), ("entry_candidates", t["entry_candidates"]),
                     ("selected", [pred["entry_frame"]])]
            stages = {name: coverage(pool, times, truth) for name, pool in pools}
            bucket = "correct_at_0_3_seconds"
            for (name, _), reason in zip(pools, ("original_time_coverage_missing", "decode_filter_loss", "internal_collision_upper_bound_loss", "entry_grid_loss", "selection_error")):
                if not stages[name]["within_0_3_seconds"]:
                    bucket = reason
                    break
            item["entry_coverage"] = {"stages": stages, "first_failure": bucket,
                "selected_signed_error_seconds": error_seconds(pred["entry_frame"], times, truth),
                "gt_after_internal_collision": truth > times[t["internal_collision_frame"]]}
        audited.append(item)
    components = {}
    for field in FIELDS:
        n = len(truths[field])
        component = {"labeled_count": n, "cohort_count": len(checked), "complete": n == len(checked), "value": None}
        if n:
            if field.endswith("time_seconds"):
                component.update(value=sum(predictions[field]) / n, correct_count=sum(predictions[field]))
            else:
                component.update(macro_f1(truths[field], predictions[field], ("LEFT", "RIGHT") if field == "entry_side" else (0, 1)))
        components[field] = component
    complete = all(c["complete"] for c in components.values())
    s2 = sum(weight * components[field]["value"] for field, weight in zip(FIELDS, (.35, .35, .15, .15))) if complete else None
    return {"schema_version": 1, "cohort_id": gt["cohort_id"], "scope": trace_document.get("scope"),
        "complete_gt": complete, "S2": s2, "components": components, "videos": audited,
        "source_group_count": len({g["metadata"]["source_group_id"] for g in checked.values()}),
        "entry_error_counts": dict(Counter(x["entry_coverage"]["first_failure"] for x in audited if x["entry_coverage"])),
        "entry_selection_origin_counts": dict(Counter(x["selection_provenance"]["entry"] for x in audited)),
        "missing_gt": [{"ID": ID, "fields": [k for k in FIELDS if v["labels"][k] is None]} for ID, v in checked.items() if any(x is None for x in v["labels"].values())],
        "interpretation": "No adoption gate. Partial components have labeled-only denominators. Human/evidence authenticity requires operational verification; hashes only bind artifacts."}


def import_archive(report_path, freeze_path, representation, root):
    report, frozen = read(report_path), read(freeze_path)
    require(report.get("status") == "complete" and report.get("sources_model_inputs_unchanged") is True, "Incomplete/unbound archived run")
    require(report["freeze_sha256"] == sha(freeze_path), "Archive freeze hash mismatch")
    groups = frozen["input_sets"][representation]
    videos = []
    for row in report["sets"][representation]["V3"]:
        group = [x for x in groups if x["ID"] == row["ID"]]
        digest = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
        require(digest == row["input_manifest_sha256"], "Archived per-video manifest mismatch")
        for item in group:
            require(sha(Path(root) / item["path"]) == item["file_sha256"], "Archived input bytes changed")
        videos.append(normalize_trace(row, [x["original_number"] for x in group]))
    return {"schema_version": 1, "scope": "development_archived_V3_policy_trace_not_V5_server_trace",
        "provenance": {"report_path": str(report_path), "report_sha256": sha(report_path), "freeze_path": str(freeze_path),
            "freeze_sha256": sha(freeze_path), "representation": representation,
            "recorded_sources": frozen["sources"], "recorded_model": frozen["model"]}, "videos": videos}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    sub = parser.add_subparsers(dest="command", required=True)
    archive = sub.add_parser("import-archive", help="Read archived V3 arm only; no inference")
    archive.add_argument("--report", type=Path, required=True)
    archive.add_argument("--freeze", type=Path, required=True)
    archive.add_argument("--representation", choices=("canonical", "jpeg", "mirror"), required=True)
    archive.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("evaluate")
    score.add_argument("--gt", type=Path, required=True)
    score.add_argument("--trace", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "import-archive":
        result = import_archive(args.report, args.freeze, args.representation, args.root)
    else:
        result = evaluate(read(args.gt), read(args.trace), args.root)
        result["input_binding"] = {"gt_sha256": sha(args.gt), "trace_sha256": sha(args.trace), "evaluator_sha256": sha(__file__)}
    write_new(args.output, result)
    print(json.dumps({"output": str(args.output), "S2": result.get("S2"), "videos": len(result["videos"])}))


if __name__ == "__main__":
    main()
