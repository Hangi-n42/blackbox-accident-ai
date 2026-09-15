"""Guarded CPU-only final refit of the fixed Stage3 scale candidate.

Preparation does not authorize training. A later explicit --train invocation
requires completed external/public diagnostics and both predeclared gates.
Writes only the two candidate files; never builds or submits a ZIP.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/"research/stage3_scale_augmentation"
OUTPUT = ROOT/"artifacts/candidates/stage3_scale"
PINNED_SOURCES = {
    "research/stage3_scale_augmentation/experiment.py": "b994e4d03625865ad8ebf22d11b80e50947c575ea8597bc8443a35d6be36ea46",
    "research/stage3_temporal_experiment.py": "5386ad5cb8095171ce4a339bbf9af0334d163e959130692f7456c9ca57fd16fe",
}
ARMS = ("baseline", "replicated", "augmented")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def local(path):
    resolved = (ROOT/str(path).replace("\\", "/")).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Provenance path leaves workspace: {path}")
    return resolved


def weighted_score(arm):
    values = [arm[task]["macro_f1"] for task in ("accel", "steer")]
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in values):
        raise ValueError("Invalid diagnostic macro-F1")
    computed = .7*values[0]+.3*values[1]
    if not math.isclose(computed, arm["stage3"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Saved diagnostic Stage3 score differs from fixed 0.7/0.3 formula")
    return computed


def check_gate():
    # These are development diagnostics. No leaderboard/official Public or Private
    # score file is read, and existing_external23_reference never selects this fit.
    report = read(EXPERIMENT/"report.json")
    public = read(EXPERIMENT/"public_oof.json")
    external = read(EXPERIMENT/"external_validation.json")
    parent_gate = read(EXPERIMENT/"parent_refit_gate_before_public.json")
    split = read(EXPERIMENT/"external_split.json")
    if (report.get("status") != "external_improvement_public_diagnostic_completed"
            or report.get("public_oof_executed") is not True
            or report.get("protected_files_unchanged") is not True):
        raise RuntimeError("Experiment is incomplete, failed, or modified protected files")
    if report.get("public") != public or report.get("external") != external:
        raise RuntimeError("Completed report and component diagnostic files disagree")
    if (parent_gate.get("public_oof_result_exists") is not False
            or parent_gate.get("no_official_submission_score_used") is not True
            or parent_gate.get("preserve_existing_models_and_zips") is not True):
        raise RuntimeError("Required parent gate was not frozen before the public result")
    train, validation = split["train"], split["validation"]
    if len(train) != 17 or len(validation) != 6 or len(set(train+validation)) != 23:
        raise RuntimeError("Expected the unchanged disjoint 17/6 external route split")
    for arm in ARMS:
        timings = public["fit_times"][arm]
        if len(timings) != 10:
            raise RuntimeError("Public OOF must finish all five folds for both heads")
        for task in ("accel", "steer"):
            held = [row["held_out_video"] for row in timings if row["task"] == task]
            if len(held) != 5 or set(held) != {f"OPEN_{i:03d}" for i in range(1, 6)}:
                raise RuntimeError("Public OOF fold identities are incomplete")
    ex = {arm: weighted_score(external[arm]) for arm in ARMS}
    pub = {arm: weighted_score(public[arm]) for arm in ARMS}
    checks = {"external_above_baseline": ex["augmented"] > ex["baseline"],
              "external_above_replicated": ex["augmented"] > ex["replicated"],
              "public_same17_not_below_baseline": pub["augmented"] >= pub["baseline"],
              "public_same17_not_below_replicated": pub["augmented"] >= pub["replicated"]}
    result = {"passed": all(checks.values()), "checks": checks,
              "fixed_score_formula": "0.7 * accel macro-F1 + 0.3 * steer macro-F1",
              "external_scores": ex, "public_same_external17_oof_scores": pub,
              "official_submission_scores_used": False,
              "parent_gate": parent_gate, "external_split_before_refit": split}
    if not result["passed"]:
        raise RuntimeError("Predeclared refit gate failed: "+json.dumps(result))
    return result


def protected_snapshot(paths=None):
    if paths is None:
        paths = [ROOT/"solution/stage3.py", ROOT/"solution/stage3_fast.py"]
        paths += list((ROOT/"model/stage3").glob("*.joblib"))
        paths += list((ROOT/"solution/model/stage3").glob("*.joblib"))
        paths += list((ROOT/"artifacts/model/stage3").glob("*.joblib"))
        paths += list((ROOT/"artifacts/submissions").glob("*.zip"))
    return {path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "sha256": None if path.suffix == ".zip" else sha(path)}
            for path in sorted(paths) if path.is_file()}


def preserve_text(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
            "text": path.read_text(encoding="utf-8")}


def refit(gate):
    # Set before scientific imports; all learners stay on CPU with at most two threads.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ROOT))
    for name, expected in PINNED_SOURCES.items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f"Reusable experiment source changed: {name}")
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from research.stage3_scale_augmentation import experiment as scale

    if tuple(scale.SCALES) != (.75, 1., 1.25):
        raise RuntimeError("Fixed scale factors changed")
    frozen = read(EXPERIMENT/"config_frozen_before_fit.json")
    current_parameters = json.loads(json.dumps({task: scale.make_model(task).get_params()
                                               for task in ("accel", "steer")}, default=str))
    if current_parameters != frozen["model_parameters"]:
        raise RuntimeError("Logistic/RF settings differ from the evaluated experiment")
    for item in read(EXPERIMENT/"sources.json")["files"]:
        if sha(local(item["path"])) != item["sha256"]:
            raise RuntimeError(f"External source metadata changed: {item['path']}")
    inventory = read(EXPERIMENT/"feature_cache_inventory.json")
    for item in inventory:
        path = local(item["path"])
        if path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
            raise RuntimeError(f"Previously evaluated feature cache changed: {item['path']}")
    before = protected_snapshot()
    production_key = "model/stage3/motion_model.joblib"
    frozen_protected = {name.replace("\\", "/"): info for name, info in frozen["protected_files_before"].items()}
    if before[production_key]["sha256"] != frozen_protected[production_key]["sha256"]:
        raise RuntimeError("Protected production model changed since the scale experiment")

    with threadpool_limits(limits=2):
        # Reuse the exact CAN thresholds, time alignment, downsampling and STOPPED mask.
        external = scale.load_external()
        wanted = gate["external_split_before_refit"]["train"]+gate["external_split_before_refit"]["validation"]
        if len(external) != 23 or len({row["segment"] for row in external}) != 23:
            raise RuntimeError("Final refit requires exactly the 23 previously audited segments")
        if {row["route"] for row in external} != set(wanted):
            raise RuntimeError("Final external route set differs from the selection experiment")
        frame = pd.read_csv(ROOT/"Baseline/data/stage3/labels.csv")
        if len(frame) != 50 or set(frame.ID) != {f"OPEN_{i:03d}" for i in range(1, 6)}:
            raise RuntimeError("Expected exactly the 50 original sparse public labels")
        if frame.duplicated(["ID", "frame_index"]).any():
            raise RuntimeError("Duplicate public labelled frames")
        features = {video: np.load(ROOT/"research/stage3_cache"/f"{video}.npy", allow_pickle=False)
                    for video in frame.ID.unique()}
        # Keep CSV row alignment; original frame_index is used for training just as in public_oof().
        px = np.stack([features[row.ID][int(row.frame_index)] for row in frame.itertuples()])
        ya = np.array([scale.ACCEL.index(value) for value in frame.accel_label])
        ys = np.array([scale.STEER.index(value) for value in frame.steer_label])
        if px.shape != (50, 864) or not np.isfinite(px).all():
            raise RuntimeError("Invalid sparse public feature matrix")
        datasets, row_counts = {}, {}
        for task, y in (("accel", ya), ("steer", ys)):
            ex, ey = scale.external_xy(external, task)
            public_keep = ya != 3 if task == "steer" else np.ones(len(ya), dtype=bool)
            x = np.concatenate([px[public_keep], ex])
            target = np.concatenate([y[public_keep], ey])
            if x.ndim != 2 or x.shape[1] != 864 or len(x) != len(target) or not np.isfinite(x).all():
                raise RuntimeError(f"Invalid final training features for {task}")
            datasets[task] = (x, target)
            row_counts[task] = {"external_unaugmented": len(ey), "public_unaugmented": int(public_keep.sum()),
                                "total_unaugmented": len(target), "total_augmented": 3*len(target),
                                "class_counts_unaugmented": np.bincount(target, minlength=4 if task == "accel" else 3).tolist(),
                                "public_stopped_excluded": int((ya == 3).sum()) if task == "steer" else 0}

        text_paths = [Path(__file__), ROOT/"research/stage3_scale_augmentation/experiment.py",
                      ROOT/"research/stage3_temporal_experiment.py", ROOT/"solution/stage3.py",
                      ROOT/"external_data/comma2k19/LICENSE", ROOT/"Baseline/data/stage3/labels.csv"]
        text_paths += [EXPERIMENT/name for name in ("report.json", "public_oof.json", "external_split.json",
                       "config_frozen_before_fit.json", "amendment_frozen_before_results.json",
                       "parent_refit_gate_before_public.json", "sources.json", "feature_cache_inventory.json")]
        text_paths += [ROOT/"research/stage3_external_overlap.json"]
        text_paths += sorted((ROOT/"external_data/comma2k19").glob("*_manifest.json"))
        preserved = [preserve_text(path) for path in text_paths]
        excluded = {row["segment"] for row in read(ROOT/"research/stage3_external_overlap.json")["excluded"]}
        data_list = []
        for manifest in sorted((ROOT/"external_data/comma2k19").glob("*_manifest.json")):
            for item in read(manifest):
                if item["segment"] in excluded:
                    continue
                base = local(item["local"])
                used = [base/name for name in ("global_pose/frame_times", "processed_log/CAN/speed/t",
                        "processed_log/CAN/speed/value", "processed_log/CAN/steering_angle/t",
                        "processed_log/CAN/steering_angle/value")]
                data_list.append({**item, "label_input_files": [{"path": path.relative_to(ROOT).as_posix(),
                                  "bytes": path.stat().st_size, "sha256": sha(path)} for path in used]})
        if len(data_list) != 23:
            raise RuntimeError("External provenance list does not match fitted segments")
        # Reserve a new candidate directory only after every input/gate preflight passes.
        OUTPUT.mkdir(parents=True, exist_ok=False)
        provenance = {"status": "fitting", "created_at": datetime.now(timezone.utc).isoformat(),
                      "selection_gate": gate, "threads": 2, "gpu_used": False,
                      "scales": [.75, 1., 1.25], "no_smoothing": True,
                      "feature_version": "dis256_roi144_temporal6_v1", "training_rows": row_counts,
                      "model_parameters": current_parameters, "external_data": data_list,
                      "public_data_rows": frame.to_dict(orient="records"), "preserved_text_sources": preserved,
                      "protected_before": before, "license": "comma2k19 MIT; exact license text preserved above",
                      "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")},
                      "call_paths": ["experiment.load_external -> stage3_temporal_experiment.load_external",
                                     "experiment.external_xy -> stage3_temporal_experiment.external_xy",
                                     "experiment.fit(augmented=True) -> make_model -> model.fit"],
                      "steering_mask": "External accel_train != 3 and public GT accel_label != STOPPED, applied before augmentation; never predicted accel.",
                      "final_model_independently_evaluated": False,
                      "caveats": ["The six previously held-out external validation routes are included in this final refit. Their earlier score is NOT independent performance of this fitted model.",
                                  "Public OOF uses the same 50 previously inspected labels and remains a development diagnostic.",
                                  "CAN proxy thresholds are not official competition thresholds.",
                                  "All scale copies have unit weight. Sample replication also changes effective regularization/bootstrap.",
                                  "No official Public/Private leaderboard value was used by any gate."]}
        started = time.perf_counter()
        try:
            models, timings = {}, {}
            for task in ("accel", "steer"):
                x, y = datasets[task]
                models[task], timings[task] = scale.fit(task, x, y, True)
                if timings[task]["fitted_rows"] != row_counts[task]["total_augmented"]:
                    raise RuntimeError("Fitted sample count differs from fixed three-scale augmentation")
                print(json.dumps({"task": task, "timing": timings[task]}), flush=True)
            after = protected_snapshot([local(path) for path in before])
            if before != after:
                raise RuntimeError("Protected files changed during candidate refit")
            model = {**models, "feature_version": "dis256_roi144_temporal6_v1",
                     "training_augmentation": {"global_feature_scales": [.75, 1., 1.25]},
                     "external_dataset": "commaai/comma2k19 MIT; 23 audited segments plus 50 sparse public labels"}
            model_path = OUTPUT/"motion_model.joblib"
            with model_path.open("xb") as stream:
                joblib.dump(model, stream, compress=3)
            provenance.update(status="complete", fit_times=timings, elapsed_seconds=time.perf_counter()-started,
                              model_sha256=sha(model_path), model_bytes=model_path.stat().st_size,
                              protected_after=after, protected_files_unchanged=True)
        except BaseException as error:
            provenance.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            raise
        finally:
            with (OUTPUT/"training_provenance.json").open("x", encoding="utf-8") as stream:
                json.dump(provenance, stream, ensure_ascii=False, indent=2, default=str)
        print(json.dumps({"status": "complete", "candidate": str(OUTPUT),
                          "model_sha256": provenance["model_sha256"], "independent_final_score": None}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-gate", action="store_true", help="Read completed diagnostics without fitting or writing")
    action.add_argument("--train", action="store_true", help="Fit the separate candidate only if every frozen gate passes")
    args = parser.parse_args()
    if args.train and OUTPUT.exists():
        parser.error(f"Candidate directory already exists; refusing to overwrite: {OUTPUT}")
    gate = check_gate()
    if args.check_gate:
        print(json.dumps(gate, indent=2))
        return
    refit(gate)


if __name__ == "__main__":
    main()
