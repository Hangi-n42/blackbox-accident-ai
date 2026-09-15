"""Evaluate frozen Stage 2 on public examples, recording every VLM call.

No labels are passed into the inference functions. This is a five-example public
diagnostic, not a hidden/private-score estimate. GPU runs must be scheduled by
the caller so they do not overlap other Stage evaluations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from solution.stage2 import (
    IMAGE_EXTENSIONS, _frame_number, _integer, _json_object, _motion_scan,
    _predict_file,
)

CALL_NAMES = ("overview", "collision_refinement", "entry_refinement", "final_review")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def candidates_from_prompt(prompt: str) -> dict[str, list[int]]:
    result = {}
    for name in ("Available frames", "Allowed frames", "Collision candidates", "Lane entry candidates"):
        match = re.search(re.escape(name) + r":\s*(\[[0-9, ]+\])", prompt)
        if match:
            result[name] = json.loads(match.group(1))
    return result


class RecordedVLM:
    """Observational wrapper; exact images/prompts go unchanged to LocalVLM."""
    def __init__(self, model, output_dir: Path, save_images: bool):
        self.model = model
        self.output_dir = output_dir
        self.save_images = save_images
        self.calls = []

    def _sync(self):
        torch = self.model.torch
        if self.model.device == "cuda":
            torch.cuda.synchronize()

    def ask(self, images, prompt, max_new_tokens=128):
        index = len(self.calls)
        name = CALL_NAMES[index] if index < len(CALL_NAMES) else f"extra_{index+1}"
        call_dir = self.output_dir / f"call_{index+1}_{name}"
        call_dir.mkdir(parents=True, exist_ok=True)
        (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        if self.save_images:
            for number, image in enumerate(images):
                image.save(call_dir / f"image_{number:02d}.jpg")
        self._sync()
        if self.model.device == "cuda":
            self.model.torch.cuda.reset_peak_memory_stats()
        record = dict(call=index+1, name=name, status="running", prompt=prompt,
                      candidates=candidates_from_prompt(prompt),
                      image_sizes=[list(image.size) for image in images],
                      image_count=len(images), input_pixels=sum(image.width*image.height for image in images),
                      configured_pixel_budget=self.model.pixel_budget, max_new_tokens=max_new_tokens)
        self.calls.append(record)
        write_json(call_dir / "result.json", record)
        started = time.perf_counter()
        print(json.dumps({"event":"call_start", "video":self.output_dir.name,
                          "call":index+1, "name":name, "images":len(images)}, ensure_ascii=False), flush=True)
        try:
            answer = self.model.ask(images, prompt, max_new_tokens=max_new_tokens)
            self._sync()
            record.update(status="complete", raw_output=answer, parsed=_json_object(answer))
            tokenizer = getattr(self.model.processor, "tokenizer", None)
            if tokenizer is not None:
                record["decoded_output_tokens"] = len(tokenizer.encode(answer, add_special_tokens=False))
            return answer
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            raise
        finally:
            record["seconds"] = time.perf_counter() - started
            if self.model.device == "cuda":
                record["peak_allocated_bytes"] = self.model.torch.cuda.max_memory_allocated()
                record["peak_reserved_bytes"] = self.model.torch.cuda.max_memory_reserved()
            write_json(call_dir / "result.json", record)
            print(json.dumps({"event":"call_end", "video":self.output_dir.name, "call":index+1,
                              "status":record["status"], "seconds":record["seconds"],
                              "output":record.get("raw_output"), "error":record.get("error")},
                             ensure_ascii=False), flush=True)


def collision_metrics(prediction, truth, fps, tolerance=3):
    error = abs(int(prediction) - int(truth))
    return {"prediction":int(prediction), "truth":int(truth), "absolute_error_frames":error,
            "absolute_error_seconds":error/fps if fps and fps > 0 else None,
            "within_3_frames":error <= tolerance}


def audit_candidates(calls, truth):
    """Post-hoc only: flags whether coarse narrowing made success impossible."""
    records = []
    for call in calls:
        if call["name"] == "entry_refinement":
            continue
        options = call["candidates"]
        candidates = options.get("Collision candidates", options.get("Allowed frames", options.get("Available frames", [])))
        if not candidates:
            continue
        raw = _integer(call.get("parsed", {}).get("collision_frame"))
        records.append(dict(call=call["call"], name=call["name"], collision_candidates=candidates,
                            raw_collision_frame=raw, raw_selection_is_candidate=raw in candidates,
                            nearest_candidate_error_frames=min(abs(frame-truth) for frame in candidates),
                            any_candidate_within_3_frames=any(abs(frame-truth) <= 3 for frame in candidates),
                            truth_inside_candidate_interval=min(candidates) <= truth <= max(candidates)))
    overview_ok = bool(records and records[0]["any_candidate_within_3_frames"])
    refinement_failures = [record["name"] for record in records[1:] if not record["any_candidate_within_3_frames"]]
    final_has = bool(records and records[-1]["any_candidate_within_3_frames"])
    return {"calls":records, "overview_has_tolerance_candidate":overview_ok,
            "coarse_or_later_narrowing_trap":overview_ok and bool(refinement_failures),
            "final_candidates_make_tolerance_success_impossible":not final_has,
            "earlier_narrowing_failure_recovered_at_final":bool(refinement_failures) and final_has,
            "stages_without_tolerance_candidate":refinement_failures}


def build_summary(report):
    records = report["videos"]
    finished = [row for row in records if row["status"] == "complete"]
    failed = [row for row in records if row["status"] == "error"]
    timed_calls = [call for row in records for call in row.get("calls", []) if call.get("status") == "complete"]
    times = [row["total_seconds"] for row in finished]
    summary = dict(requested_videos=len(report["selected_ids"]), completed_videos=len(finished), failed_videos=len(failed),
                   vlm_calls_completed=len(timed_calls),
                   completed_videos_collision_correct=sum(row["vlm_collision"]["within_3_frames"] for row in finished),
                   completed_videos_motion_correct=sum(row["motion_collision"]["within_3_frames"] for row in finished),
                   all_requested_processed=len(finished)+len(failed)==len(report["selected_ids"]),
                   model_load_seconds=report.get("model_load_seconds"),
                   candidate_trap_ids=[row["ID"] for row in finished if row["candidate_audit"]["coarse_or_later_narrowing_trap"]],
                   final_candidate_failure_ids=[row["ID"] for row in finished if row["candidate_audit"]["final_candidates_make_tolerance_success_impossible"]])
    if finished:
        summary["vlm_collision_accuracy_completed_only"] = summary["completed_videos_collision_correct"]/len(finished)
        summary["motion_collision_accuracy_completed_only"] = summary["completed_videos_motion_correct"]/len(finished)
        summary["mean_video_seconds"] = float(np.mean(times))
        summary["max_video_seconds"] = max(times)
        summary["measured_mean_four_call_seconds"] = float(np.mean([sum(call["seconds"] for call in row["calls"]) for row in finished]))
        summary["runtime_scope"] = "Measured on local device only; hidden video count and L40S throughput unknown."
    if summary["all_requested_processed"] and not failed:
        summary["public_collision_accuracy_at_3_frames"] = summary.get("vlm_collision_accuracy_completed_only")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT/"artifacts/model/stage2/vlm")
    parser.add_argument("--image-root", type=Path, default=ROOT/"research/stage2_public/images")
    parser.add_argument("--labels", type=Path, default=ROOT/"Baseline/data/stage2/labels.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="+", default=None, help="Space-separated or comma-separated public IDs")
    parser.add_argument("--pixel-budget", type=int, default=1_200_000)
    parser.add_argument("--save-images", action="store_true", help="Save exact per-call diagnostic input images")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection and public inputs without loading GPU/model")
    parser.add_argument("--keep-going", action="store_true", help="Record per-video failure and continue; default fails immediately")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.pixel_budget < 1024:
        parser.error("--pixel-budget must be >=1024")
    labels = pd.read_csv(args.labels)
    if labels["ID"].duplicated().any():
        parser.error("Duplicate public IDs in labels")
    if args.ids:
        wanted = [part for item in args.ids for part in item.split(",") if part]
        missing = sorted(set(wanted) - set(labels["ID"]))
        if missing:
            parser.error(f"Unknown public IDs: {missing}")
        labels = labels[labels["ID"].isin(wanted)]
    if args.limit is not None:
        labels = labels.head(args.limit)
    if labels.empty:
        parser.error("No public examples selected")
    selections = []
    for row in labels.itertuples():
        folder = args.image_root / row.ID
        paths = sorted((path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS), key=_frame_number)
        if not paths:
            parser.error(f"No image frames for {row.ID}; run research/stage2_probe.py first")
        numbers = [_frame_number(path) for path in paths]
        if len(set(numbers)) != len(numbers):
            parser.error(f"Duplicate original frame numbers for {row.ID}")
        selections.append((row, paths))
    if args.dry_run:
        print(json.dumps({"status":"dry_run_ok", "model_dir":str(args.model_dir),
                          "pixel_budget":args.pixel_budget, "max_calls":4*len(selections),
                          "selected":[{"ID":row.ID,"frames":len(paths),"collision_truth":int(row.t_collision)} for row,paths in selections],
                          "model_loaded":False, "gpu_used":False}, indent=2))
        return
    output_dir = args.output_dir or ROOT/"artifacts/eval_stage2"/datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir/"report.json").exists():
        parser.error(f"An evaluation already exists at {output_dir}; use a fresh output directory")
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(output_dir/"evaluation.log", encoding="utf-8")])
    from solution.vlm import LocalVLM
    report = {"scope":"Public examples only; original collision labels. No ground truth fed into model.",
              "created_at":datetime.now(timezone.utc).isoformat(), "status":"loading",
              "selected_ids":[row.ID for row,_ in selections], "model_dir":str(args.model_dir.resolve()),
              "pixel_budget":args.pixel_budget, "comparison_tolerance_frames":3,
              "tolerance_note":"Public examples are 10 FPS (verified separately); hidden evaluation uses per-video time mapping.",
              "stage2_source_sha256":hashlib.sha256((ROOT/"solution/stage2.py").read_bytes()).hexdigest(),
              "vlm_source_sha256":hashlib.sha256((ROOT/"solution/vlm.py").read_bytes()).hexdigest(),
              "videos":[]}
    write_json(output_dir/"report.json", report)
    started = time.perf_counter()
    try:
        vlm = LocalVLM(args.model_dir, pixel_budget=args.pixel_budget)
    except Exception as error:
        report.update(status="model_load_error",error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
        write_json(output_dir/"report.json", report)
        raise
    report["model_load_seconds"] = time.perf_counter()-started
    report["device"] = vlm.device
    if vlm.device == "cuda":
        report["gpu_name"] = vlm.torch.cuda.get_device_name()
    report["status"] = "running"
    try:
        for row, original_paths in selections:
            start = time.perf_counter()
            record = {"ID":row.ID,"status":"running","official_collision_frame":int(row.t_collision)}
            report["videos"].append(record)
            recorded = RecordedVLM(vlm, output_dir/row.ID, args.save_images)
            try:
                motion_start = time.perf_counter()
                paths, scores, motion_side = _motion_scan(original_paths)
                record["motion_scan_seconds"] = time.perf_counter()-motion_start
                record["valid_frames"] = len(paths)
                record["skipped_frames"] = len(original_paths)-len(paths)
                record["motion_proposal_scores"] = [{"frame":_frame_number(path),"score":float(score)} for path,score in zip(paths,scores)]
                motion_frame = _frame_number(paths[int(scores.argmax())])
                record["motion_collision"] = collision_metrics(motion_frame, row.t_collision, 10.0)
                record["weak_motion_side"] = motion_side
                prediction, diagnostics = _predict_file(paths, scores, recorded)
                record.update(status="complete",prediction=prediction,diagnostics=diagnostics,
                              vlm_collision=collision_metrics(prediction["collision_frame"], row.t_collision, 10.0),
                              candidate_audit=audit_candidates(recorded.calls, int(row.t_collision)))
                if len(recorded.calls) > 4:
                    raise AssertionError("Stage2 exceeded four-call budget")
            except Exception as error:
                record.update(status="error",error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
                if not args.keep_going:
                    raise
            finally:
                record["calls"] = recorded.calls
                record["total_seconds"] = time.perf_counter()-start
                write_json(output_dir/row.ID/"video_result.json", record)
                report["summary"] = build_summary(report)
                write_json(output_dir/"report.json", report)
                print(json.dumps({"event":"video_end","ID":row.ID,"status":record["status"],
                                  "seconds":record["total_seconds"],"prediction":record.get("prediction"),
                                  "collision":record.get("vlm_collision"),
                                  "candidate_audit":record.get("candidate_audit")},ensure_ascii=False),flush=True)
        report["status"] = "complete" if all(row["status"]=="complete" for row in report["videos"]) else "complete_with_errors"
    except BaseException:
        report["status"] = "interrupted_or_error"
        raise
    finally:
        vlm.close()
        report["summary"] = build_summary(report)
        write_json(output_dir/"report.json", report)
        print(json.dumps({"report":str(output_dir/"report.json"),"summary":report["summary"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
