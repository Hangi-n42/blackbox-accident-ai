"""Sequential frozen V3/A/B public diagnostics; no training or submission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from solution import stage2_v4 as candidate
from solution import stage2_motion_collision as baseline
from solution.vlm_candidate import CandidateVLM


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class Recorder:
    def __init__(self, model, directory, variant):
        self.model, self.directory, self.variant = model, directory, variant
        self.calls = []

    def ask(self, images, prompt, max_new_tokens=128):
        names = ["overview", "collision_fine", "entry_coarse", "space"] if self.variant == "baseline" else ["overview", "entry_coarse", "space", "entry_fine"]
        index = len(self.calls)
        target = self.directory/f"call_{index+1}_{names[index]}"
        target.mkdir(parents=True, exist_ok=False)
        offered = {name: json.loads(value) for name, value in re.findall(r"(Available frames|Allowed frames|Lane entry candidates):\s*(\[[0-9, ]+\])", prompt)}
        bounded = []
        budget = max(1024, self.model.pixel_budget//len(images))
        for number, image in enumerate(images):
            image.save(target/f"input_{number}.png")
            scale = min(1., math.sqrt(budget/(image.width*image.height)))
            bounded.append([max(32, int(image.width*scale)//32*32), max(32, int(image.height*scale)//32*32)])
        row = {"call": index+1, "name": names[index], "prompt": prompt, "offered_frames": offered,
               "input_images": [{"size": list(image.size), "raw_rgb_sha256": hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()} for image in images],
               "bounded_image_sizes": bounded, "bounded_pixel_total": sum(w*h for w, h in bounded),
               "pixel_budget": self.model.pixel_budget, "max_new_tokens": max_new_tokens, "status": "running"}
        self.calls.append(row)
        self.model.torch.cuda.synchronize()
        self.model.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        write(target/"result.json", row)
        try:
            answer = self.model.ask(images, prompt, max_new_tokens=max_new_tokens)
            self.model.torch.cuda.synchronize()
            row.update(raw_output=answer, status="complete")
            return answer
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row.update(seconds=time.perf_counter()-started,
                       gpu_peak_allocated_bytes=self.model.torch.cuda.max_memory_allocated(),
                       gpu_peak_reserved_bytes=self.model.torch.cuda.max_memory_reserved())
            write(target/"result.json", row)
            print(json.dumps({"variant": self.variant, "ID": self.directory.name, "call": row["name"],
                              "seconds": row["seconds"], "output": row.get("raw_output")}), flush=True)


def score_after_inference(report):
    # Official labels and original PTS are consulted ONLY after inference finishes.
    import av
    labels = pd.read_csv(ROOT/"Baseline/data/stage2/labels.csv")
    maps, truth = {}, {}
    for row in labels.itertuples():
        with av.open(str(ROOT/"Baseline/data/stage2"/row.path)) as container:
            times = [float(frame.pts*frame.time_base) for frame in container.decode(video=0)]
        maps[row.ID] = times
        truth[row.ID] = int(row.t_collision)
    for variant, arm in report["arms"].items():
        passed = 0
        for row in arm["videos"]:
            times, gt = maps[row["ID"]], truth[row["ID"]]
            if not all(0 <= n < len(times) for n in row["frame_numbers"]):
                raise ValueError("Public filenames do not map to decoded source frame indices")
            pred = row["prediction"]["collision_frame"]
            error = abs(times[pred]-times[gt])
            row["official_collision_diagnostic"] = {"provided_label_frame": gt, "prediction": pred,
                 "error_seconds_from_source_pts": error, "within_0_3_seconds": error <= .3+1e-9,
                 "provided_label_is_not_reannotated": True}
            passed += int(error <= .3+1e-9)
            audits = []
            for call in row["calls"]:
                for kind, options in call["offered_frames"].items():
                    if call["name"] not in {"overview", "collision_fine"}:
                        continue
                    nearest = min(abs(times[n]-times[gt]) for n in options)
                    audits.append({"call": call["name"], "candidate_kind": kind, "nearest_gt_error_seconds": nearest,
                                   "candidate_within_0_3": nearest <= .3+1e-9})
            row["collision_candidate_oracle"] = {"vlm_candidate_sets": audits,
               "final_output_set": [pred], "final_motion_singleton_within_0_3": error <= .3+1e-9,
               "note": "All variants choose final collision from the same motion singleton; VLM coverage does not change final choice."}
        arm["summary"] = {"videos": len(arm["videos"]), "official_collision_correct": passed,
                          "official_collision_accuracy": passed/len(arm["videos"]),
                          "mean_predict_seconds": float(np.mean([r["predict_seconds"] for r in arm["videos"]])),
                          "mean_scan_seconds": float(np.mean([r["scan_seconds"] for r in arm["videos"]])),
                          "mean_call_seconds": float(np.mean([sum(c["seconds"] for c in r["calls"]) for r in arm["videos"]])),
                          "calls_total": sum(len(r["calls"]) for r in arm["videos"]),
                          "no_official_entry_side_space_gt": True}
    report["source_time_maps"] = {key: {"decoded_frame_pts_seconds": value,
             "mapping_scope": "public extracted filename number = original decoded frame index only"} for key, value in maps.items()}
    if "A" in report["arms"] and "B" in report["arms"]:
        a = {row["ID"]: row for row in report["arms"]["A"]["videos"]}
        report["a_b_contract"] = []
        for b in report["arms"]["B"]["videos"]:
            prior = a[b["ID"]]
            fields = all(prior["prediction"][key] == b["prediction"][key] for key in ("collision_frame", "entry_side", "evasion_space"))
            initial = all(all(x[key] == y[key] for key in ("prompt", "raw_output", "input_images", "bounded_image_sizes", "max_new_tokens")) for x, y in zip(prior["calls"], b["calls"][:3]))
            report["a_b_contract"].append({"ID": b["ID"], "other_three_fields_identical": fields,
                                            "first_three_calls_identical": initial})
            if not fields or not initial:
                raise AssertionError("A/B retained predictions or initial calls diverged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=["baseline", "A", "B"], default=["baseline", "A", "B"])
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT/"research/v4_stage2/public_run")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory exists; refusing to overwrite experiment")
    frozen = json.loads((ROOT/"research/v4_stage2/frozen_inputs.json").read_text())
    def frozen_ok():
        return all(sha(ROOT/name) == value for name, value in frozen["protected_sha256"].items())
    assert frozen_ok(), "Frozen source changed"
    contracts = json.loads((ROOT/"research/v4_stage2/mock_contract.json").read_text())
    assert contracts["passed"] and contracts["candidate_sha256"] == sha(ROOT/"solution/stage2_v4.py")
    model_dir = ROOT/"artifacts/candidates/qwen3_vl_4b_nf4"
    manifest = json.loads((model_dir/"EXPORT_MANIFEST.json").read_text())
    for item in manifest["files"]:
        path = model_dir/item["name"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], f"Changed model asset {path}"
    folders = sorted(path for path in (ROOT/"research/stage2_public/images").iterdir() if path.is_dir())
    if args.ids:
        folders = [path for path in folders if path.name in args.ids]
    if args.limit:
        folders = folders[:args.limit]
    if not folders:
        parser.error("No requested input folders")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT/name for name in ("solution/stage2_v4.py", "scripts/evaluate_stage2_v4.py", "scripts/test_stage2_v4.py")]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "scope": "repeatedly inspected public diagnostic, not independent accuracy",
              "no_training": True, "model_dir": str(model_dir), "model_manifest_sha256": sha(model_dir/"EXPORT_MANIFEST.json"),
              "sources_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}, "frozen_inputs": frozen,
              "arms": {}, "network_attempts": 0}
    for path in sources:
        (args.output/path.name).write_bytes(path.read_bytes())
    def deny_network(*args, **kwargs):
        report["network_attempts"] += 1
        raise RuntimeError("Network access blocked in offline diagnostic")
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    try:
        with CandidateVLM(model_dir, precision="nf4") as model:
            report["model_runtime"] = model.candidate_metadata
            for variant in args.variants:
                report["arms"][variant] = {"videos": []}
                for folder in folders:
                    paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in candidate.base.IMAGE_EXTENSIONS), key=candidate.base._frame_number)
                    before_count = len(paths)
                    started = time.perf_counter()
                    paths, scores, _ = candidate.base._motion_scan(paths)
                    scan_seconds = time.perf_counter()-started
                    numbers = [candidate.base._frame_number(path) for path in paths]
                    recorder = Recorder(model, args.output/variant/folder.name, variant)
                    started = time.perf_counter()
                    prediction, diagnostics = baseline._predict_file(paths, scores, recorder) if variant == "baseline" else candidate._predict_file(paths, scores, recorder, variant=variant)
                    predict_seconds = time.perf_counter()-started
                    assert prediction["collision_frame"] in numbers and prediction["entry_frame"] in numbers
                    assert len(recorder.calls) == (4 if variant == "baseline" else diagnostics["calls"])
                    row = {"ID": folder.name, "prediction": prediction, "diagnostics": diagnostics,
                           "frame_numbers": numbers, "motion_scores": scores.tolist(), "skipped_images": before_count-len(paths),
                           "scan_seconds": scan_seconds, "predict_seconds": predict_seconds, "calls": recorder.calls}
                    report["arms"][variant]["videos"].append(row)
                    write(args.output/"report.json", report)
        score_after_inference(report)
        report.update(status="complete", protected_sources_unchanged=frozen_ok())
        assert report["protected_sources_unchanged"] and report["network_attempts"] == 0
        for item in manifest["files"]:
            assert sha(model_dir/item["name"]) == item["sha256"]
        report["protected_model_assets_unchanged"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write(args.output/"report.json", report)
    print(json.dumps({"status": report["status"], "summaries": {name: arm["summary"] for name, arm in report["arms"].items()}}), flush=True)


if __name__ == "__main__":
    main()
