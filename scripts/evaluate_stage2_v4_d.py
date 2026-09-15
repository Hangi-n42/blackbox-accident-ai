"""Final D-only GPU comparison against frozen V3; no further prompt search."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_stage2_v4 import Recorder, sha, write, score_after_inference
from solution import stage2_v4_d as candidate


def main():
    output = ROOT/"research/v4_stage2/D_run"
    if output.exists():
        raise FileExistsError("Refusing to overwrite D experiment")
    frozen = json.loads((ROOT/"research/v4_stage2/frozen_D.json").read_text())
    original = json.loads((ROOT/"research/v4_stage2/frozen_inputs.json").read_text())
    protected = {**original["protected_sha256"], **frozen["protected_sha256"]}
    verify = lambda: all(sha(ROOT/name) == digest for name, digest in protected.items())
    assert verify()
    assert json.loads((ROOT/"research/v4_stage2/mock_D_contract.json").read_text())["passed"]
    model_dir = ROOT/"artifacts/candidates/qwen3_vl_4b_nf4"
    manifest = json.loads((model_dir/"EXPORT_MANIFEST.json").read_text())
    for item in manifest["files"]:
        assert sha(model_dir/item["name"]) == item["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT/"solution/stage2_v4_d.py", Path(__file__), ROOT/"scripts/test_stage2_v4_d.py"]
    for path in sources:
        (output/path.name).write_bytes(path.read_bytes())
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "scope": "D is V3 with first approach sentence changed, not C-based",
              "no_training": True, "frozen_D": frozen, "arms": {"D": {"videos": []}}, "network_attempts": 0,
              "sources_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sources},
              "recorder_note": "Recorder uses baseline question sequence names for this V3-based D; all videos in this file are D."}
    def deny(*args, **kwargs):
        report["network_attempts"] += 1
        raise RuntimeError("Offline D network access blocked")
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    try:
        with candidate.CandidateVLM(model_dir, precision="nf4") as model:
            report["model_runtime"] = model.candidate_metadata
            for folder in sorted((ROOT/"research/stage2_public/images").iterdir()):
                if not folder.is_dir():
                    continue
                paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in candidate.base.IMAGE_EXTENSIONS), key=candidate.base._frame_number)
                input_count = len(paths)
                start = time.perf_counter()
                paths, scores, _ = candidate.base._motion_scan(paths)
                scan_seconds = time.perf_counter()-start
                numbers = [candidate.base._frame_number(path) for path in paths]
                recorder = Recorder(model, output/"D"/folder.name, "baseline")
                start = time.perf_counter()
                prediction, d = candidate._predict_file(paths, scores, recorder)
                predict_seconds = time.perf_counter()-start
                assert len(recorder.calls) == 4
                assert prediction["collision_frame"] in numbers and prediction["entry_frame"] in numbers
                report["arms"]["D"]["videos"].append({"ID": folder.name, "prediction": prediction, "diagnostics": d,
                     "frame_numbers": numbers, "motion_scores": scores.tolist(), "skipped_images": input_count-len(paths),
                     "scan_seconds": scan_seconds, "predict_seconds": predict_seconds, "calls": recorder.calls})
                write(output/"report.json", report)
        score_after_inference(report)
        prior_report = json.loads((ROOT/"research/v4_stage2/public_run/report.json").read_text(encoding="utf-8"))
        baseline = {row["ID"]: row for row in prior_report["arms"]["baseline"]["videos"]}
        report["baseline_d_comparison"] = []
        for row in report["arms"]["D"]["videos"]:
            prior = baseline[row["ID"]]
            first, old_first = row["calls"][0], prior["calls"][0]
            replacement_only = first["prompt"] == old_first["prompt"].replace(candidate.ORIGINAL, candidate.REPLACEMENT, 1)
            same_visual = all(first[key] == old_first[key] for key in ("input_images", "bounded_image_sizes", "max_new_tokens"))
            same_collision = row["prediction"]["collision_frame"] == prior["prediction"]["collision_frame"]
            assert replacement_only and same_visual and same_collision
            comparison = {"ID": row["ID"], "baseline": prior["prediction"], "D": row["prediction"],
                          "first_prompt_exact_replacement": replacement_only, "first_visual_budget_identical": same_visual,
                          "final_motion_collision_identical": same_collision, "later_calls": []}
            for old_call, call in zip(prior["calls"][1:], row["calls"][1:]):
                comparison["later_calls"].append({"name": call["name"], "same_prompt": old_call["prompt"] == call["prompt"],
                     "same_images": old_call["input_images"] == call["input_images"], "same_output": old_call["raw_output"] == call["raw_output"],
                     "same_candidates": old_call["offered_frames"] == call["offered_frames"]})
            comparison["entry_side_from_valid_raw_response"] = row["diagnostics"]["coarse"].get("entry_side") in ("LEFT", "RIGHT")
            report["baseline_d_comparison"].append(comparison)
        assert verify() and report["network_attempts"] == 0
        for item in manifest["files"]:
            assert sha(model_dir/item["name"]) == item["sha256"]
        report.update(status="complete", protected_model_assets_unchanged=True, protected_AB_C_D_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write(output/"report.json", report)
    print(json.dumps({"status": report["status"], "summary": report["arms"]["D"]["summary"], "comparison": report["baseline_d_comparison"]}), flush=True)


if __name__ == "__main__":
    main()
