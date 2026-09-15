"""Evaluate C once against saved A, preserving all A/B sources and results."""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
import socket
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_stage2_v4 import Recorder, sha, write, score_after_inference
from solution import stage2_v4_c as candidate


def main():
    output = ROOT/"research/v4_stage2/C_run"
    if output.exists():
        raise FileExistsError("Refusing to overwrite C experiment")
    original = json.loads((ROOT/"research/v4_stage2/frozen_inputs.json").read_text())
    frozen = json.loads((ROOT/"research/v4_stage2/frozen_C.json").read_text())
    protected = {**original["protected_sha256"], **frozen["protected_sha256"]}
    verify = lambda: all(sha(ROOT/name) == digest for name, digest in protected.items())
    assert verify()
    assert json.loads((ROOT/"research/v4_stage2/mock_C_contract.json").read_text())["passed"]
    model_dir = ROOT/"artifacts/candidates/qwen3_vl_4b_nf4"
    manifest = json.loads((model_dir/"EXPORT_MANIFEST.json").read_text())
    for item in manifest["files"]:
        assert sha(model_dir/item["name"]) == item["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT/"solution/stage2_v4_c.py", Path(__file__), ROOT/"scripts/test_stage2_v4_c.py"]
    for path in sources:
        (output/path.name).write_bytes(path.read_bytes())
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "no_training": True,
              "scope": "C public development diagnostic, not independent accuracy", "frozen_C": frozen,
              "sources_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sources},
              "network_attempts": 0, "arms": {"C": {"videos": []}}}
    def deny(*args, **kwargs):
        report["network_attempts"] += 1
        raise RuntimeError("Offline C network access blocked")
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
                start = time.perf_counter()
                input_count = len(paths)
                paths, scores, _ = candidate.base._motion_scan(paths)
                scan_seconds = time.perf_counter()-start
                numbers = [candidate.base._frame_number(p) for p in paths]
                recorder = Recorder(model, output/"C"/folder.name, "C")
                start = time.perf_counter()
                prediction, d = candidate._predict_file(paths, scores, recorder)
                predict_seconds = time.perf_counter()-start
                assert len(recorder.calls) == d["calls"] <= 4
                assert prediction["entry_frame"] in numbers and prediction["collision_frame"] in numbers
                report["arms"]["C"]["videos"].append({"ID": folder.name, "prediction": prediction, "diagnostics": d,
                     "frame_numbers": numbers, "motion_scores": scores.tolist(), "skipped_images": input_count-len(paths),
                     "scan_seconds": scan_seconds, "predict_seconds": predict_seconds, "calls": recorder.calls})
                write(output/"report.json", report)
        score_after_inference(report)
        saved_a = json.loads((ROOT/"research/v4_stage2/public_run/report.json").read_text(encoding="utf-8"))["arms"]["A"]
        a = {row["ID"]: row for row in saved_a["videos"]}
        report["a_c_contract"] = []
        for row in report["arms"]["C"]["videos"]:
            prior = a[row["ID"]]
            fields = all(prior["prediction"][key] == row["prediction"][key] for key in ("collision_frame", "entry_side", "evasion_space"))
            calls = all(all(x[key] == y[key] for key in ("prompt", "raw_output", "input_images", "bounded_image_sizes", "max_new_tokens")) for x, y in zip(prior["calls"], row["calls"][:3]))
            report["a_c_contract"].append({"ID": row["ID"], "other_three_fields_identical": fields, "first_three_calls_identical": calls})
            assert fields and calls
        assert verify() and report["network_attempts"] == 0
        for item in manifest["files"]:
            assert sha(model_dir/item["name"]) == item["sha256"]
        report.update(status="complete", protected_model_assets_unchanged=True, protected_AB_and_C_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write(output/"report.json", report)
    print(json.dumps({"status": report["status"], "summary": report["arms"]["C"]["summary"]}), flush=True)


if __name__ == "__main__":
    main()
