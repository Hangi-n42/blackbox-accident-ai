"""40-call maximum: actual extracted package V3/D on identical public_eval RGB.

No workspace solution imports, model/prompt changes, or old-result overwrites.
"""
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
PACKAGE = ROOT/"artifacts/submissions/verify_v4"
OUTPUT = ROOT/"research/v4_stage2/repro_run"
DATA = ROOT/"artifacts/public_eval/stage2/images"
CODE = PACKAGE/"model/stage2/code"
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
    os.environ[key] = "1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from solution import stage2_motion_collision as baseline
from solution import stage2_v4_d as candidate


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class Recorder:
    def __init__(self, model, folder, arm):
        self.model, self.folder, self.arm = model, folder, arm
        self.calls = []

    def ask(self, images, prompt, max_new_tokens=128):
        index = len(self.calls)
        if index >= 4:
            raise RuntimeError("Four-call per-file limit exceeded")
        name = ("overview", "collision_fine", "entry_coarse", "space")[index]
        destination = self.folder/f"call_{index+1}_{name}"
        destination.mkdir(parents=True, exist_ok=False)
        budget = max(1024, self.model.pixel_budget//len(images))
        bounded = []
        for index, image in enumerate(images):
            image.save(destination/f"input_{index}.png")
            scale = min(1., math.sqrt(budget/(image.width*image.height)))
            bounded.append([max(32, int(image.width*scale)//32*32), max(32, int(image.height*scale)//32*32)])
        row = {"name": name, "prompt": prompt, "max_new_tokens": max_new_tokens,
               "input_images": [{"size": list(image.size), "rgb_sha256": hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()} for image in images],
               "bounded_image_sizes": bounded, "pixel_budget": self.model.pixel_budget,
               "offered_frames": {label: json.loads(value) for label, value in re.findall(r"(Available frames|Allowed frames|Lane entry candidates):\s*(\[[0-9, ]+\])", prompt)},
               "status": "running"}
        self.calls.append(row)
        self.model.torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            answer = self.model.ask(images, prompt, max_new_tokens=max_new_tokens)
            self.model.torch.cuda.synchronize()
            row.update(status="complete", raw_output=answer)
            return answer
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["seconds"] = time.perf_counter()-start
            write(destination/"result.json", row)
            print(json.dumps({"arm": self.arm, "ID": self.folder.name, "call": name, "output": row.get("raw_output")}), flush=True)


def main():
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite reproduction")
    assert Path(candidate.__file__).resolve().is_relative_to(CODE)
    assert Path(baseline.__file__).resolve().is_relative_to(CODE)
    source_paths = sorted((CODE/"solution").glob("*.py"))
    before = {path.relative_to(PACKAGE).as_posix(): sha(path) for path in source_paths}
    model_dir = PACKAGE/"model/stage2/vlm"
    manifest = json.loads((model_dir/"EXPORT_MANIFEST.json").read_text())
    for row in manifest["files"]:
        assert sha(model_dir/row["name"]) == row["sha256"]
    corrected_scope = ROOT/"research/v4_validation_scope_corrected.json"
    prior_scope_sha = sha(corrected_scope)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
              "scope": "actual packaged V3/D same-input reproduction; not new prompt search",
              "package": str(PACKAGE), "input_root": str(DATA), "source_sha256_before": before,
              "script_sha256": sha(Path(__file__)), "corrected_validation_scope_sha256": prior_scope_sha,
              "network_attempts": 0, "arms": {}}
    def deny(*args, **kwargs):
        report["network_attempts"] += 1
        raise RuntimeError("Offline reproduction blocked a network request")
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    try:
        with candidate.CandidateVLM(model_dir, precision="nf4") as model:
            report["model_runtime"] = model.candidate_metadata
            report["runtime_settings"] = {"torch_threads": model.torch.get_num_threads(),
                 "torch_interop_threads": model.torch.get_num_interop_threads(),
                 "deterministic_algorithms": model.torch.are_deterministic_algorithms_enabled(),
                 "cudnn_deterministic": model.torch.backends.cudnn.deterministic,
                 "cudnn_benchmark": model.torch.backends.cudnn.benchmark,
                 "cuda_matmul_allow_tf32": model.torch.backends.cuda.matmul.allow_tf32}
            for arm, function in (("V3", baseline._predict_file), ("D", candidate._predict_file)):
                report["arms"][arm] = []
                for folder in sorted(path for path in DATA.iterdir() if path.is_dir()):
                    paths = sorted((path for path in folder.iterdir() if path.suffix.lower() in candidate.base.IMAGE_EXTENSIONS), key=candidate.base._frame_number)
                    start = time.perf_counter()
                    paths, scores, _ = candidate.base._motion_scan(paths)
                    scan_seconds = time.perf_counter()-start
                    recorder = Recorder(model, OUTPUT/arm/folder.name, arm)
                    start = time.perf_counter()
                    prediction, diagnostics = function(paths, scores, recorder)
                    elapsed = time.perf_counter()-start
                    assert len(recorder.calls) == 4
                    report["arms"][arm].append({"ID": folder.name, "prediction": prediction, "diagnostics": diagnostics,
                        "input_files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in paths],
                        "frame_numbers": [candidate.base._frame_number(path) for path in paths],
                        "motion_scores": scores.tolist(), "scan_seconds": scan_seconds, "predict_seconds": elapsed,
                        "calls": recorder.calls})
                    write(OUTPUT/"report.json", report)
                pd.DataFrame([dict(ID=row["ID"], **row["prediction"]) for row in report["arms"][arm]], columns=candidate.COLUMNS).to_csv(OUTPUT/f"{arm}_predictions.csv", index=False)
        for name, module in list(sys.modules.items()):
            if name.startswith("solution") and getattr(module, "__file__", None):
                assert Path(module.__file__).resolve().is_relative_to(CODE), (name, module.__file__)
        actual = pd.read_csv(ROOT/"artifacts/submissions/verify_v4_results/stage2.csv")
        reproduced = pd.read_csv(OUTPUT/"D_predictions.csv")
        pd.testing.assert_frame_equal(actual, reproduced)
        report["same_input_full_package_D_exact_match"] = True
        comparison = []
        for v3, d in zip(report["arms"]["V3"], report["arms"]["D"]):
            assert v3["ID"] == d["ID"]
            comparison.append({"ID": d["ID"], "V3": v3["prediction"], "D": d["prediction"],
                "changed_fields": [key for key in d["prediction"] if v3["prediction"][key] != d["prediction"][key]]})
        report["same_input_V3_D_comparison"] = comparison
        report["source_sha256_after"] = {path.relative_to(PACKAGE).as_posix(): sha(path) for path in source_paths}
        assert report["source_sha256_after"] == before
        assert sha(corrected_scope) == prior_scope_sha
        for row in manifest["files"]:
            assert sha(model_dir/row["name"]) == row["sha256"]
        assert report["network_attempts"] == 0
        total_calls = sum(len(row["calls"]) for rows in report["arms"].values() for row in rows)
        assert total_calls <= 40
        report.update(status="complete", calls_total=total_calls, protected_sources_model_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write(OUTPUT/"report.json", report)
    print(json.dumps({"status": report["status"], "full_package_D_exact_match": report["same_input_full_package_D_exact_match"], "comparison": report["same_input_V3_D_comparison"]}), flush=True)


if __name__ == "__main__":
    main()
