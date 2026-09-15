"""Fresh-process, socket-blocked NF4 reload and exact public-output comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT/"artifacts/candidates/qwen3_vl_4b_nf4")
    parser.add_argument("--reference-report", type=Path, default=ROOT/"artifacts/eval_stage2/4b_nf4_public5/report.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"artifacts/eval_stage2/4b_nf4_offline_reload")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve prior verification")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(repr(args[1:] if args and isinstance(args[0], socket.socket) else args))
        raise RuntimeError("Network socket connections are blocked for offline verification")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    source = json.loads(args.reference_report.read_text(encoding="utf-8"))
    manifest = json.loads((args.model_dir/"EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = args.model_dir/item["name"]
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise AssertionError(f"NF4 exported file checksum mismatch: {item['name']}")
    from scripts import evaluate_stage2_4b
    sys.argv = [sys.argv[0], "--precision", "nf4", "--model-dir", str(args.model_dir),
                "--output-dir", str(args.output_dir)]
    evaluate_stage2_4b.main()
    reloaded = json.loads((args.output_dir/"report.json").read_text(encoding="utf-8"))
    original_rows = {row["ID"]: row for row in source["videos"]}
    comparisons = []
    for row in reloaded["videos"]:
        before = original_rows[row["ID"]]
        comparisons.append({"ID": row["ID"], "prediction_equal": row["prediction"] == before["prediction"],
                            "call_count_equal": len(row["calls"]) == len(before["calls"]) == 4,
                            "calls": [{"call": current["call"],
                                       "prompt_equal": current["prompt"] == prior["prompt"],
                                       "raw_output_equal": current["raw_output"] == prior["raw_output"],
                                       "image_sizes_equal": current["image_sizes"] == prior["image_sizes"],
                                       "max_new_tokens_equal": current["max_new_tokens"] == prior["max_new_tokens"]}
                                      for current, prior in zip(row["calls"], before["calls"])]})
    exact = (reloaded["status"] == "complete"
             and set(original_rows) == {row["ID"] for row in reloaded["videos"]}
             and all(row["prediction_equal"] and row["call_count_equal"]
                     and all(all(value for key, value in call.items() if key != "call") for call in row["calls"])
                     for row in comparisons))
    verification = {"status": "passed" if exact and not attempts else "failed",
                    "scope": "Exact public-five output equivalence and offline reload only; no generalization claim.",
                    "pid": os.getpid(), "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
                    "transformers_offline": os.environ["TRANSFORMERS_OFFLINE"],
                    "socket_connections_blocked": True, "blocked_attempts": attempts,
                    "export_files_verified": len(manifest["files"]),
                    "export_manifest_sha256": digest(args.model_dir/"EXPORT_MANIFEST.json"),
                    "reference_report_sha256": digest(args.reference_report),
                    "reloaded_report_sha256": digest(args.output_dir/"report.json"),
                    "verification_source_sha256": digest(Path(__file__)),
                    "all_predictions_and_call_text_equal": exact,
                    "same_predictor_source": source["stage2_v2_source_sha256"] == reloaded["stage2_v2_source_sha256"],
                    "same_ask_source": source["vlm_source_sha256"] == reloaded["vlm_source_sha256"],
                    "comparisons": comparisons, "reload_summary": reloaded["summary"]}
    (args.output_dir/"reload_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)
    if verification["status"] != "passed":
        raise AssertionError("Offline reload equivalence failed; inspect reload_verification.json")


if __name__ == "__main__":
    main()
