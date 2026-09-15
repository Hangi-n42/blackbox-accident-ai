"""Compare an isolated 4B candidate with the frozen Stage2 V2 public diagnostic.

Additional flags: --precision nf4|fp16_offload, --gpu-memory 5GiB,
--cpu-memory 12GiB, --reference-report PATH, --export-nf4 PATH, --bnb-smoke.
All ordinary public evaluator flags, including --limit and --ids, are supported.
No submission sources, prompts, frame policies, or image budgets are modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_nf4(candidate, target, sources):
    """Save the already evaluated quantized instance, never overwrite any assets."""
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"NF4 export destination already exists: {target}")
    if candidate.candidate_metadata["precision"] != "nf4":
        raise ValueError("Only an NF4 instance can be exported by this option")
    original = candidate.model_path.resolve()
    manifest_path = original/"download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != "Qwen/Qwen3-VL-4B-Instruct" or not manifest.get("revision"):
        raise ValueError("A pinned official 4B download manifest is required")
    card = (original/"README.md").read_text(encoding="utf-8")
    if "license: apache-2.0" not in card.split("---", 2)[1]:
        raise ValueError("Official model card does not declare the expected Apache-2.0 license")
    index = json.loads((original/"model.safetensors.index.json").read_text(encoding="utf-8"))
    manifest_files = {item["name"]: item for item in manifest["files"]}
    for name in set(index["weight_map"].values()):
        info = manifest_files.get(name, {})
        if not info.get("expected_lfs_sha256") or info.get("sha256") != info["expected_lfs_sha256"]:
            raise ValueError(f"Source shard is not checksum-verified in manifest: {name}")
        if (original/name).stat().st_size != info["bytes"]:
            raise ValueError(f"Source shard size changed: {name}")
    license_paths = [original/"LICENSE", original/"LICENSE-APACHE-2.0.txt",
                     ROOT/"artifacts/model/stage2/vlm/LICENSE-APACHE-2.0.txt"]
    license_path = next((path for path in license_paths if path.is_file()), None)
    if license_path is None:
        raise FileNotFoundError("Apache license text is required for export")
    target.mkdir(parents=True, exist_ok=False)
    marker = target/"EXPORT_INCOMPLETE.json"
    marker.write_text(json.dumps({"status": "saving", "source": str(original)}), encoding="utf-8")
    started = time.perf_counter()
    candidate.model.save_pretrained(str(target), safe_serialization=True, max_shard_size="2GB")
    candidate.processor.save_pretrained(str(target))
    (target/"LICENSE-APACHE-2.0.txt").write_bytes(license_path.read_bytes())
    (target/"README.original.md").write_bytes((original/"README.md").read_bytes())
    (target/"SOURCE_DOWNLOAD_MANIFEST.json").write_bytes(manifest_path.read_bytes())
    if (original/"NOTICE").is_file():
        (target/"NOTICE").write_bytes((original/"NOTICE").read_bytes())
    change_notice = (
        "# Local NF4 derivative of Qwen3-VL-4B-Instruct\n\n"
        f"Original publisher: Qwen. Source: {manifest['source']}\n"
        f"Original revision: {manifest['revision']}\n\n"
        "License: Apache-2.0; see LICENSE-APACHE-2.0.txt and README.original.md.\n\n"
        "Modification: pretrained weights were converted to bitsandbytes NF4 with "
        "FP16 compute, without double quantization, then saved with Transformers "
        "save_pretrained. No fine-tuning or public/evaluation-label training was performed. "
        "Processor assets were reserialized with save_pretrained. This derivative is "
        "locally generated and is not an official Qwen-distributed checkpoint.\n\n"
        "Reload equivalence and offline inference are NOT yet verified by export. "
        "Those checks are required before adoption.\n")
    (target/"CHANGES.md").write_text(change_notice, encoding="utf-8")
    (target/"README.md").write_text("---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-VL-4B-Instruct\n---\n\n"+change_notice,
                                  encoding="utf-8")
    metadata = {"status": "saved_unverified", "created_at": datetime.now(timezone.utc).isoformat(),
                "repository": manifest["repository"], "revision": manifest["revision"],
                "source": manifest["source"], "license": "Apache-2.0",
                "license_text_source_path": str(license_path),
                "runtime": candidate.candidate_metadata, "candidate_sources_sha256": sources,
                "reload_equivalence_verified": False, "offline_reload_verified": False,
                "files": [{"name": str(path.relative_to(target)), "bytes": path.stat().st_size,
                           "sha256": digest(path)} for path in sorted(target.rglob("*"))
                          if path.is_file() and path != marker],
                "save_and_hash_seconds": time.perf_counter()-started}
    (target/"EXPORT_MANIFEST.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    marker.unlink()
    return {"status": "saved_unverified", "path": str(target),
            "manifest_sha256": digest(target/"EXPORT_MANIFEST.json"),
            "payload_bytes": sum(row["bytes"] for row in metadata["files"]),
            "seconds": metadata["save_and_hash_seconds"],
            "reload_equivalence_verified": False, "offline_reload_verified": False}


def bnb_smoke():
    import bitsandbytes as bnb
    import torch
    import psutil

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(0)
    layer = bnb.nn.Linear4bit(256, 128, bias=False, compute_dtype=torch.float16,
                             quant_type="nf4", compress_statistics=False)
    original = layer.weight.detach().clone().to("cuda", dtype=torch.float16)
    layer = layer.to("cuda")
    x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        y = layer(x)
        reference = torch.nn.functional.linear(x, original)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()) or tuple(y.shape) != (4, 128):
        raise AssertionError("NF4 GPU smoke produced invalid output")
    return {"status": "passed", "scope": "NF4 CUDA linear operation only; no Qwen model loaded",
            "torch": torch.__version__, "bitsandbytes": bnb.__version__,
            "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "output_shape": list(y.shape), "output_dtype": str(y.dtype),
            "mean_abs_difference_from_fp16": float((y-reference).abs().float().mean()),
            "seconds": time.perf_counter()-start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "process_rss_bytes": psutil.Process().memory_info().rss}


def compare(report, reference):
    source_key = "stage2_v2_source_sha256"
    rows = {r["ID"]: r for r in reference.get("videos", []) if r["status"] == "complete"}
    pairs = []
    for row in report["videos"]:
        if row["status"] != "complete" or row["ID"] not in rows:
            continue
        prior = rows[row["ID"]]
        pairs.append({"ID": row["ID"], "candidate": row["prediction"], "reference": prior["prediction"],
                      "candidate_collision": row["vlm_collision"], "reference_collision": prior["vlm_collision"],
                      "candidate_seconds": row["total_seconds"], "reference_seconds": prior["total_seconds"],
                      "candidate_calls": len(row["calls"]), "reference_calls": len(prior["calls"])})
    return {"same_predictor_source": report.get(source_key) == reference.get(source_key),
            "same_pixel_budget": report["pixel_budget"] == reference.get("pixel_budget"),
            "same_inherited_ask_source": report["vlm_source_sha256"] == reference.get("vlm_source_sha256"),
            "reference_model_dir": reference.get("model_dir"),
            "reference_summary": reference.get("summary"), "paired_videos": pairs,
            "limitations": ["Same question/candidate policy; adaptive later prompts naturally depend on model answers.",
                            "NF4 vs FP16 comparison changes model size and numerical precision together.",
                            "Five repeatedly inspected public videos are not an independent holdout.",
                            "Entry frame, entry side, and evasion space have no official public labels."]}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--precision", choices=["nf4", "fp16_offload"], default="nf4")
    parser.add_argument("--gpu-memory", default="5GiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument("--reference-report", type=Path,
                        default=ROOT/"artifacts/eval_stage2/v2_final_public5/report.json")
    parser.add_argument("--bnb-smoke", action="store_true")
    parser.add_argument("--export-nf4", type=Path, default=None,
                        help="Export loaded NF4 model only after all five public videos complete")
    options, remaining = parser.parse_known_args()
    if options.export_nf4 is not None:
        if options.precision != "nf4":
            parser.error("--export-nf4 requires --precision nf4")
        if options.export_nf4.exists():
            parser.error("--export-nf4 destination must not exist")
    if options.bnb_smoke:
        result = bnb_smoke()
        output = ROOT/"artifacts/eval_stage2/4b_bnb_smoke.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    import psutil
    from scripts import evaluate_stage2_vlm as evaluator
    from solution.stage2_v2 import _predict_file
    from solution.vlm_candidate import CandidateVLM
    import solution.vlm as base_module

    reference = json.loads(options.reference_report.read_text(encoding="utf-8"))
    source_paths = [ROOT/p for p in ("solution/stage2_v2.py", "solution/vlm.py",
                                    "solution/vlm_candidate.py", "scripts/evaluate_stage2_4b.py")]
    sources = {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths}
    runtime = {}
    report_state = {}
    original_write = evaluator.write_json
    original_loader = base_module.LocalVLM

    def load_candidate(model_path, *, pixel_budget):
        candidate = CandidateVLM(model_path, pixel_budget=pixel_budget, precision=options.precision,
                                 gpu_memory=options.gpu_memory, cpu_memory=options.cpu_memory)
        runtime.update(candidate.candidate_metadata)
        manifest = Path(model_path)/"download_manifest.json"
        if manifest.is_file():
            runtime["download_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        if options.export_nf4 is not None:
            original_close = candidate.close

            def close_and_export():
                report = report_state.get("report", {})
                complete_ids = {row["ID"] for row in report.get("videos", []) if row["status"] == "complete"}
                expected_ids = {f"S2_{number:03d}" for number in range(1, 6)}
                try:
                    if report.get("status") == "complete" and complete_ids == expected_ids:
                        runtime["nf4_export"] = export_nf4(candidate, options.export_nf4, sources)
                    else:
                        runtime["nf4_export"] = {"status": "skipped", "reason": "All five public videos must complete"}
                except Exception as error:
                    runtime["nf4_export"] = {"status": "error", "error": f"{type(error).__name__}: {error}",
                                             "traceback": traceback.format_exc()}
                finally:
                    original_close()

            candidate.close = close_and_export
        return candidate

    def write_report(path, value):
        if isinstance(value, dict) and "selected_ids" in value:
            report_state["report"] = value
            value.update(predictor_variant="v2_final_separate_entry_direction_4b_candidate",
                         stage2_v2_source_sha256=sources["solution/stage2_v2.py"],
                         candidate_sources_sha256=sources,
                         candidate_precision=options.precision,
                         candidate_runtime=runtime,
                         process_memory=dict(psutil.Process().memory_info()._asdict()),
                         host_available_bytes=psutil.virtual_memory().available)
            value["comparison_to_2b"] = compare(value, reference)
            if path.name == "report.json":
                for source in source_paths:
                    target = path.parent/(source.stem+"_source.py")
                    if not target.exists():
                        target.write_bytes(source.read_bytes())
        original_write(path, value)

    evaluator._predict_file = _predict_file
    evaluator.CALL_NAMES = ("overview", "collision_refinement", "entry_refinement", "evasion_space")
    evaluator.write_json = write_report
    base_module.LocalVLM = load_candidate
    if not any(arg == "--model-dir" or arg.startswith("--model-dir=") for arg in remaining):
        remaining.extend(["--model-dir", str(ROOT/"artifacts/candidates/qwen3_vl_4b")])
    sys.argv = [sys.argv[0], *remaining]
    try:
        evaluator.main()
        if runtime.get("nf4_export", {}).get("status") == "error":
            raise RuntimeError(f"NF4 export failed: {runtime['nf4_export']['error']}")
    finally:
        base_module.LocalVLM = original_loader


if __name__ == "__main__":
    main()
