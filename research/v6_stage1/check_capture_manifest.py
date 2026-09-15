"""Read-only metadata/lineage gate. PASS is not proof of physical capture or accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

SPLITS = {"development", "holdout", "unassigned"}
KINDS = {"original_camera", "physical_display_recapture", "synthetic_transform", "unknown"}
REQUIRED = {"record_id", "source_group", "content_id", "original_camera_id",
            "domain", "acquisition_kind", "label", "status", "split", "path",
            "sha256", "parent_original_id", "display_id", "recapture_camera_id",
            "media", "rights", "evidence", "annotation", "capture_conditions"}


def audit(data, base, verify_files=True):
    errors, warnings, eligible = [], [], []
    if not isinstance(data, dict):
        return {"status": "NOT_READY", "errors": ["manifest must be an object"], "accepted_counts": None}
    rows = data.get("records", [])
    if not isinstance(rows, list):
        rows = []
        errors.append("records must be a list")
    protocol = data.get("protocol", {})
    if not isinstance(protocol, dict):
        protocol = {}
        errors.append("protocol must be an object")
    if data.get("schema_version") != 1:
        errors.append("unsupported schema version")
    if not protocol.get("frozen_at") or not protocol.get("selection_rule"):
        errors.append("protocol selection/freeze missing")
    allowed_device_fields = {"original_camera_id", "display_id", "recapture_camera_id"}
    device_fields = protocol.get("device_holdout_fields", [])
    if not isinstance(device_fields, list) or any(f not in allowed_device_fields for f in device_fields):
        errors.append("invalid device_holdout_fields")
        device_fields = []
    if not rows:
        errors.append("no acquired records; preparation is not acquisition")
    ids = {}
    def fail(rid, message):
        errors.append(f"{rid}: {message}")
    def check_file(rid, item):
        name, expected = item.get("path"), item.get("sha256")
        if not name or not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            fail(rid, "missing path or lowercase SHA256")
            return
        relative = Path(name)
        path = (base / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(base.resolve()):
            fail(rid, "path must stay inside the manifest directory")
            return
        if verify_files:
            if not path.is_file():
                fail(rid, f"missing file {name}")
            else:
                with path.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != expected:
                    fail(rid, f"SHA mismatch {name}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            fail(index, "record must be an object")
            continue
        rid = row.get("record_id", f"row{index}")
        before = len(errors)
        missing = REQUIRED - row.keys()
        if missing:
            fail(rid, f"missing fields {sorted(missing)}")
        if any(not isinstance(row.get(k), dict) for k in ("media", "rights", "annotation", "capture_conditions")) or not isinstance(row.get("evidence"), list) or any(not isinstance(e, dict) for e in row.get("evidence", [])):
            fail(str(rid), "nested metadata objects/evidence list malformed")
            continue
        if not isinstance(rid, str) or not rid or rid in ids:
            fail(str(rid), "record_id missing/duplicate")
            continue
        ids[rid] = row
        for field in ("source_group", "content_id", "original_camera_id"):
            if not row.get(field):
                fail(rid, f"{field} must be known")
        kind = row.get("acquisition_kind")
        if kind not in KINDS or row.get("split") not in SPLITS:
            fail(rid, "invalid acquisition kind or split")
        if row.get("split") == "unassigned":
            fail(rid, "split not frozen")
        if row.get("status") != "human_verified":
            fail(rid, "not human_verified; do not count as acquired verified footage")
        annotation = row.get("annotation") or {}
        if not all(annotation.get(k) for k in ("reviewer", "reviewed_at", "label_basis")):
            fail(rid, "human review provenance missing")
        if row.get("split") == "holdout" and annotation.get("model_predictions_seen") is not False:
            fail(rid, "holdout annotation must precede model prediction review")
        rights = row.get("rights") or {}
        required_use = protocol.get("required_use")
        if not required_use or rights.get("status") != "verified" or not rights.get("basis") or required_use not in rights.get("permitted_uses", []):
            fail(rid, "use permission not verified for this protocol")
        media = row.get("media") or {}
        if not media.get("codec") or not all(isinstance(media.get(k), int) and media[k] > 0 for k in ("width", "height", "decoded_frame_count")):
            fail(rid, "codec/dimensions/decoded frame count missing")
        if not media.get("timing_basis"):
            fail(rid, "timing basis missing; nominal FPS must not imply measured PTS")
        check_file(rid, row)
        evidence = row.get("evidence") or []
        for item in evidence:
            check_file(rid, item)
        kinds = {item.get("kind") for item in evidence}
        if kind == "physical_display_recapture":
            if row.get("label") != "RERECORDED" or not all(row.get(k) for k in ("parent_original_id", "display_id", "recapture_camera_id", "capture_conditions")):
                fail(rid, "physical recapture pairing/device/conditions incomplete")
            if not {"setup_photo_or_video", "capture_log"}.issubset(kinds):
                fail(rid, "physical capture needs setup evidence and capture log")
        elif kind == "original_camera":
            if row.get("label") != "ORIGINAL" or row.get("parent_original_id") is not None or row.get("display_id") is not None or row.get("recapture_camera_id") is not None:
                fail(rid, "original capture label/lineage/device inconsistency")
            if "source_provenance" not in kinds:
                fail(rid, "original camera provenance evidence missing")
        else:
            warnings.append(f"{rid}: {kind} is not a physical road recapture/original validation pair")
        if len(errors) == before and row.get("domain") == "road" and kind in {"physical_display_recapture", "original_camera"}:
            eligible.append(rid)
    for rid, row in ids.items():
        if row.get("acquisition_kind") == "physical_display_recapture":
            parent = ids.get(row.get("parent_original_id"))
            if not parent or parent.get("acquisition_kind") != "original_camera":
                fail(rid, "parent original record missing")
                continue
            for field in ("source_group", "content_id", "original_camera_id", "domain", "split"):
                if parent.get(field) != row.get(field):
                    fail(rid, f"parent mismatch in {field}")
    groups = ("source_group", "content_id", "sha256", *device_fields)
    asset_labels = {}
    for rid, row in ids.items():
        digest = row.get("sha256")
        if digest:
            asset_labels.setdefault(digest, set()).add(row.get("label"))
    if any(len(labels) > 1 for labels in asset_labels.values()):
        errors.append("identical asset SHA has conflicting class labels")
    for field in groups:
        seen = {}
        for row in ids.values():
            value, split = row.get(field), row.get("split")
            if value and split in {"development", "holdout"}:
                seen.setdefault(value, set()).add(split)
        for value, splits in seen.items():
            if len(splits) > 1:
                fail(field, f"development/holdout overlap {value}")
    present_splits = {r.get("split") for r in ids.values()}
    if not {"development", "holdout"}.issubset(present_splits):
        warnings.append("development-versus-holdout separation not empirically tested: one side absent")
    counts = {"road_originals": 0, "road_physical_recaptures": 0}
    for rid in eligible:
        counts["road_originals" if ids[rid]["acquisition_kind"] == "original_camera" else "road_physical_recaptures"] += 1
    if not all(counts.values()):
        errors.append("no complete verified road original/physical-recapture classes")
    if not verify_files:
        warnings.append("metadata-only mode does not verify asset existence or hash")
    return {"status": "METADATA_CHECKS_PASS" if not errors and verify_files else "NOT_READY",
            "errors": errors, "warnings": warnings, "provisional_row_eligible_counts": counts,
            "accepted_counts": counts if not errors and verify_files else None,
            "limitations": "Metadata/file checks do not prove real capture, truthful rights, semantic non-overlap, decoding correctness, statistical adequacy, or model performance."}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    result = audit(json.loads(args.manifest.read_text(encoding='utf-8')), args.manifest.resolve().parent, not args.metadata_only)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        with args.report.open('x', encoding='utf-8') as f:
            f.write(text)
    print(text)
    raise SystemExit(0 if result['status'] == 'METADATA_CHECKS_PASS' else 2)
