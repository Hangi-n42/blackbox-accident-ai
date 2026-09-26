"""Fix a new CCD cohort using metadata only; no imagery or predictions are read."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'artifacts/stage2_goal_20260920/ccd_intake'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert not (HERE / 'selection.json').exists()
old_selection = read(OLD / 'selection.json')['selected']
public_audit_path = OLD.parent / 'ccd_public_overlap.json'
public_groups = read(public_audit_path)['public_source_groups']
excluded = set(public_groups) | {r['source_group'] for r in old_selection}
members = {Path(r['name']).stem: r for r in read(OLD / 'zip_members.json')}
eligible = []
for line in (OLD / 'annotation.txt').read_text().splitlines():
    vid, rest = line.split(',', 1)
    labels, start, group, day, weather, ego = rest.rsplit(',', 5)
    source_group = 'CCD_YT_' + group
    if ego != 'Yes' or source_group in excluded:
        continue
    sid = 'CCD_' + vid
    eligible.append(dict(ID=sid, video_id=vid, labels=json.loads(labels), original_startframe=start,
                         source_group=source_group, day=day, weather=weather, ego=ego,
                         member=members[vid], order_sha256=hashlib.sha256(sid.encode('utf-8')).hexdigest()))
ordered = sorted(eligible, key=lambda r: (r['order_sha256'], r['ID']))
selected, used = [], set()
for row in ordered:
    if row['source_group'] in used:
        continue
    selected.append(row)
    used.add(row['source_group'])
    if len(selected) == 12:
        break
assert len(selected) == len(used) == 12 and not used & excluded
assert not {r['ID'] for r in selected} & {r['ID'] for r in old_selection}
plan = dict(status='fixed_before_download_visual_review_or_new_predictions', created_utc=datetime.now(timezone.utc).isoformat(),
            selection='SHA256(UTF-8 full CCD_ID) ascending, first 12 unique nonexcluded source groups with official ego=Yes',
            previous_order_limit='Old intake recorded salted order without the salt. This explicitly specified new order is not claimed to continue that unreproducible ranking.',
            excluded_public_source_groups=public_groups, excluded_prior_source_groups=sorted({r['source_group'] for r in old_selection}),
            excluded_source_groups=sorted(excluded), eligible_metadata_clips=len(eligible), eligible_metadata_groups=len({r['source_group'] for r in eligible}),
            source_annotation_sha256=sha(OLD / 'annotation.txt'), source_zip_members_sha256=sha(OLD / 'zip_members.json'),
            previous_selection_sha256=sha(OLD / 'selection.json'), public_identity_audit_sha256=sha(public_audit_path),
            selected=selected, expected_compressed_bytes=sum(r['member']['compressed_size'] for r in selected),
            expected_video_bytes=sum(r['member']['size'] for r in selected),
            selection_inputs='IDs, ego metadata, source groups, official archive member index only; no model predictions or image review',
            fallback='No replacement based on imagery, eligibility or model output. Ineligible/unknown cases retain explicit status.',
            provider_labels='Stored separately from future AI reference review; binary provider labels do not certify Stage2 entry/contact.')
for name in ('annotation.txt', 'zip_members.json', 'zip_range_access.json', 'repo_readme.txt'):
    shutil.copyfile(OLD / name, HERE / name)
(HERE / 'selection.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: plan[k] for k in ('status', 'eligible_metadata_clips', 'eligible_metadata_groups', 'expected_compressed_bytes', 'expected_video_bytes')}))
print(json.dumps([{'ID': r['ID'], 'source_group': r['source_group'], 'compressed_bytes': r['member']['compressed_size']} for r in selected]))
