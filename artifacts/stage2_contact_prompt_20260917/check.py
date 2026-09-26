"""Pre-inference gate: can the proposed prompt-only change affect final contact?"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/pipeline_diagnosis_20260917'
CODE = ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'
sys.path.insert(0, str(CODE))
from solution import stage2_uncapped_jerk_v6c as policy
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

old_freeze = json.loads((OLD / 'freeze.json').read_text())
sources = {p: h for p, h in old_freeze['files'].items()
           if p.startswith('scripts/mac/') or '/stage2/code/solution/' in p}
assert all(sha(ROOT / p) == h for p, h in sources.items()), 'Baseline source drift'
prompt = ('These dashcam frames are chronological. Identify the same vehicle that '
          'physically contacts the camera vehicle. Choose the first numbered frame '
          'with evidence of their physical contact, not an earlier near approach, '
          'braking or turn, and not merely the largest or later camera shake. '
          'Use only the supplied allowed frames. Return JSON with collision_frame.')
save('freeze.json', dict(configuration=old_freeze['stage2_configuration'],
     candidate_prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
     changed_component='refined contact question only; proposal/motion/other outputs fixed',
     gates=['Final contact must depend on changed question before GPU work',
            'Public contact >=4/5 and MAE <=0.54s; other outputs unchanged',
            'If structural gate fails, reject without VLM inference or prompt tuning'],
     source_hashes=sources, baseline_freeze_sha256=sha(OLD/'freeze.json'),
     check_script_sha256=sha(Path(__file__)),
     inference_planned_only_after_structural_gate=True))

manifest = json.loads((OLD/'targeted_data/public_contact_manifest.json').read_text())
evaluation = {r['id']: r for r in json.loads((OLD/'evaluation_manifest.json').read_text()) if r['stage'] == 2}
rows = []
checks = 0
for i, case in enumerate(manifest):
    folder = OLD/'stage2_public_metal/stage2/public'/case['id']
    result = json.loads((folder/'result.json').read_text())
    job = json.loads((OLD/f'stage2_public_metal/job_{i:03d}.json').read_text())
    paths = [Path(p) for p in job['paths']]
    assert job['ID'] == case['id']
    assert paths == [ROOT / p['path'] for p in case['input_images']]
    assert all(p.is_file() for p in paths)
    # Reuse prior image integrity and inference; do not rehash/decode the dataset.
    for name, digest in result['source_sha256'].items():
        assert sha(ROOT/'scripts/mac'/name) == digest
    scores = np.load(folder/'motion.npz')
    base, new = scores['base_scores'], scores['new_scores']
    diagnostic = result['diagnostics']
    assert policy._score_hash(base) == diagnostic['uncapped_jerk']['base_score_sha256']
    assert policy._score_hash(new) == diagnostic['uncapped_jerk']['new_score_sha256']
    numbers = [policy.primitives._frame_number(p) for p in paths]
    candidates = diagnostic['collision_candidates']
    finals = []
    for selected in candidates:
        # Both production replacements are reproduced, then the actual final helper is called.
        selected_prediction = dict(result['prediction'], collision_frame=selected)
        motion_prediction = dict(selected_prediction, collision_frame=numbers[int(np.argmax(base))])
        final, _ = policy.apply_collision(paths, motion_prediction, diagnostic, new, base)
        repeated, _ = policy.apply_collision(paths, motion_prediction, diagnostic, new, base)
        assert final == repeated == result['prediction']
        assert all(final[k] == selected_prediction[k] for k in ('entry_frame','entry_side','evasion_space'))
        finals.append(final['collision_frame'])
        checks += 1
    assert len(set(finals)) == 1
    # Time mapping comes from the fixed public evaluation manifest, not an assumed FPS.
    assert evaluation[case['id']]['time_mapping'] == 'public 10Hz frame / 10 seconds'
    error = abs(result['prediction']['collision_frame']/10 - case['truth_seconds'])
    rows.append(dict(id=case['id'], truth_frame=case['truth_frame'], truth_seconds=case['truth_seconds'],
          baseline_prediction=result['prediction'], refined_frame=diagnostic['collision_replacement']['base_collision_frame'],
          tested_refined_candidates=candidates, resulting_final_contact_frames=sorted(set(finals)),
          other_fields_unchanged=True, cached_result_sha256=sha(folder/'result.json'),
          contact_error_seconds=error, correct_within_0_3s=error <= .3 + 1e-9,
          cached_motion_sha256=sha(folder/'motion.npz'),
          input_order_matches_frozen_manifest=True))

save('structural_results.json', dict(status='prompt_only_candidate_rejected_before_inference',
     cases=rows, case_count=len(rows), replacement_trials=checks,
     deterministic_helper_repeat_pass=True, final_predictions_identical=True,
     other_fields_identical=True, fresh_vlm_calls=0,
     full_pipeline_other_field_caution='Changing refined contact normally changes entry and space images; freezing those outputs is an isolated diagnostic, not an end-to-end policy change.',
     fresh_gpu_repeatability_test='not run: structural no-effect gate failed'))
metrics = dict(n=len(rows), correct=sum(r['correct_within_0_3s'] for r in rows),
               mae_seconds=sum(r['contact_error_seconds'] for r in rows)/len(rows))
assert metrics['correct'] == 4 and abs(metrics['mae_seconds']-.54) < 1e-9
save('metrics.json', dict(cached_baseline_recomputed=metrics,
     prompt_only_counterfactual_final=metrics,
     new_vlm_predictions=False, official_aggregate_score=False))
print(json.dumps(dict(cases=len(rows), trials=checks, unchanged=True, fresh_vlm_calls=0)))
