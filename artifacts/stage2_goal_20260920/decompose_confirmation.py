"""Separate observed candidate/selection failures from unidentifiable counterpart causes."""
import json
from pathlib import Path
from evaluate_confirmation import grade

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


if __name__ == '__main__':
    evaluation = read(HERE / 'evaluation.json')
    inputs = {c['ID']: c for c in read(HERE / 'ccd_intake/inputs.json')}
    annotations = {c['ID']: c for c in read(HERE / 'ccd_adjudication/records.json')['records']}
    contact, entry = [], []
    for row in evaluation['rows']:
        sid = row['ID']
        times = {img['frame']: img['pts_seconds'] for img in inputs[sid]['images']}
        label = annotations[sid]
        c = row['contact_evaluation']
        audit = dict(ID=sid, reference_contact_eligible=c is not None,
                     candidate_omission=None, counterpart_misidentification='not_identifiable_from_number_only_VLM_responses',
                     source_contact_evidence=label['collision']['status'], model_rendered_contact_evidence_sufficiency='not_separately_adjudicated',
                     selection_error_with_supported_candidate=None,
                     rationale='These are separate axes, not exclusive causal counts. Unknown label does not prove model evidence absence.')
        if c:
            lo, hi = c['interval_seconds']
            internal = grade(times[row['internal_vlm_collision_frame']], lo, hi)
            audit.update(candidate_omission=not c['any_presented_possibly_correct'],
                         candidate_contains_definite_correct=c['any_presented_definitely_correct'],
                         internal_vlm_result=internal['result'], final_baseline_result=c['baseline']['result'],
                         final_candidate_result=c['candidate']['result'],
                         selection_error_with_supported_candidate=c['any_presented_definitely_correct'] and internal['result'] == 'wrong')
        contact.append(audit)
        e = row['entry_evaluation']
        if e:
            lo, hi = e['interval_seconds']
            all_frames = list(times)
            sets = [all_frames, [f for f in all_frames if f <= row['internal_vlm_collision_frame']], row['entry_candidates'], [row['baseline']['entry_frame']]]
            stages = []
            for name, frames in zip(['original', 'before_internal_contact', 'presented_12', 'selected'], sets):
                choices = [grade(times[f], lo, hi)['result'] for f in frames]
                stages.append(dict(stage=name, definite='correct' in choices, possible=any(x != 'wrong' for x in choices)))
            first_loss = next((stages[i]['stage'] for i in range(1, 4) if stages[i-1]['definite'] and not stages[i]['definite']), None)
            entry.append(dict(ID=sid, stages=stages, first_definite_candidate_loss=first_loss,
                              selected_result=e['baseline']['result'], label_interval_seconds=[lo, hi],
                              same_counterparty_reference=label['counterparty'],
                              actual_model_counterparty_alignment='not_identifiable_from_frame_number_alone'))
    result = dict(status='complete', contact_axes=contact, entry_funnel=entry,
                  source_contact_unscorable_ids=[c['ID'] for c in contact if not c['reference_contact_eligible']],
                  candidate_omission_ids=[c['ID'] for c in contact if c['candidate_omission'] is True],
                  internal_selection_error_with_supported_candidate_ids=[c['ID'] for c in contact if c['selection_error_with_supported_candidate'] is True],
                  entry_definite_coverage_counts={stage: sum(next(x for x in e['stages'] if x['stage'] == stage)['definite'] for e in entry)
                      for stage in ['original', 'before_internal_contact', 'presented_12', 'selected']},
                  output_context_mismatch_ids=[r['ID'] for r in evaluation['rows'] if r['internal_vlm_collision_frame'] != r['candidate']['collision_frame']],
                  entry_after_final_contact_ids=[r['ID'] for r in evaluation['rows'] if r['entry_after_final_collision']])
    (HERE / 'decomposition.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['contact_axes', 'entry_funnel']}, ensure_ascii=False, indent=2))
