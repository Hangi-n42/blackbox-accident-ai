"""Compare complete recorded Stage scores; never infer an official Private score."""
from decimal import Decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {'stage1': Decimal('0.20'), 'stage2': Decimal('0.40'), 'stage3': Decimal('0.40')}


def main():
    directory = ROOT / 'artifacts/submissions'
    complete, pending = [], []
    seen = set()
    for path in sorted(directory.glob('submission_status*.json')):
        record = json.loads(path.read_text(encoding='utf-8-sig'))
        identifier = record.get('submission_id')
        if identifier is None:
            continue
        if identifier in seen:
            raise ValueError(f'Duplicate submission record: {identifier}')
        seen.add(identifier)
        row = {'submission_id': identifier, 'zip': record['zip'],
               'record_source': str(path.relative_to(ROOT)),
               'last_checked_at_kst': record.get('checked_at_kst'),
               'server_state': record.get('server_state')}
        scores = record.get('official_scores') or {}
        if any(scores.get(stage) is None for stage in WEIGHTS):
            pending.append(row)
            continue
        values = {stage: Decimal(str(scores[stage])) for stage in WEIGHTS}
        if not all(value.is_finite() and 0 <= value <= 1 for value in values.values()):
            raise ValueError(f'Invalid recorded score: {identifier}')
        row['official_stage_scores'] = scores
        row['calculated_weighted_score'] = str(sum(values[stage] * weight for stage, weight in WEIGHTS.items()))
        row['official_private_score'] = record.get('private_score')
        complete.append(row)
    complete.sort(key=lambda row: Decimal(row['calculated_weighted_score']), reverse=True)
    result = {
        'scope': 'Comparison of last-observed Stage scores in local records. Not a live server query, official Private result, or final ranking.',
        'weights': {stage: str(weight) for stage, weight in WEIGHTS.items()},
        'selection_unit': 'One complete submission; never combine best Stage scores from different submissions.',
        'best_complete_recorded_submission': complete[0]['submission_id'] if complete else None,
        'complete': complete, 'pending_or_unscored': pending,
    }
    target = directory / 'recorded_score_comparison.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
