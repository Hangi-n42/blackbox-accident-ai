"""Read-only verification of completed Mac outputs; stored-answer replay, no model calls."""
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'mac_run'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ahash(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def same(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (path, set(actual), set(expected))
        for key in expected:
            same(actual[key], expected[key], path + '.' + str(key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, path + '.' + str(i))
    elif type(expected) is float:
        assert math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def rational_grade(p, low, high):
    assert low <= high
    distances = [abs(p-low), abs(p-high)]
    minimum = Fraction(0) if low <= p <= high else min(distances)
    maximum = max(distances)
    tag = 'correct' if maximum <= Fraction(3, 10) else 'wrong' if minimum > Fraction(3, 10) else 'indeterminate'
    return dict(result=tag, error_lower_seconds=float(minimum), error_upper_seconds=float(maximum))


def rational_paired_delta(baseline, candidate, low, high):
    # Closed acceptance intervals and exact rational set differences; no sampled-time search.
    def accepted(p):
        a, b = max(low, p-Fraction(3, 10)), min(high, p+Fraction(3, 10))
        return None if a > b else (a, b)
    def outside(a, b):
        return a is not None and (b is None or a[0] < b[0] or a[1] > b[1])
    b, c = accepted(baseline), accepted(candidate)
    values = []
    if outside(c, b): values.append(1)
    if outside(b, c): values.append(-1)
    both = b is not None and c is not None and max(b[0], c[0]) <= min(b[1], c[1])
    intervals = sorted(x for x in (b, c) if x is not None)
    neither = not intervals or intervals[0][0] > low or intervals[-1][1] < high or (len(intervals) == 2 and intervals[0][1] < intervals[1][0])
    if both or neither: values.append(0)
    assert values
    return [min(values), max(values)]


def independent_aggregate(scores):
    n = len(scores)
    tags = [s['result'] for s in scores]
    counts = {k: tags.count(k) for k in ('correct', 'wrong', 'indeterminate')}
    return dict(n=n, **counts,
                accuracy_bounds=[counts['correct']/n, (n-counts['wrong'])/n] if n else None,
                mae_bounds_seconds=[sum(x[k] for x in scores)/n for k in ('error_lower_seconds','error_upper_seconds')] if n else None)


def verify_scores(report, cases):
    import importlib.util
    spec = importlib.util.spec_from_file_location('frozen_evaluator', HERE/'evaluate_confirmation.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)  # Definitions and boundary assertions only, never __main__.
    # Exhaustively check every possible point and bracket on this cohort's 0.1-second grid.
    grid = [Fraction(i,10) for i in range(50)]
    checked = 0
    for lo_index, lo in enumerate(grid):
        for hi in grid[lo_index:]:
            for p in grid:
                same(evaluator.grade(float(p),float(lo),float(hi)), rational_grade(p,lo,hi))
                checked += 1
    paired_grid_checks = 0
    for lo, hi in [(Fraction(0),Fraction(49,10)),(Fraction(7,10),Fraction(7,10)),
                   (Fraction(3,10),Fraction(9,10)),(Fraction(29,10),Fraction(32,10)),
                   (Fraction(0),Fraction(1,10)),(Fraction(1),Fraction(19,10))]:
        for baseline in grid:
            for candidate in grid:
                same(evaluator.paired_accuracy_delta(float(baseline),float(candidate),float(lo),float(hi)),
                     rational_paired_delta(baseline,candidate,lo,hi))
                paired_grid_checks += 1
    annotations = read(HERE/'ccd_adjudication/records.json')
    assert annotations['model_predictions_seen'] is False
    labels = {x['ID']:x for x in annotations['records']}
    sources = {x['ID']:x for x in cases}
    provider = {x['ID']:x for x in read(HERE/'ccd_intake/selection.json')['selected']}
    evaluation = read(HERE/'evaluation.json')
    assert evaluation['status'] == 'complete' and evaluation['official_S2'] is None
    assert evaluation['source_cases'] == len(cases) and evaluation['model_calls'] == len(cases)*4
    assert [x['ID'] for x in evaluation['rows']] == [x['ID'] for x in cases]
    output_rows = {x['ID']:x for x in evaluation['rows']}
    recomputed = {field:[] for field in ('contact_evaluation','entry_evaluation')}
    category_pairs = {key:{p:[] for p in ('baseline','candidate')} for key in ('entry_side','evasion_space')}
    provider_scores = {p:[] for p in ('baseline','candidate')}
    paired_checks = 0
    for record in report['videos']:
        sid = record['ID']; truth = labels[sid]; row = output_rows[sid]
        times = {x['frame']:Fraction(str(x['pts_seconds'])) for x in sources[sid]['images']}
        pred = {p:record[p+'_prediction'] for p in ('baseline','candidate')}
        for p in pred: same(row[p], pred[p])
        same(row['original_times'], {p:{k:float(times[pred[p][k]]) for k in ('collision_frame','entry_frame')} for p in pred})
        assert row['entry_after_final_collision'] == (times[pred['candidate']['entry_frame']] > times[pred['candidate']['collision_frame']])
        for field, name, key in [('collision','contact_evaluation','collision_frame'),('entry','entry_evaluation','entry_frame')]:
            t = truth[field]
            if not t.get('evaluation_eligible',False):
                assert row[name] is None
                continue
            lo, hi = times[t['lower_frame']], times[t['upper_frame']]
            assert lo <= hi and t['status'] != 'unknown'
            actual = row[name]
            scores = {p:rational_grade(times[pred[p][key]],lo,hi) for p in pred}
            for p in pred: same(actual[p],scores[p])
            same(actual['interval_seconds'],[float(lo),float(hi)])
            assert actual['label_kind'] == t['status'] and actual['evidence_level'] == t.get('evidence_level')
            delta = rational_paired_delta(times[pred['baseline'][key]],times[pred['candidate'][key]],lo,hi)
            same(actual['paired_accuracy_delta_bounds'],delta)
            same(evaluator.paired_accuracy_delta(float(times[pred['baseline'][key]]),float(times[pred['candidate'][key]]),float(lo),float(hi)),delta)
            paired_checks += 1
            choices = record['diagnostics']['collision_candidates' if field == 'collision' else 'entry_candidates']
            assert row['collision_candidates' if field == 'collision' else 'entry_candidates'] == choices
            choices_grades = [rational_grade(times[f],lo,hi)['result'] for f in choices]
            assert actual['any_presented_definitely_correct'] == ('correct' in choices_grades)
            assert actual['any_presented_possibly_correct'] == any(g!='wrong' for g in choices_grades)
            if field == 'entry':
                assert actual['any_original_definitely_correct'] == any(rational_grade(t,lo,hi)['result']=='correct' for t in times.values())
                cap = record['diagnostics']['collision']['collision_frame']
                assert row['internal_vlm_collision_frame'] == cap
                assert actual['any_before_internal_contact_definitely_correct'] == any(rational_grade(t,lo,hi)['result']=='correct' for f,t in times.items() if f<=cap)
            gain = scores['baseline']['result']=='wrong' and scores['candidate']['result']=='correct'
            loss = scores['baseline']['result']=='correct' and scores['candidate']['result']=='wrong'
            assert actual['definite_gain'] == gain and actual['definite_loss'] == loss
            recomputed[name].append(dict(ID=sid, **scores, delta=delta, gain=gain, loss=loss))
        for key, classes in [('entry_side',['LEFT','RIGHT']),('evasion_space',[0,1])]:
            label = truth[key].get('value') if truth[key].get('evaluation_eligible',False) else None
            assert row['category_labels'][key] == label
            if label is not None:
                assert label in classes
                for p in pred:
                    assert pred[p][key] in classes
                    category_pairs[key][p].append((label,pred[p][key]))
        ref_frame = provider[sid]['labels'].index(1)
        ref = row['provider_reference_only']
        assert ref['not_DACON_ground_truth'] is True and ref['first_positive_frame'] == ref_frame
        assert ref['time_seconds'] == float(times[ref_frame])
        for p in pred:
            score = rational_grade(times[pred[p]['collision_frame']],times[ref_frame],times[ref_frame])
            same(ref[p],score); provider_scores[p].append(score)
    fields = {}
    for name, rows in recomputed.items():
        fields[name] = dict(eligible_ids=[r['ID'] for r in rows],
                           baseline=independent_aggregate([r['baseline'] for r in rows]),
                           candidate=independent_aggregate([r['candidate'] for r in rows]),
                           definite_gain_ids=[r['ID'] for r in rows if r['gain']],
                           definite_loss_ids=[r['ID'] for r in rows if r['loss']],
                           transitions={a+'->'+b:sum(r['baseline']['result']==a and r['candidate']['result']==b for r in rows) for a in ('correct','wrong','indeterminate') for b in ('correct','wrong','indeterminate')},
                           paired_accuracy_delta_bounds=[sum(r['delta'][k] for r in rows)/len(rows) for k in (0,1)] if rows else None)
    same(evaluation['evaluation'],fields)
    for key, by_policy in category_pairs.items():
        classes = ['LEFT','RIGHT'] if key=='entry_side' else [0,1]
        for policy, pairs in by_policy.items():
            matrix = [[0,0],[0,0]]
            for truth,pred in pairs: matrix[classes.index(truth)][classes.index(pred)] += 1
            per_class = []
            for value in classes:
                tp = sum(t==value and p==value for t,p in pairs)
                fp = sum(t!=value and p==value for t,p in pairs)
                fn = sum(t==value and p!=value for t,p in pairs)
                per_class.append(float(Fraction(2*tp,2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
            same(evaluation['categorical_evaluation'][key][policy],dict(n=len(pairs),classes=classes,confusion=matrix,macro_f1=sum(per_class)/2 if pairs else None,zero_division=0))
    for p in provider_scores: same(evaluation['provider_reference_metrics'][p],independent_aggregate(provider_scores[p]))
    gate = bool(fields['contact_evaluation']['definite_gain_ids']) and not fields['contact_evaluation']['definite_loss_ids']
    assert evaluation['contact_confirmation_gate_pass'] == gate
    assert evaluation['other_three_unchanged'] == len(cases)
    assert evaluation['collision_changed_ids'] == [x['ID'] for x in report['videos'] if x['baseline_prediction']['collision_frame']!=x['candidate_prediction']['collision_frame']]
    return dict(status='PASS', rational_grade_grid_tests=checked, rational_paired_delta_grid_tests=paired_grid_checks,
                paired_interval_set_difference_checks=paired_checks,
                fixed_cohort_aggregation_exact=True, category_confusion_and_macro_f1_exact=True, provider_reference_separate=True,
                official_S2_null=True, evaluation=fields, categorical_evaluation=evaluation['categorical_evaluation'],
                original_gate_pass=gate, grade_scope='Exact rational 0.1-second grid; tolerance-only subnanosecond cases outside this cohort are not certified',
                evaluation_sha256=sha(HERE/'evaluation.json'))


def main():
    targets = [HERE / ('runtime_score_review' + suffix) for suffix in ('.json', '.md')]
    assert not any(path.exists() for path in targets), 'Never overwrite a prior review'
    report = read(OUT / 'report.json')
    assert report['status'] == 'complete', 'Wait until all original model workers complete'
    locked = read(HERE / 'freeze.json')
    protocol = read(HERE / 'protocol.json')
    all_cases = read(HERE / 'ccd_intake/inputs.json')
    excluded = set(locked['excluded_duplicate_ids'])
    assert excluded.issubset({c['ID'] for c in all_cases})
    cases = [c for c in all_cases if c['ID'] not in excluded]
    assert locked['status'] == 'locked_before_predictions'
    assert report['freeze_sha256'] == sha(HERE / 'freeze.json')
    assert report['worker_source_sha256'] == sha(ROOT / 'artifacts/stage2_goal_20260919/expert_mac_review/run_paired_mac.py')
    assert datetime.fromisoformat(locked['created_utc']) <= datetime.fromisoformat(report['started_utc'])
    assert datetime.fromisoformat(report['started_utc']) <= datetime.fromisoformat(report['ended_utc'])
    assert len(cases) == len(report['workers']) == len(report['videos']) == locked['run_cases']
    expected = [case['ID'] for case in cases]
    assert expected == [i for i in protocol['case_ids'] if i not in excluded]
    assert expected == [v['ID'] for v in report['videos']] == [w['ID'] for w in report['workers']]
    assert len(set(expected)) == len(cases)
    assert report['configuration'] == dict(decode_mode='sync', compute_dtype='native', deepstack_fix=True,
                                         policy='fixed_auxiliary_sum_removal')
    assert 'arm64' in report['platform']
    for name, digest in locked['files'].items():
        assert sha(ROOT / name) == digest, name

    sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
    sys.path.insert(0, str(HERE.parent / 'stage2_goal_20260919/candidate_runtime'))
    import candidate
    import numpy as np
    from PIL import Image
    assert np.__version__ == report['packages']['numpy'], 'Use the original NumPy version'

    # Replay the same four questions using stored text. Reconstruct and verify the
    # exact bounded image pixels; do not instantiate a model or MLX processor.
    class StoredAnswers:
        def __init__(self, calls):
            self.calls, self.position = calls, 0

        def ask(self, images, prompt, max_new_tokens=128):
            call = self.calls[self.position]
            assert prompt == call['prompt'] and max_new_tokens == call['max_new_tokens']
            budget = max(1024, 1_200_000 // len(images))
            sizes, digests = [], []
            for image in images:
                image = image.convert('RGB')
                scale = min(1., math.sqrt(budget / (image.width * image.height)))
                size = (max(32, int(image.width * scale) // 32 * 32),
                        max(32, int(image.height * scale) // 32 * 32))
                bounded = image.resize(size, Image.Resampling.BICUBIC)
                sizes.append(list(size))
                digests.append(hashlib.sha256(bounded.tobytes()).hexdigest())
            assert sizes == call['image_sizes'] and digests == call['image_sha256']
            assert call['decode_mode'] == 'sync' and call['compute_dtype'] == 'native' and call['deepstack_fix'] is True
            assert call['token_trace'] and call['processor_input_sha256']
            assert call['seconds'] >= 0 and call['peak_memory'] > 0
            self.position += 1
            return call['text'].strip()

    verified_images = 0
    rows = []
    artifact_hashes = {}
    for case, worker, top in zip(cases, report['workers'], report['videos'], strict=True):
        identity = case['ID']
        folder = OUT / identity
        record, wr = read(folder / 'result.json'), read(folder / 'worker_report.json')
        calls, job = read(folder / 'calls.json'), read(OUT / (identity + '.job.json'))
        assert record == top and calls == record['calls']
        assert worker['exit_status'] == 0 and wr['status'] == 'complete'
        assert wr['model_calls'] == len(calls) == 4 and wr['network_attempts'] == 0
        assert wr['ID'] == identity and worker['wall_seconds'] >= wr['wall_seconds'] > 0
        assert datetime.fromisoformat(locked['created_utc']) < datetime.fromisoformat(wr['started_utc'])
        assert datetime.fromisoformat(wr['started_utc']) <= datetime.fromisoformat(wr['ended_utc'])
        assert set(job) == {'ID', 'paths', 'image_sha256'} and job['ID'] == identity
        paths = [ROOT / item['path'] for item in case['images']]
        assert job['paths'] == [str(path.resolve()) for path in paths]
        assert job['image_sha256'] == [item['sha256'] for item in case['images']]
        for path, expected_hash in zip(paths, job['image_sha256'], strict=True):
            assert sha(path) == expected_hash, path
            verified_images += 1
        assert sha(ROOT / case['source_video']) == case['source_sha256']
        assert sha(ROOT / case['pts_source']) == case['pts_sha256']
        pts = read(ROOT / case['pts_source'])
        assert pts['source_sha256'] == case['source_sha256']
        numbers, times = [], {}
        previous = None
        for item, native in zip(case['images'], pts['mapping'], strict=True):
            seconds = Fraction(native['native_pts']) * Fraction(native['time_base'])
            assert item['frame'] == native['frame_id'] and item['pts_seconds'] == native['time_s'] == float(seconds)
            assert previous is None or previous < seconds
            previous = seconds
            numbers.append(item['frame']); times[item['frame']] = float(seconds)
        assert numbers == sorted(set(numbers)) and record['frames'] == len(numbers)
        assert numbers == [candidate.reference.primitives._frame_number(path) for path in paths]
        with np.load(folder / 'motion.npz', allow_pickle=False) as cache:
            features, base, new, proposed = (cache[key] for key in ('features', 'base_scores', 'new_scores', 'candidate_scores'))
            reproduced_base, reproduced_new = candidate.reference._scores_from_features(features)
            assert reproduced_base.dtype == base.dtype and reproduced_base.shape == base.shape
            assert reproduced_new.dtype == new.dtype and reproduced_new.shape == new.shape
            assert reproduced_base.tobytes() == base.tobytes() and reproduced_new.tobytes() == new.tobytes()
            jerk = features[:, 0]
            median = np.median(jerk)
            scale = np.median(np.abs(jerk - median)) * 1.4826
            direct = np.maximum((jerk - median) / max(float(scale), 1e-3), 0)
            direct[0] = 0
            assert direct.dtype == proposed.dtype and direct.tobytes() == proposed.tobytes()
            assert len(features) == len(numbers)
            assert all(np.isfinite(array).all() for array in (features, base, new, proposed))
            baseline, prediction = record['baseline_prediction'], record['candidate_prediction']
            assert baseline['collision_frame'] == numbers[int(np.argmax(new))]
            assert prediction['collision_frame'] == numbers[int(np.argmax(proposed))]
            assert record['diagnostics']['uncapped_jerk']['new_score_sha256'] == ahash(new)
            assert record['diagnostics']['contact_ablation']['new_score_sha256'] == ahash(proposed)
            assert record['diagnostics']['uncapped_jerk']['base_score_sha256'] == ahash(base)
            assert all(baseline[key] == prediction[key] for key in ('entry_frame', 'entry_side', 'evasion_space'))
            replay = StoredAnswers(calls)
            replay_prediction, replay_diagnostics = candidate.predict_file(paths, base, new, features, replay)
            assert replay.position == 4 and replay_prediction == prediction and replay_diagnostics == record['diagnostics']
        for label, values in [('baseline', baseline), ('candidate', prediction)]:
            with (folder / (label + '.csv')).open(newline='') as stream:
                reader = csv.DictReader(stream)
                assert reader.fieldnames == candidate.COLUMNS
                csv_rows = list(reader)
            assert len(csv_rows) == 1 and csv_rows[0]['ID'] == identity
            for key, value in values.items():
                assert csv_rows[0][key] == str(value)
            assert set(values) == {'collision_frame', 'entry_frame', 'entry_side', 'evasion_space'}
            assert all(type(values[key]) is int and values[key] in times for key in ('collision_frame', 'entry_frame'))
            assert type(values['evasion_space']) is int and values['evasion_space'] in (0, 1)
            assert values['entry_side'] in ('LEFT', 'RIGHT')
        assert record['max_peak_memory_GB'] == max(call['peak_memory'] for call in calls)
        rows.append(dict(ID=identity, frames=len(numbers), worker_exit_status=worker['exit_status'],
            model_calls=4, stored_answer_replays=4, python_network_attempts=wr['network_attempts'],
            baseline_prediction=baseline, candidate_prediction=prediction,
            baseline_collision_seconds=times[baseline['collision_frame']],
            candidate_collision_seconds=times[prediction['collision_frame']],
            other_three_outputs_unchanged=True, all_three_score_arrays_exact=True,
            calls_prompts_and_rendered_image_hashes_exact=True, native_pts_mapping_exact=True,
            parent_worker_wall_seconds=worker['wall_seconds'], worker_wall_seconds=wr['wall_seconds'],
            motion_seconds=record['motion_seconds'], model_load_seconds=record['model_load_seconds'],
            generation_seconds=sum(call['seconds'] for call in calls),
            peak_mlx_memory_GB=record['max_peak_memory_GB'], peak_process_rss_bytes=wr['max_process_rss_bytes']))
        for path in [folder / name for name in ('result.json', 'worker_report.json', 'calls.json', 'motion.npz', 'baseline.csv', 'candidate.csv')]:
            artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
    assert report == read(OUT / 'report.json'), 'Completed report changed during review'
    for path in (OUT / 'report.json', HERE / 'freeze.json', Path(__file__)):
        artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(),
        review_type='Independent runtime/code/output verification; not new model inference or label adjudication',
        new_model_calls_by_reviewer=0, stored_model_calls_verified=sum(r['model_calls'] for r in rows),
        stored_answer_replays=4*len(rows), workers_verified=len(rows), input_images_sha_verified=verified_images,
        review_lock_bound_files_sha_verified=len(locked['files']), score_arrays_exact=3*len(rows),
        review_precedes_run=True, models_and_frozen_files_unchanged=True,
        model_processor_tensor_hashes='Recorded hashes inspected for presence; tensors not regenerated',
        motion_scope='All three score arrays recomputed from saved motion features; optical flow itself not recomputed',
        native_pts_scope='All frozen PNG indices mapped to existing native PTS; source video not decoded again',
        network_scope='Python socket connect/connect_ex/create_connection intercepted; not full native OS network tracing',
        timing_scope='Parent worker wall includes process launch and input hashes; worker wall includes imports/load/motion/shared four generations/output. Call seconds exclude processor/image preparation.',
        memory_scope='MLX peak memory is cumulative within each fresh worker; RSS is macOS process high-water mark. Sequential worker peaks are not summed.',
        limitations=['Shared four calls support exact final-rule comparison, not two independently timed complete pipelines.',
                    'AI review labels remain AI observations; no official/human GT certification.',
                    'Mac MLX success does not establish CUDA/NF4 numerical equivalence.',
                    'No model quality or score-improvement claim is made by this runtime review.'],
        score_verification=verify_scores(report, cases),
        report_environment={k:report[k] for k in ('platform','python','chip','packages')}, artifact_sha256=artifact_hashes, videos=rows)
    with targets[0].open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ['# Mac 실행 독립 검증', '', '**판정: PASS.** 완료된 Mac 실행의 코드·입출력·후보 대조 계약을 확인했다. 검토자는 새 VLM/GPU 추론을 실행하지 않았다.', '',
        f"- 실제 worker 12개 모두 exit 0, 저장된 모델 호출 48회, 입력 PNG {verified_images:,}개 해시 및 동결 파일 {len(locked['files']):,}개 해시 확인.",
        '- 저장된 motion features로 재계산한 원본 base/new와 jerk-only 후보 점수 36개 배열이 캐시와 정확히 일치했다. 원본 번호 argmax와 다른 3개 출력 보존은 12/12 확인했다.',
        '- 실제 48개 답변을 반환하는 CPU 재생으로 질문·토큰 예산·시트 크기·리사이즈 후 RGB 해시·전체 diagnostics를 정확 재현했다. 모델이나 processor tensor는 재실행하지 않았다.',
        '- 모든 프레임의 원본 번호·native PTS 대응을 확인했다. 모델 worker에는 ID·이미지 경로·해시만 전달되며 정답·PTS는 전달되지 않는다.',
        '- 검수 lock 생성 시각이 각 worker 및 상위 실행 시작보다 앞섰고, 동결 검수·코드·모델 파일 해시는 유지됐다.', '',
        '| ID | 프레임 | 기준 접촉 | 후보 접촉 | worker 시간 | MLX peak |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['ID']} | {row['frames']} | {row['baseline_prediction']['collision_frame']} | {row['candidate_prediction']['collision_frame']} | {row['parent_worker_wall_seconds']:.3f}초 | {row['peak_mlx_memory_GB']:.3f}GB |")
    lines += ['', '## 해석 범위', '',
        '- 네 질의를 한 번 실행하여 기준·후보가 공유하고 최종 접촉 수식만 바꾼 대조다. 두 전체 모델을 각각 실행한 속도 비교가 아니다.',
        '- 상위 worker 시간에는 프로세스 시작·입력 해시가 포함된다. 내부 worker 시간에는 import·모델 load·motion·공유질의·저장이 포함된다. call seconds는 processor/이미지 준비 이후 생성 구간이다.',
        '- Python socket 접속 시도는 0회다. 네이티브 라이브러리까지 포괄한 OS 네트워크 패킷 계측으로 표현하지 않는다.',
        '- motion score 계산은 저장된 features로 재검증했다. 광학흐름 추출은 다시 실행하지 않았다.',
        '- MLX peak는 worker 내 누적 최고점이고 RSS는 macOS 프로세스 최대치다. 순차 실행된 12개 worker의 메모리를 합산하지 않는다.',
        '- 영상→PNG 추출 및 RGB 무손실성은 기존 입력 준비 검사를 이용했다. 이번에는 PNG 전수 해시와 native PTS 표 대응을 검증했으며 원본 영상을 다시 디코딩하지 않았다.',
        '- AI 검수의 의미 정확도·공식 정답 여부는 이 실행 검증의 범위가 아니다. Mac 통과를 CUDA/NF4 수치 동등성으로 확대하지 않는다.',
        '- 새 후보 탐색·재훈련·추가 모델 호출·운영 파일 수정은 없다. 실행 통과만으로 정확도 개선이나 생산 채택을 선언하지 않는다.', '',
        '기계 판독 결과와 근거 해시는 runtime_score_review.json에 있다. 재검증 코드는 verify_runtime_score.py이며 기존 리뷰 출력은 덮어쓰지 않는다.', '']
    scoring = result['score_verification']
    lines += ['## 점수 독립 재계산', '',
              '- Fraction 유리수 계산으로 0.1초 격자의 grade 63,750건을 전수 대조했다. 같은 정답 시각을 공유하는 paired delta는 닫힌 허용구간 차집합으로 15,000건을 대조했다.',
              '- 실제 적격 contact/entry의 dmin/dmax, ±0.3초 포함 판정, 고정 분모의 정확도·MAE 경계, 9개 전이, paired delta를 독립 재계산했다.',
              '- 방향·공간의 실제 적격 범주를 확인하고 confusion matrix·범주별 TP/FP/FN으로 macro-F1을 재계산했다. 공급자 first-positive는 별도 참조로 유지되고 공식 S2는 null이다.', '',
              '| 항목 | 적격 수 | 기준 C/W/U | 후보 C/W/U | paired 정확도 차이 범위 |',
              '|---|---:|---|---|---|']
    for field, values in scoring['evaluation'].items():
        base, cand = values['baseline'], values['candidate']
        bs = '/'.join(str(base[k]) for k in ('correct','wrong','indeterminate'))
        cs = '/'.join(str(cand[k]) for k in ('correct','wrong','indeterminate'))
        lines.append(f"| {field} | {base['n']} | {bs} | {cs} | {values['paired_accuracy_delta_bounds']} |")
    lines += ['', '고정 gate의 통과 여부와 모든 구간 정답에서 순개선이 보장되는지는 별개다. paired 차이와 모든 전이를 함께 해석해야 한다. 이 검증은 AI 정답의 의미적 타당성 또는 대회 점수를 인증하지 않는다.', '']
    with targets[1].open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('status', 'workers_verified', 'stored_model_calls_verified', 'input_images_sha_verified', 'score_arrays_exact')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
