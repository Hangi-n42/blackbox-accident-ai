"""Read-only CPU verification of completed target-marker calls; never loads a model."""
import ast
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
OLD = ROOT / 'artifacts/stage2_goal_20260920'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rgb_sha(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def bounded(image):
    from PIL import Image
    image = image.convert('RGB')
    scale = min(1., math.sqrt(1_200_000 / (image.width*image.height)))
    size = tuple(max(32, int(d*scale)//32*32) for d in image.size)
    return image.resize(size, Image.Resampling.BICUBIC)


# Reuse only independent rational maths, never execute the prior verification main.
math_source = OLD / 'verify_runtime_score.py'
nodes = [n for n in ast.parse(math_source.read_text()).body if isinstance(n, ast.FunctionDef)
         and n.name in {'same','rational_grade','rational_paired_delta','independent_aggregate'}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(math_source), 'exec'))


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution import stage2_v2 as v2
    from solution import stage2_uncapped_jerk_v6c as reference
    targets = [HERE / ('independent_verification'+s) for s in ('.json','.md')]
    assert not any(p.exists() for p in targets)
    frozen = read(HERE/'freeze.json')
    report = read(HERE/'run/report.json')
    evaluation = read(HERE/'evaluation.json')
    protocol = read(HERE/'protocol.json')
    ids = protocol['cases']
    assert len(ids)==len(set(ids))==5
    assert frozen['status']=='locked_before_marker_predictions'
    assert report['status']=='complete' and evaluation['status']=='complete'
    assert report['freeze_sha256']==sha(HERE/'freeze.json')
    assert datetime.fromisoformat(frozen['created_utc']) < datetime.fromisoformat(report['started_utc']) < datetime.fromisoformat(report['ended_utc'])
    for name,digest in frozen['files'].items(): assert sha(ROOT/name)==digest, name
    assert [(w['ID'],w['arm']) for w in report['workers']]==[(sid,arm) for sid in ids for arm in ('baseline','marked')]
    assert all(w['exit_status']==0 for w in report['workers'])
    sources = {r['ID']:r for r in read(OLD/'ccd_intake/inputs.json')}
    old_runs = {r['ID']:r for r in read(OLD/'mac_run/report.json')['videos']}
    labels = {r['ID']:r for r in read(OLD/'ccd_adjudication/records.json')['records']}
    input_checks = {r['ID']:r for r in read(HERE/'input_checks.json')['cases']}
    annotation_tasks = {r['ID']:r for r in read(HERE/'annotation_tasks.json')}
    annotations = {}
    for who in ('a','b'):
        review = read(HERE/f'reviewer_{who}_marks.json')
        peer = read(HERE/f'peer_review_{who}.json')
        assert review['model_predictions_seen'] is False and peer['model_predictions_seen'] is False and peer['status']=='PASS'
        for case in review['cases']:
            assert case['ID'] not in annotations
            annotations[case['ID']] = case
    assert set(annotations)==set(ids)
    rows=[]; worker_rows=[]; calls_count=cached_replays=0; hashes={}
    for sid in ids:
        source, old, checks = sources[sid], old_runs[sid], input_checks[sid]
        paths=[ROOT/f['path'] for f in source['images']]
        numbers=[v2._frame_number(p) for p in paths]
        times={f['frame']:Fraction(str(f['pts_seconds'])) for f in source['images']}
        assert len(paths)==50 and numbers==list(range(50))
        pts=read(ROOT/source['pts_source'])
        assert sha(ROOT/source['source_video'])==source['source_sha256'] and sha(ROOT/source['pts_source'])==source['pts_sha256']
        for item,native,path in zip(source['images'],pts['mapping'],paths,strict=True):
            assert sha(path)==item['sha256']
            assert Fraction(native['native_pts'])*Fraction(native['time_base'])==times[item['frame']]
            assert native['frame_id']==item['frame'] and native['time_s']==item['pts_seconds']
        candidates=old['diagnostics']['entry_candidates']
        assert checks['candidates']==candidates and len(candidates)==12
        indices=[numbers.index(n) for n in candidates]
        baseline=v2._sheet(paths,indices,columns=4)
        base_path=HERE/'inputs'/f'{sid}_baseline.png'
        marker_path=HERE/'inputs'/f'{sid}_marked.png'
        saved_base=Image.open(base_path).convert('RGB')
        saved_mark=Image.open(marker_path).convert('RGB')
        assert baseline.size==saved_base.size==saved_mark.size
        assert baseline.tobytes()==saved_base.tobytes()
        expected=baseline.copy(); allowed=Image.new('1',baseline.size)
        draw,mask_draw=ImageDraw.Draw(expected),ImageDraw.Draw(allowed)
        items={x['frame']:x for x in annotations[sid]['frames']}
        assert len(items)==len(annotations[sid]['frames'])==12 and set(items)==set(candidates)
        reconstructed_boxes=[]
        for slot,frame in enumerate(candidates):
            item=items[frame]; box=item['box_xyxy']; path=paths[numbers.index(frame)]
            task_frame=next(f for f in annotation_tasks[sid]['frames'] if f['frame']==frame)
            assert Path(task_frame['path']).resolve()==path.resolve() and sha(path)==task_frame['sha256']
            if 'source_image' in item:
                assert (ROOT/item['source_image']).resolve()==path.resolve()
            with Image.open(path) as original:
                w,h=original.size; iw,ih=ImageOps.contain(original,(384,228)).size
            assert [w,h]==[item['width'],item['height']]
            if box is None:
                assert item['status'] in ('absent','occluded','uncertain')
                reconstructed_boxes.append(dict(frame=frame,box=None,status=item['status']))
                continue
            assert item['status']=='visible' and all(type(x) is int for x in box)
            x0,y0,x1,y1=box
            assert 0<=x0<x1<w and 0<=y0<y1<h
            left=(slot%4)*384+(384-iw)//2; top=(slot//4)*256+28
            target=[left+round(x0*iw/w),top+round(y0*ih/h),left+round(x1*iw/w),top+round(y1*ih/h)]
            assert target[2]-target[0]>=1 and target[3]-target[1]>=1
            draw.rectangle(target,outline=(255,255,0),width=2);mask_draw.rectangle(target,outline=1,width=2)
            reconstructed_boxes.append(dict(frame=frame,box=target,source_box=box,status='visible'))
        same(checks['boxes'],reconstructed_boxes)
        assert expected.tobytes()==saved_mark.tobytes()
        changed=np.any(np.asarray(saved_base)!=np.asarray(saved_mark),axis=2)
        allowed_mask=np.asarray(allowed,dtype=bool)
        assert changed.any() and not np.any(changed & ~allowed_mask)
        assert np.all(np.asarray(saved_mark)[changed]==(255,255,0))
        assert checks['changed_pixels']==int(changed.sum()) and checks['allowed_pixels']==int(allowed_mask.sum())
        same(checks['changed_fraction'],float(changed.mean()))
        assert checks['canvas']==list(baseline.size)
        bounded_arms={a:bounded(im) for a,im in [('baseline',saved_base),('marked',saved_mark)]}
        for arm,image in bounded_arms.items():
            saved=Image.open(HERE/'inputs'/f'{sid}_{arm}_model_input.png').convert('RGB')
            assert saved.size==image.size and saved.tobytes()==image.tobytes()
            assert checks[arm+'_bounded_rgb_sha256']==rgb_sha(image)
            assert checks['bounded_size']==list(image.size)
        assert rgb_sha(bounded_arms['baseline'])==old['calls'][2]['image_sha256'][0]
        assert rgb_sha(bounded_arms['baseline'])!=rgb_sha(bounded_arms['marked'])
        truth=labels[sid]['entry']; assert truth['evaluation_eligible'] is True
        lo,hi=times[truth['lower_frame']],times[truth['upper_frame']]
        row=dict(ID=sid,interval_seconds=[float(lo),float(hi)],already_inside=(lo==hi==0),arms={})
        arm_calls={}
        with np.load(OLD/'mac_run'/sid/'motion.npz',allow_pickle=False) as motion:
            for arm in ('baseline','marked'):
                folder=HERE/'run'/f'{sid}_{arm}'
                worker=read(folder/'worker_report.json');result=read(folder/'result.json');calls=read(folder/'calls.json')
                job=read(HERE/'inputs'/f'{sid}_{arm}.job.json')
                assert set(job)=={'ID','arm','sheet','sheet_sha256','paths','candidates','prompt','max_new_tokens'}
                assert job['ID']==worker['ID']==result['ID']==sid and job['arm']==worker['arm']==result['arm']==arm
                assert job['paths']==[str(p) for p in paths] and job['candidates']==candidates
                assert job['sheet']==str(HERE/'inputs'/f'{sid}_{arm}.png') and sha(Path(job['sheet']))==job['sheet_sha256']
                assert worker['status']=='complete' and worker['network_attempts']==0 and worker['model_calls']==len(calls)==1
                assert datetime.fromisoformat(frozen['created_utc']) < datetime.fromisoformat(worker['started_utc']) <= datetime.fromisoformat(worker['ended_utc'])
                calls_count+=1; call=calls[0];arm_calls[arm]=call
                assert call==result['call'] and call['text'].strip()==result['raw']
                assert call['prompt']==job['prompt']==old['calls'][2]['prompt']
                assert call['max_new_tokens']==job['max_new_tokens']==old['calls'][2]['max_new_tokens']==40
                assert call['image_sizes']==[list(bounded_arms[arm].size)] and call['image_sha256']==[rgb_sha(bounded_arms[arm])]
                assert call['deepstack_fix'] is True and call['compute_dtype']=='native' and call['decode_mode']=='sync'
                assert call['token_trace'] and call['processor_input_sha256'] and call['seconds']>=0 and call['peak_memory']>0
                parsed=v2._json_object(result['raw']);integer=v2._integer(parsed.get('entry_frame'))
                chosen=numbers[v2._choice(parsed,'entry_frame',paths,indices,0)]
                same(result['parsed'],parsed)
                assert result['raw_integer']==integer and result['valid_offered_integer']==(integer in candidates)
                assert result['entry_frame']==chosen and result['parser_changed']==(integer!=chosen)
                class Replay:
                    count=0
                    def ask(self,images,prompt,max_new_tokens):
                        i=self.count;self.count+=1;historical=old['calls'][i]
                        assert prompt==historical['prompt'] and max_new_tokens==historical['max_new_tokens']
                        assert [rgb_sha(bounded(im)) for im in images]==historical['image_sha256']
                        return result['raw'] if i==2 else historical['text']
                replay=Replay()
                prediction,diagnostics=reference._predict_file(paths,motion['base_scores'],motion['new_scores'],replay)
                assert replay.count==4 and prediction['entry_frame']==chosen
                cached_replays+=3
                assert all(prediction[k]==old['baseline_prediction'][k] for k in ('collision_frame','entry_side','evasion_space'))
                row['arms'][arm]=dict(prediction=prediction,raw=result['raw'],valid_offered_integer=result['valid_offered_integer'],
                                     parser_changed=result['parser_changed'],time_seconds=float(times[chosen]),**rational_grade(times[chosen],lo,hi))
                if arm=='baseline':
                    assert call['processor_input_sha256']==old['calls'][2]['processor_input_sha256']
                    row['historical_baseline_raw_match']=result['raw'].strip()==old['calls'][2]['text'].strip()
                    row['historical_baseline_prediction_match']=prediction==old['baseline_prediction']
                parent_worker=next(w for w in report['workers'] if w['ID']==sid and w['arm']==arm)
                assert parent_worker['wall_seconds']>=worker['wall_seconds']>0
                worker_rows.append(dict(ID=sid,arm=arm,entry_frame=chosen,valid_offered_integer=result['valid_offered_integer'],parser_changed=result['parser_changed'],
                     exit_status=parent_worker['exit_status'],parent_wall_seconds=parent_worker['wall_seconds'],worker_wall_seconds=worker['wall_seconds'],
                     generation_seconds=call['seconds'],load_seconds=result['model_load_seconds'],mlx_peak_GB=call['peak_memory'],rss_bytes=worker['rss_bytes']))
                for filename in ('worker_report.json','result.json','calls.json'): hashes[str((folder/filename).relative_to(ROOT))]=sha(folder/filename)
        control=arm_calls['baseline']['processor_input_sha256'];marked=arm_calls['marked']['processor_input_sha256']
        assert set(control)==set(marked) and 'pixel_values' in control
        processor_changed=[key for key in control if control[key]!=marked[key]]
        assert processor_changed==['pixel_values'], processor_changed
        assert arm_calls['baseline']['prompt_sha256']==arm_calls['marked']['prompt_sha256']
        a,b=row['arms']['baseline'],row['arms']['marked']
        bt,mt=times[a['prediction']['entry_frame']],times[b['prediction']['entry_frame']]
        row['definite_gain']=a['result']=='wrong' and b['result']=='correct' and b['valid_offered_integer'] and not b['parser_changed']
        row['definite_loss']=a['result']=='correct' and b['result']=='wrong'
        row['paired_accuracy_delta_bounds']=rational_paired_delta(bt,mt,lo,hi)
        # Difference of two absolute distances is monotone in T, so endpoints suffice.
        endpoint_delta=[abs(mt-t)-abs(bt-t) for t in (lo,hi)]
        row['paired_absolute_error_delta_bounds_seconds']=[float(min(endpoint_delta)),float(max(endpoint_delta))]
        row['false_first_on_late_entry']=not row['already_inside'] and b['prediction']['entry_frame']==numbers[0] and a['prediction']['entry_frame']!=numbers[0]
        rows.append(row)
    same(evaluation['rows'],rows)
    metrics={a:independent_aggregate([r['arms'][a] for r in rows]) for a in ('baseline','marked')}
    same(evaluation['metrics'],metrics)
    accuracy_delta=[sum(r['paired_accuracy_delta_bounds'][i] for r in rows)/len(rows) for i in (0,1)]
    mae_delta=[sum(r['paired_absolute_error_delta_bounds_seconds'][i] for r in rows)/len(rows) for i in (0,1)]
    same(evaluation['paired_accuracy_delta_bounds'],accuracy_delta);same(evaluation['paired_MAE_delta_bounds_seconds'],mae_delta)
    gate=dict(gains=sum(r['definite_gain'] for r in rows),losses=sum(r['definite_loss'] for r in rows),
              no_new_false_first=not any(r['false_first_on_late_entry'] for r in rows),paired_MAE_nonincrease=mae_delta[1]<=1e-9,
              baseline_matches_history=all(r['historical_baseline_prediction_match'] for r in rows))
    gate['pass']=gate['gains']>=1 and gate['losses']==0 and gate['no_new_false_first'] and gate['paired_MAE_nonincrease'] and gate['baseline_matches_history']
    same(evaluation['gate'],gate)
    assert evaluation['actual_model_calls']==calls_count==10 and evaluation['cached_answer_replays']==cached_replays==30
    assert evaluation['other_three_outputs_unchanged']==5 and evaluation['official_S2'] is None and evaluation['automatic_policy'] is False
    assert read(HERE/'run/report.json')==report
    for name,digest in frozen['files'].items(): assert sha(ROOT/name)==digest, name
    for p in (Path(__file__),math_source,HERE/'evaluation.json',HERE/'run/report.json',HERE/'freeze.json'): hashes[str(p.relative_to(ROOT))]=sha(p)
    summary=dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),new_model_calls_by_verifier=0,
        frozen_file_hashes_verified=len(frozen['files']),freeze_precedes_all_workers=True,workers_verified=10,real_model_calls_verified=calls_count,
        cached_answers_replayed=cached_replays,source_images_verified=250,marker_candidate_tiles_verified=60,
        baseline_sheets_exact=5,marked_sheets_exact=5,bounded_model_images_exact=10,only_declared_marker_pixels_changed=True,
        processor_hash_comparison='All5 arm pairs changed pixel_values only; input_ids, attention and grid hashes unchanged where present. Tensor contents not regenerated.',
        raw_parser_and_output_replays_exact=True,other_three_outputs_unchanged=5,all_frozen_files_unchanged_at_end=True,
        fixed_denominator=5,metrics=metrics,paired_accuracy_delta_bounds=accuracy_delta,paired_MAE_delta_bounds_seconds=mae_delta,gate=gate,
        historical_raw_matches=sum(r['historical_baseline_raw_match'] for r in rows),historical_prediction_matches=sum(r['historical_baseline_prediction_match'] for r in rows),
        resource_summary=dict(parent_worker_wall_sum_seconds=sum(w['parent_wall_seconds'] for w in worker_rows),
             top_report_elapsed_seconds=(datetime.fromisoformat(report['ended_utc'])-datetime.fromisoformat(report['started_utc'])).total_seconds(),
             peak_mlx_GB=max(w['mlx_peak_GB'] for w in worker_rows),peak_rss_bytes=max(w['rss_bytes'] for w in worker_rows)),
        limitations=['Exposed five-case privileged-marker development diagnostic, not independent generalization or deployable policy.',
                     'Identity and attention effects are combined; prediction-only outputs do not identify which vehicle the model tracks.',
                     'Semantic validity of expert boxes and interval labels is not certified by this byte/math review.',
                     'Calls reused Q1/Q2/Q4 context; ten fresh Q3 calls are not ten complete four-call model runs.',
                     'Python socket monitoring and stored processor hashes are narrower than full OS/network or independent processor regeneration.',
                     'Mac MLX verification does not prove CUDA/NF4 equivalence; official S2 remains null.'],
        worker_records=worker_rows,rows=rows,artifact_sha256=hashes)
    with targets[0].open('x') as stream: json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# Q3 same-target marker independent verification','','**Runtime, input-contract and scoring verification: PASS.** No new model calls were made by this verifier.','',
           f"Frozen files: {len(frozen['files'])}; fresh workers/calls: 10/10; cached answers replayed: 30; source PNG: 250; candidate tiles: 60.",
           'Five original sheets reproduce the historical Q3 RGB exactly. Five marked sheets reproduce only the fixed yellow rectangle pixels. All10 saved bounded images match actual calls.',
           'Within each arm pair, recorded processor hashes differ only for pixel_values. Prompts, input_ids, attention and image-grid tensors remain unchanged where present. Tensor contents were not regenerated.',
           'Raw text, original JSON parser, offered-frame membership, snap/fallback flags, full four-call stored-answer replay and the other three final fields were independently checked.','',
           '| Case | Control entry | Marked entry | Control grade | Marked grade | Valid marked selection |', '|---|---:|---:|---|---|---|']
    for r in rows:
        a,b=r['arms']['baseline'],r['arms']['marked'];lines.append(f"| {r['ID']} | {a['prediction']['entry_frame']} | {b['prediction']['entry_frame']} | {a['result']} | {b['result']} | {b['valid_offered_integer'] and not b['parser_changed']} |")
    lines += ['',f"Shared-truth accuracy delta bounds: {accuracy_delta}. Shared-truth mean absolute-error delta bounds (seconds): {mae_delta}.",
              f"Frozen diagnostic gate: {gate}. Historical control raw/prediction matches: {summary['historical_raw_matches']}/5 and {summary['historical_prediction_matches']}/5.",
              'Interval grades use exact rational seconds; paired accuracy uses closed acceptance-interval differences. Paired absolute-error extrema were computed from interval endpoints independently of the experiment evaluator.',
              '', '## Scope limits',''] + ['- '+x for x in summary['limitations']]
    lines += ['', 'The source code, frozen inputs/reviews, model files and parent evaluation were not modified. Full hashes, case scores and resource measurements are in independent_verification.json.','']
    with targets[1].open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps({k:summary[k] for k in ('status','frozen_file_hashes_verified','workers_verified','real_model_calls_verified','paired_accuracy_delta_bounds','paired_MAE_delta_bounds_seconds','gate')},indent=2))


if __name__=='__main__':
    main()
