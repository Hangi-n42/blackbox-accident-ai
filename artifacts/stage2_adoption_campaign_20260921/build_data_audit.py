"""Metadata/reference audit only. Never opens model prediction/call/result files."""
import hashlib,json,re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
READS=[]
def read(name):
    p=ROOT/name;READS.append(str(p.relative_to(ROOT)));return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def local(value):
    value=value.replace('\\','/')
    if '/research/' in value:value='research/'+value.split('/research/',1)[1]
    return ROOT/value
def rel(p):return str(p.relative_to(ROOT))
def main():
    assert not (OUT/'data_audit.json').exists()
    sources={}
    acq_names=['research/v6/nexar_review_candidates/acquisition.json','research/v6/nexar_review_round2/acquisition.json','research/v7/fresh_sources_retry1/acquisition.json','research/v7/new_validation_sources/acquisition.json','artifacts/stage2_goal_20260919/acquisition/acquisition.json','artifacts/stage2_goal_20260919/acquisition/acquisition_round2.json']
    for name in acq_names:
        for r in read(name)['records']:
            p=local(r.get('path',r.get('source_video')));sid=r.get('ID',r.get('id',p.stem));h=r.get('sha256',r.get('local_sha256'))
            sources[sid]=dict(ID=sid,source_video=rel(p),provider_path='train/positive/'+sid+'.mp4',source_sha256=h,source_manifest=name,source_uri=r.get('url',r.get('source_uri',r.get('source_url'))),declared_independence=r.get('independence_verified',False))
    annotation_name='artifacts/data_curation_20260917/nexar/annotations.json';annotations=read(annotation_name)['records'];annot={r['id']:r for r in annotations}
    split_name='artifacts/data_curation_20260917/nexar/split_manifest.json';split={r['id']:r for r in read(split_name)}
    for r in annotations:
        s=r['source'];sources[r['id']]=dict(ID=r['id'],source_video=s['video'],provider_path=s['provider_path'],source_sha256=s['sha256'],source_manifest=annotation_name,source_uri=None,declared_independence=False)
    for r in sources.values():
        p=ROOT/r['source_video'];r.update(exists=p.exists(),actual_bytes=p.stat().st_size if p.exists() else None,actual_sha256=sha(p) if p.exists() else None)
        r['matches_recorded_sha256']=r['actual_sha256']==r['source_sha256']
    train=['00000','00003','00006','00013'];train_hashes={sources[s]['actual_sha256'] for s in train}
    grouped=defaultdict(list)
    for sid,r in sources.items():grouped[r['actual_sha256']].append(sid)
    duplicates=[ids for h,ids in grouped.items() if h and len(ids)>1]
    human=[]
    paths=list((ROOT/'research/v6_stage2/user_reviews').glob('NEXAR_REVIEW_*_review_*.json'))+list((ROOT/'research/v6_stage2/round2_validation/intake_20260915_actual/user_reviews').glob('*/*_review_*.json'))
    for p in sorted(paths):
        d=read(rel(p));human.append(dict(path=rel(p),sha256=sha(p),ID=d['ID'],created_at=d.get('created_at'),record_type=d.get('record_type'),evaluation_eligible=d.get('evaluation_eligible'),source_group_id=d.get('source_group_id'),source_sha256=d.get('source_video_sha256'),stored_exposure=d.get('exposure'),current_exposure='Previously reviewed and evaluated; stored unseen is historical, not current.',entry=d.get('review',{}).get('entry'),requires=d.get('requires')))
    ai3=read('research/v7/ai_review_stage2/review_summary.json');ai6=read('research/v7/next_review_adjudication.json')
    ai_notes=[]
    for r in ai3['reviews']:ai_notes.append(dict(ID=r['ID'],reference_type='AI_secondary_evidence',contact=r['contact']['status'],entry=r['entry']['status'],entry_eligible=False,reason=r['entry']['reason'],source_sha256=r['source_sha256']))
    for r in ai6['records']:ai_notes.append(dict(ID=r['ID'],reference_type='AI_secondary_evidence',contact='conditional_inferred_interval' if r['contact_interval'] else 'unknown',entry='unknown' if r['entry'] is None else r['entry'],entry_eligible=False,source_sha256=r['source_sha256']))
    ai24=read('artifacts/stage2_goal_20260919/acquisition/ai_screening.json')
    # Enumerate paths only. Do not deserialize calls, predictions or model results.
    caches=defaultdict(list)
    for p in (ROOT/'artifacts/data_pilot_20260916').rglob('calls.json'):
        if 'baseline' in str(p):caches[p.parent.name].append(dict(path=rel(p),bytes=p.stat().st_size,content_read=False))
    manifests=['artifacts/data_pilot_20260916/nexar/inference_inputs.json','artifacts/data_pilot_20260916/nexar/inference_inputs_round2.json','artifacts/data_pilot_20260916/nexar/inference_inputs_six.json','artifacts/data_pilot_20260916/nexar_validation/inference_inputs.json']
    inputmeta=defaultdict(list)
    for name in manifests:
        for case in read(name):
            images=case['images'];inputmeta[case['ID']].append(dict(manifest=name,frames=len(images),first_frame=images[0]['frame'],last_frame=images[-1]['frame'],frame_directory=str(Path(images[0]['path']).parent),first_image_exists=(ROOT/images[0]['path']).exists(),last_image_exists=(ROOT/images[-1]['path']).exists()))
    priorities=['00118','00222','00554','00362','00630','00537','00199','00863','00658'];queue=[]
    for sid in priorities:
        a=annot[sid];entry=a['entry'];s=sources[sid]
        conditional=sid in priorities[:6]
        queue.append(dict(ID=sid,priority=1 if conditional else 2,source_video=s['source_video'],source_sha256=s['actual_sha256'],provider_path=s['provider_path'],distinct_provider_clip_from_train4=sid not in train,byte_distinct_from_train4=s['actual_sha256'] not in train_hashes,incident_independence='unverified',reference_type='AI secondary / previously exposed development',contact_status=a['contact']['status'],entry_status=entry['status'],entry_interval_frames=entry.get('frame_interval'),entry_interval_seconds=entry.get('time_interval_s'),already_inside_at_start=entry.get('already_inside_at_start'),identity_is_candidate_only=entry.get('identity_is_candidate_only'),counterpart=a['counterpart'],entry_evidence=entry.get('evidence'),entry_eligible_as_existing_strong_reference=False,weak_before_start_policy_candidate=bool(entry.get('already_inside_at_start') and entry.get('identity_is_candidate_only') is False),proposed_action='Re-adjudicate applicability/identity and freeze a weak competition-convention reference before reading predictions.' if conditional else 'Low priority: contact/counterpart or geometry uncertain; retain unknown unless new visible evidence resolves it.',source_annotation=a['source_annotation'],entry_evidence_annotation_source=entry.get('evidence_annotation_source'),evidence_images=entry.get('evidence_images',a['contact'].get('evidence_images',[])),history=split[sid]['history'],baseline_cache_metadata=caches[sid],input_metadata=inputmeta[sid]))
    overlap=read('research/v6_stage2/nexar_source_overlap_audit/report.json')
    summary=Counter(a['entry']['status'] for a in annotations)
    result=dict(status='COMPLETE_METADATA_AUDIT',created_utc=datetime.now(timezone.utc).isoformat(),scope='Local reference/provenance/cache-path audit. No new visual adjudication, prediction contents, fitting, inference or downloads.',train_ids=train,new_human_reviews_found=0,human_record_count=len(human),human_unique_nexar_cases=len({r['ID'] for r in human}),latest_human_created_at=max(r['created_at'] for r in human),human_records=human,source_count=len(sources),actual_sources_hash_verified=sum(r['matches_recorded_sha256'] for r in sources.values()),sources=sorted(sources.values(),key=lambda r:r['ID']),exact_byte_duplicate_id_groups=duplicates,incident_independence_certified=False,independence_limits=['Different provider clip IDs and file SHA exclude byte-identical files only; edited reposts or continuous-drive/incident overlap remain unverified.','Nexar provider metadata has event/alert time and light/weather/scene fields, not an original incident/drive identity.','Existing11-source/55-pair sampled overlap audit had no candidates but status remained visual_review_pending and excludes later cohorts.'],existing_overlap_audit=dict(path='research/v6_stage2/nexar_source_overlap_audit/report.json',status=overlap['status'],source_count=overlap['source_count'],pair_count=overlap['pair_count'],candidate_count=len(overlap['candidates'])),ai_14_23_reference_status=ai_notes,ai_14_23_entry_evaluable=0,known_24_29_status='Existing six-case AI screening, no confirmed collision/entry truth; do not repeat as new independent data.',latest_curation_40=dict(path=annotation_name,entry_status_counts=dict(summary),exact_entry_eligible=sum(bool(a['entry'].get('exact_timing_label_eligible')) for a in annotations),independent_evaluation_eligible=sum(bool(a['use'].get('independent_evaluation_eligible')) for a in annotations),all_exposed_development=all(s['split']=='development_exposed' for s in split.values())),review_candidate_queue=queue,review_candidate_count=len(queue),maximum_additional_visual_screens=12,visual_screens_performed_by_this_audit=0,ready_for_new_human_or_official_entry_evaluation=0,weak_reference_candidates_before_review=6,baseline_cache_reuse='Six priority cases have baseline calls and inference frame manifests. Contents have NOT been opened; current12-candidate/RGB/processor/code equivalence remains unverified and must be checked only after reference/eligibility freeze.',evaluation_contract=['Freeze all screened cases, retained unknowns, exact source hashes and eligibility before opening reserved predictions.','Report official supplied labels, human drafts, provider event hints, AI conditional intervals and task-convention entry0 separately.','A first-frame entry policy may map same-counterpart already-inside observations to0 under the competition rule; do not rewrite physical left-censoring as exact physical entry time.','Count eligible incidents, not frames/candidate pairs; report before-start/during-clip separately and never convert unknown to wrong or correct.','A broad reference interval is graded as correct/wrong/indeterminate and paired changes share the same unknown time.','All selected sources already underwent review and some baseline evaluation. This is expanded development evidence, not a new sealed independent test.'],read_reference_metadata_paths=sorted(set(READS)),prediction_content_files_read=[],new_model_calls=0,new_training_runs=0,new_downloads=0,existing_files_modified=False)
    (OUT/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# Stage2 캠페인 자료 감사','','새 사람 정답은 확인되지 않았다. 검색한 Nexar 사람 초안은 '+str(len(human))+'개 파일/9사례이고 00003 수정본 중복을 포함한다. 최신 작성시각은 '+result['latest_human_created_at']+'이다. 모두 기존 검수·평가에 노출됐으며 원기록의 unseen은 과거 시점 표기다. 공식 정답이나 독립확정 사람GT로 승격하지 않는다.','','## 참조 적격성','','- 00014/15/16 AI검수3건과 00017/18/19/21/22/23 조정6건은 진입 참조 적격0건. 접촉추론구간3개는 진입정답이 아니다.','- Nexar40 최신20260917 참조는 exact entry0건·독립평가0건이다. 일부 오래된 entry0 기록보다 최신 physical left-censoring/조건부 주석을 우선한다.','- 다만 동일 상대의 시작부터 내부 관찰은 대회 before-start→첫프레임 규칙에 따른 약한 AI참조 후보가 될 수 있다. 물리적 최초진입 시점과 제출규칙을 분리해 새 참조를 예측열람 전에 고정해야 한다.','','## 최대12건 중 우선9건 제안','','|ID|현재 AI 진입 근거|우선도|baseline cache|','|---|---|---:|---:|']
    for q in queue:lines.append(f"|{q['ID']}|{q['entry_status']}|{q['priority']}|{len(q['baseline_cache_metadata'])}|")
    lines+=['','우선6건: 00118·00222·00554·00362·00630은 같은 상대가 시작부터 내부라는 관찰, 00537은 진입 가능구간이 있으나 미확정이다. 추가3건00199·00863·00658은 실제 접촉/상대 또는 경계가 불확실하여 현재 평가부적격이다. 새 영상검수는 이 감사에서 수행하지 않았으며 예측 내용을 열지 않았다.','','## 원천·노출·재사용','',f"로컬 Nexar {len(sources)}원본의 실제 SHA를 검사했고 {result['actual_sources_hash_verified']}개가 기록과 일치했다. ID가 다른 byte중복 그룹은 {duplicates}이다. 이는 사고 독립성 인증이 아니다. 원천 drive/incident ID가 없고 재편집·재업로드 중복은 배제하지 못한다.",'','기존 baseline calls와 원프레임 manifest 경로는 우선6건에 존재한다. calls 내용은 읽지 않았으므로 현재 Q3 12후보·RGB·processor·코드와 일치한다고 아직 주장할 수 없다. 참조/적격성을 잠근 뒤 기존캐시 CPU 재생 또는 동일입력 대조가 필요하다.','','## 평가 분모','','새 공식/확정 사람 진입평가 적격분모는0이다. 약한 추가참조 후보6건도 아직 적격확정분모가 아니다. 검수한 모든 사례/unknown을 남기고, 실제 적격사고 수와 before-start/during-clip을 각각 보고한다. 기존 노출자료를 새로운 독립평가라고 부르지 않는다.','','근거: [사람 초안](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/user_reviews/), [round2 초안](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/round2_validation/intake_20260915_actual/user_reviews/), [AI3](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/ai_review_stage2/review_summary.json), [AI6조정](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/next_review_adjudication.json), [최신40참조](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/data_curation_20260917/nexar/annotations.json), [노출기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/data_curation_20260917/nexar/split_manifest.json). 모든 원본 SHA·정확 경로·cache metadata는 data_audit.json에 보존했다.']
    (OUT/'data_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['status','source_count','actual_sources_hash_verified','exact_byte_duplicate_id_groups','human_record_count','human_unique_nexar_cases','latest_human_created_at','review_candidate_count','ready_for_new_human_or_official_entry_evaluation']},ensure_ascii=False))
if __name__=='__main__':main()
