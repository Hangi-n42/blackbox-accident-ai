"""Summarize the completed frozen experiment without additional model fitting."""
from pathlib import Path
import hashlib,json
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2),encoding='utf8')

def main():
    report=json.loads((OUT/'report.json').read_text());ext=report['external'];pub=report['public']
    audit={'criterion':'Augmented external stage3 > baseline and > identical-triplication control',
           'external_scores':{name:ext[name]['stage3'] for name in ('baseline','replicated','augmented')},
           'delta_vs_baseline':ext['delta']['stage3'],'delta_vs_replicated':ext['delta_vs_replicated']['stage3'],
           'exceeds_baseline':ext['delta']['stage3']>0,'exceeds_replicated':ext['delta_vs_replicated']['stage3']>0,
           'public_oof_executed':report['public_oof_executed'],
           'executed_source':{'path':'experiment.py','sha256':sha(OUT/'experiment.py')},
           'amendment':{'path':'amendment_frozen_before_results.json','sha256':sha(OUT/'amendment_frozen_before_results.json')},
           'original_gate_record':'gate_before_public.json',
           'note':'Supplementary audit added after completion; original pre-result config, amendment and gate records remain unchanged.'}
    assert audit['public_oof_executed']==(audit['exceeds_baseline'] and audit['exceeds_replicated'])
    save('gate_audit.json',audit)
    text=['# Stage 3 고정 배율 증강 단일 실험 결과','']
    if pub and pub['delta']['stage3']<=0:
        text.extend(['**이번 증강은 미채택을 권고한다.** 동일 외부17 조건의 공개 OOF에서 비증강 대비 개선하지 못했다. 이 실험으로 최종23경로 재학습을 정당화할 근거는 확보하지 못했다. 추가 배율 탐색은 하지 않았다.',''])
    text.extend([
          '같은 외부 17개 학습 경로·6개 검증 경로, 같은 모델·seed42에서 비증강, 단순 3회 복제, 배율 {0.75, 1.0, 1.25} 증강을 비교했다. CPU 2스레드만 사용했다. 검증 특징은 원래 값을 사용했으며, 정지 정답의 조향은 학습·평가에서 제외했다. 최종 23경로 재학습이나 배포는 수행하지 않았다.','',
          '| 평가 | 학습군 | 가감속 Macro-F1 | 조향 Macro-F1 | Stage 3 복합점수 |',
          '|---|---|---:|---:|---:|'])
    for source,title in [(ext,'외부 6경로'),(pub,'공개 5그룹 OOF')]:
        if source is None:continue
        for name,label in [('baseline','비증강'),('replicated','단순 3회 복제'),('augmented','배율 증강')]:
            s=source[name];text.append(f"| {title} | {label} | {s['accel']['macro_f1']:.6f} | {s['steer']['macro_f1']:.6f} | {s['stage3']:.6f} |")
    text.extend(['',f"외부 복합점수의 증강 효과는 비증강 대비 {ext['delta']['stage3']:+.6f}, 단순 복제 대비 {ext['delta_vs_replicated']['stage3']:+.6f}다. 두 기준을 모두 초과하여 사전 gate를 통과했다."])
    if pub:
        text.extend(['',f"동일 외부17 조건의 공개 OOF에서 증강−비증강은 {pub['delta']['stage3']:+.6f}, 증강−단순 복제는 {pub['delta_vs_replicated']['stage3']:+.6f}다.",
                     '', '| 공개 OOF 클래스 | 증강−비증강 F1 | 증강−단순복제 F1 |','|---|---:|---:|'])
        for task in ('accel','steer'):
            for name,value in pub['delta'][task]['class_f1'].items():text.append(f"| {name} | {value:+.6f} | {pub['delta_vs_replicated'][task]['class_f1'][name]:+.6f} |")
        previous=ROOT/'research/stage3_temporal_v1/public_oof_comparison.csv';current=OUT/'public_oof_comparison.csv'
        a=pd.read_csv(previous);b=pd.read_csv(current)
        joined=a.merge(b,on=['ID','frame_index'],suffixes=('_previous','_current'),validate='one_to_one')
        reproduced={'rows':len(joined),'accel_changed':int((joined['accel_raw']!=joined['baseline_accel']).sum()),
                    'steer_changed':int((joined['steer_raw']!=joined['baseline_steer']).sum()),
                    'previous_predictions_sha256':sha(previous),'current_predictions_sha256':sha(current),
                    'same_split_sha256':sha(OUT/'external_split.json')}
        save('baseline_reproduction.json',reproduced)
        text.extend(['',f"이번에 다시 학습한 비증강 OOF와 temporal_v1의 비증강 OOF를 비교했다. {len(joined)}개 공개 라벨행에서 가감속 예측 변화 {reproduced['accel_changed']}개, 조향 변화 {reproduced['steer_changed']}개였다.",
                     '',f"기존 외부23경로 모델의 공개 OOF {pub['existing_external23_reference']:.6f}는 학습량이 다른 참고값이다. 이번 외부17경로 증강값과의 차이를 증강 효과로 해석하지 않는다."])
    text.extend(['','## 해석과 제한','',
                 '- 공개 OOF는 이미 여러 모델 선택에 사용한 동일 50개 희소 라벨의 개발 진단이다. 독립 테스트 정확도나 private 성능으로 해석할 수 없다. 외부 6경로 검증도 이전 실험에 사용되었다.',
                 '- 특징 배율 증강은 실제 카메라 FOV 변환과 동치가 아니다. 외부 CAN 범주 임계값도 대회 공식 임계값이 아닌 기존 proxy 규칙이다.',
                 '- 단순 복제 자체도 규제·bootstrap·최소 리프 표본수의 영향을 바꾼다. 같은 표본수의 복제 대조군을 함께 비교했다. 이 대조군은 채택 후보가 아니다.',
                 '- 외부 평균 개선은 모든 클래스·경로의 개선을 뜻하지 않는다. 외부 클래스별·경로별 변화는 external_findings.md와 external_validation.json, 공개 상세는 public_oof.json을 참조한다.',
                 '- 각 external17_*.joblib은 외부17경로만으로 학습한 실험 재현용 체크포인트다. 공개 전체를 더한 제출용 최종 모델이 아니다.',
                 '',f"완료된 실행 시간: {report['elapsed_seconds']:.2f}초. 대조군 추가 전 중단한 최초 실행 시간은 이 값에 포함하지 않는다. 기존 코드·모델의 SHA 및 ZIP size/mtime 보존 검사: {report['protected_files_unchanged']}.",
                 '', '설정·출처·소스·분할·특징 캐시 해시·예측·체크포인트·점수는 이 디렉터리에 보관했다. 대조군은 결과 확인 전에 별도 amendment로 고정했고 원본 설정을 덮어쓰지 않았다. gate_audit.json은 완료 후 양쪽 gate 수치와 실제 실행 소스 SHA를 보완 기록한다.'])
    (OUT/'summary.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'external_delta':ext['delta']['stage3'],'external_delta_vs_replicated':ext['delta_vs_replicated']['stage3'],
                      'public_delta':None if pub is None else pub['delta']['stage3'],
                      'public_delta_vs_replicated':None if pub is None else pub['delta_vs_replicated']['stage3']},indent=2))

if __name__=='__main__':main()
