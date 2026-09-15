"""Summaries of completed audits and fixed experiment; no model fitting."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from research.stage3_temporal_experiment import load_external,ACCEL,STEER
from research.v4_stage3.source_weight_experiment import scores
OUT=Path(__file__).resolve().parent

def main():
    audit=json.loads((OUT/'audit_report.json').read_text());report=json.loads((OUT/'experiment_report.json').read_text());codec=json.loads((OUT/'codec_sparse_diagnostic.json').read_text())
    folds=report['training_details'];paired=[]
    for b in [r for r in folds if r['arm']=='baseline']:
        c=next(r for r in folds if r['arm']=='source_equal' and r['scope']==b['scope'] and r['task']==b['task'] and r.get('held')==b.get('held'))
        paired.append({'scope':b['scope'],'task':b['task'],'held':b.get('held'),
          'same_x_y_and_class_weights':all(b[k]==c[k] for k in ['feature_sha256','target_sha256','class_weight_values']),
          'same_effective_total':np.isclose(b['effective_weight_sum'],c['effective_weight_sum']).item()})
    assert all(r['same_x_y_and_class_weights'] and r['same_effective_total'] for r in paired)
    (OUT/'weight_control_audit.json').write_text(json.dumps(paired,indent=2),encoding='utf8')
    rows=[r for r in load_external() if r['vehicle']=='99c94dc769b5d96e'];pred={a:np.load(OUT/f'source_stress_{a}_predictions.npz') for a in ('baseline','source_equal')}
    route=[];offset=0
    for r in rows:
        n=len(r['accel']);record={'route':r['route'],'rows':n}
        for arm in pred:
            m=scores(r['accel'],r['steer'],pred[arm]['accel'][offset:offset+n],pred[arm]['steer'][offset:offset+n])
            record[arm+'_stage3']=m['stage3'];record[arm+'_accel']=m['accel']['macro_f1'];record[arm+'_steer']=m['steer']['macro_f1']
        record['delta']=record['source_equal_stage3']-record['baseline_stage3'];route.append(record);offset+=n
    pd.DataFrame(route).to_csv(OUT/'source_stress_per_route.csv',index=False)
    pub=pd.read_csv(OUT/'public_oof_predictions.csv');old=pd.read_csv(ROOT/'research/stage3_oof_external.csv')
    merged=pub.merge(old,on=['ID','frame_index'],suffixes=('_current','_old'),validate='one_to_one')
    reproducibility={'public_rows':len(merged),'old_baseline_changes':{'accel':int((merged.baseline_accel!=merged.accel_linear).sum()),'steer':int((merged.baseline_steer!=merged.steer_forest).sum())},
      'candidate_public_changed':{t:int((pub['baseline_'+t]!=pub['source_equal_'+t]).sum()) for t in ('accel','steer')}}
    (OUT/'baseline_reproduction.json').write_text(json.dumps(reproducibility,indent=2),encoding='utf8')
    text=['# V4 Stage3 시간·라벨 감사 및 출처 가중치 단일 실험','',
      '기존 제출 코드·모델·ZIP은 보존했다. GPU 없이 CPU 2스레드만 사용했다. 후보 전체자료 재학습이나 최종 체크포인트 생성은 하지 않았다.','',
      '## 1. 시간·라벨 감사','',
      '- 공개 5영상의 동일 해독 픽셀에서 train20Hz/stride2와 deploy10Hz/stride1 특징 및 flow midpoint가 일치했다. 기존 학습 캐시까지 비트 단위로 일치했다.',
      '- 외부 학습23구간 모두 frame_times와 특징 캐시 행 수가 맞고 timestamp는 증가한다.',
      f"- OPEN_001과 제외된 CAN 구간은1200프레임 전체 해독 BGR SHA가 같았다. 공개 라벨시각과 상대 CAN 시각의 최대 차이는 {audit['public_can']['max_label_nominal_vs_can_seconds']:.9f}초였다. CAN 범위 밖2프레임은 기존 np.interp 경계값으로 처리되어 있으며 정책은 바꾸지 않았다.",
      '- 고정proxy와 공개GT의 일치는 가감속10/10, 조향9/10이다. 가감속10개는 전부 CONSTANT여서 가감속/정지 임계값을 검증하지 못한다. 36초 조향은 GT STRAIGHT, proxy RIGHT(-3.96187도)로 불일치했다. 공식 임계값을 추정·변경하지 않았다.',
      '', '## 2. 재인코딩 민감도','',
      '기존 corrected10Hz 파일은 CRF18로 재인코딩됐다. 동일 픽셀 시간축 비교와 다르다. 2998행 중 가감속128행, 조향130행 예측이 달라졌다. 아래는 공개GT를 이미 학습한 고정 모델의 학습자료 진단이며 OOF/독립 성능이 아니다.',
      '', '| 픽셀 입력 | 가감속 F1 | 조향 F1 | S3 |','|---|---:|---:|---:|']
    for k,label in [('original_pixels','원본 해독'),('crf18_corrected_10hz','CRF18 재인코딩')]:
        r=codec[k];text.append(f"| {label} | {r['accel']['macro_f1']:.6f} | {r['steer']['macro_f1']:.6f} | {r['stage3']:.6f} |")
    text.extend(['','공개50라벨 위치에서는 가감속4개·조향3개 예측이 달라졌다. 코덱 민감도가 공식 점수의 원인이라고 확정할 수 없으며 이번에 코덱 증강을 추가하지 않았다.','',
      '## 3. 사전고정 비교와 결과','',
      '동일 행·특징·정답·seed·모델·unweighted scaler·class_weight 수치를 유지했다. 후보는 공개/외부 각각의 class_weight×sample_weight 합을 N/2로 맞춰 총 유효가중치N을 보존했다. 표본을 복제하지 않았다. 출처를 재가중하면 클래스별 유효질량도 바뀌며 이를 모든fold에 기록했다. 출처1:1과 클래스별질량 보존을 동시에 주장하지 않는다.','',
      '| 진단 | 군 | 가감속 Macro-F1 | 조향 Macro-F1 | S3 |','|---|---|---:|---:|---:|'])
    for scope,title in [('public','공개5그룹 OOF'),('source_stress','RAV4→Civic 밀집 stress')]:
        for arm,label in [('baseline','기존가중'),('source_equal','출처1:1')]:
            r=report[scope][arm];text.append(f"| {title} | {label} | {r['accel']['macro_f1']:.6f} | {r['steer']['macro_f1']:.6f} | {r['stage3']:.6f} |")
    text.extend(['',f"공개OOF delta={report['public_stage3_delta']:+.6f}; 차량source stress delta={report['source_stress_stage3_delta']:+.6f}. 상태: {report['status']}.",
      f"source stress 경로별 개선 {sum(r['delta']>0 for r in route)}/12, 하락 {sum(r['delta']<0 for r in route)}/12. 경로별 Macro-F1은 결측클래스에0을 사용하므로 그 평균을 전체 pooled 점수로 대체하지 않는다.",
      f"기존 external23 OOF 재현: {reproducibility['old_baseline_changes']}. 후보의 공개 예측 변경: {reproducibility['candidate_public_changed']}.",
      '', '## 4. 해석 제한과 결정','',
      '- 공개OOF는 같은50개 희소 라벨을 반복 사용한 개발진단이다. 나머지 공개4영상의 원출처는 확정되지 않았다.',
      '- OPEN_001과 일치한 원본segment 및 그route는 모든23개 외부학습목록에서 제외되어 있다. OPEN_001 heldout에 이 알려진 동일영상proxy가 들어간 것은 아니다. 포함검사와실제목록은 overlap_audit.json에 기록했다. 다른4영상의전체출처중복미확인과구분한다.',
      '- source stress 학습은 출처가 확인된 OPEN_001 GT10개와 다른 RAV4 11구간뿐이다. Civic12구간은 이 모델 학습에 포함하지 않았다. 공개4개는 출처불명 때문에 제외했다.',
      '- 이는 두 차량·특정 고속도로의 단방향 스트레스이며 새 untouched 데이터가 아니다. GT10개 가감속이 전부 등속이라는 제한이 있고, 최종공개50개 후보와 학습자료가 다르다.',
      '- 기존17/6 분할은 이미 사용한 개발셋이다. 이번에는 정보가 중복되는 추가17/6 반복을 하지 않았다.',
      '- 외부proxy는 공식정답 정의와 동일하지 않다. 공개OOF만 상승해도 자동 채택하지 않으며, root 검토 전 최종23구간 재학습·모델 생성은 없다.',
      f"- 감사 {audit['elapsed_seconds']:.2f}초, 가중치 실험(코덱 sparse 진단 포함) {report['seconds']:.2f}초. 보호파일 검사 {report['protected_unchanged']}.",
      '', '설정은 audit_plan_frozen.json 및 experiment_plan_frozen.json, 상세값은 audit_report.json 및 experiment_report.json, 가중치 통제는 weight_control_audit.json에 있다. 라이선스·원출처는 기존 comma2k19 manifest와 감사에 보존한 원source 경로를 따른다.'])
    (OUT/'summary.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'public_delta':report['public_stage3_delta'],'source_delta':report['source_stress_stage3_delta'],'status':report['status'],'reproduction':reproducibility},indent=2))

if __name__=='__main__':main()
