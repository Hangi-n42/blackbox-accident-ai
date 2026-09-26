# DA3 pilot 독립 계산 검토

검토대상 pilot.py, point_motion/prepare.py·windows.json, 저장pose. 실행script/정답/게이트 수정 없음.

## 확인 결과

1. world-to-camera extrinsics에서 C=-R^Tt를 계산하는 einsum축이 맞다. 독립4×4행렬역산과저장C의일치를감사했다.
2. reverse입력을모델에넣은뒤extrinsics를원래순서로돌려같은원래t에적합하므로q는first와같아야한다. reverse를두번복원하는문제없음.
3. quadratic c0+v*t+0.5*a*t²의중앙q=(v·a)/(v·v)계산이맞다. q는고정전역회전·이동·positive scale에불변이다. synthetic straight acceleration/Sim3/reverse 산술검사통과.
4. Sim3 rowvector식 bc.T@ac SVD, properrotation보정, scale후RMS/span 정규화구현맞다. 거의직선궤적은일부회전축결정이약하므로이일치검사로물리pose정확성을입증하지는못한다.
5. sensor_ratio_a_v는창평균이아니다. prepare.py에서중앙v[10],a[10],a[10]/v[10]로만들었고28개모두원본labels_npz 중앙값과일치한다. 새중앙정답보정불필요. 창평균비율은감사JSON의참고값이며사전평가target/게이트를바꾸지않음.
6. 모델q는2초전체quadfit의중앙도함수라가속도의시간변화나회전경로에서실제중앙순간값과차이가날수있다. 이는추정법한계이며정답정의오류가아니다.

## 해석 주의

- 600s cap은forward진입전검사이므로최대현재forward하나만큼초과가능. 전체28창완료를보장하지않는다.
- windows순서에train이먼저있으므로cap으로held도달전끝날수있다. 학습측진단의실패/불안정은기록할수있지만held일반화실패/전체coverage를계산한것처럼쓰면안된다.
- stable검사는first/reverse/middle간일관성이며실제정확도의증거아님. 값이안정적으로틀릴수있다.
- speed^2>1e-12는절대relative-scale수치의하한이다.수학적q비율은scale불변이지만이분모gate자체가모든scale에불변하지는않는다.현재선택은moving>5m/s라zero-speed추론능력평가로확대금지.
- 연속프레임이많아도독립원천수가늘지않고,기존28창이미개발노출.공식점수아님.

## 재실행 가능한 감사

`artifacts/mac_experiments/scipy_compat/.venv/bin/python artifacts/stage3_followthrough_20260920/da3/expert_audit.py`

이명령은저장pose산술만다시계산하며모델forward나정답수정을하지않는다. `expert_arithmetic_audit.json`의saved_poses_audited와snapshot_may_be_incomplete로감사범위를명시한다. pilot최종종료후한번재실행필요.


## 21frame CPU/MPS 교차검사에 따른 해석 수정

동일strictload모델·float32·입력21frame·전처리504·firstreference에서CPU와MPS차이가크다. expanded_11_225 CPU q=-1.496787, MPS q=-1.043118, Sim3궤적오차0.291249. 두pose모두참값은아니며, 이차이는동일방법의backend재현성이현재확보되지않았음을뜻한다. MPS만의숫자로DA3전체의일반성능을기각하지말고CPU3대표창대조의범위까지구분해야한다.

MPS저장20pose의인접움직임을추가진단했다. first7개창모두최대step이frame4→5에집중하고, reverse7개중6개는원래시각복원후frame15→16에집중했다. 입력순서에서같은위치에연관된패턴이며실제차량운동이라고해석할수없다. 첫referenceframe단독튀는현상으로만설명할수도없다. `reference_step_audit.json`은이사후진단의수치다. SDPA·메모리배치·보간·기타연산중구체원인은확인하지않았다. 원인을확정하는광범위한backenddebug는이번범위밖이다.

CPU3창대조선정은기존prefrozen순서첫A/D/C한개씩이다. 정답을보고유리한창을새로고른것이아니다. refhead/resolution/threshold를바꾸지않고device만분리한대조로서타당하다. 다만3개모두개발학습측창이며독립평가가아니다.


## CPU3창 최종 대조 검토

CPU control은72.91초,3창×3변형전체완료. 사전안정성조건통과0/3이다.

| 창 | 센서중앙q | CPU first / reverse복원 / middle | 판정 |
|---|---:|---|---|
| expanded_11_225 가속 | +0.146363 | −1.496787 / −0.780269 / −2.684126 | 세변형모두가속부호불일치·불안정 |
| expanded_19_275 감속 | −0.153632 | +1.211502 / −1.458360 / +2.827462 | first/middle부호불일치·순서민감 |
| expanded_11_525 등속 | −0.001586 | −0.104479 / −0.173395 / +0.738872 | 등속오차·참조민감 |

현재21frame504/2초/Small camera-decoder 경로를가감속특징으로채택할근거없음. CPU에서도명확한센서방향불일치·순서/참조민감성이있어MPS연산만고치면해결된다고말할수없다. 모든DA3규모/시간창/rayhead/최신모델에대한실패판정은아님. 신규classifier학습·점수실험으로확대하지않는판정이타당하다.

최종독립산술감사: MPS20pose+CPU9pose+CPUcrosscheck1=30개,중앙센서28개,CSV q29개비교완료. CPU첫pose는crosscheck재사용으로독립추론이중복된것이아니다. 모든pose역산·회전직교성검사통과. 모델수치/표현실패와후처리산술오류를구분한다.
