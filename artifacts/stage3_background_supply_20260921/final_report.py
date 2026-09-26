from run import *
from PIL import Image,ImageDraw
from collections import Counter
import shutil
rows=read(O/'validated/results.json');raw_pair=read(O/'corrected/source_pair.json');previous=read(S/'results.json');old={r['key']:r for r in previous};baseline09=read(O/'corrected/baseline_expanded09.json');allrows={r['key']:r for r in rows}
# Final point overlays: strict candidate is a subset of RGB-reviewed source-pair points.
(O/'validated/evidence').mkdir(exist_ok=True)
for r in rows:
 key=r['key'];rgb=np.load(B/'depth'/(key+'_input.npz'))['rgb'];pts=np.load(O/'validated'/(key+'.npz'));sheet=Image.new('RGB',(1008,816),'white');dd=ImageDraw.Draw(sheet)
 for n,j in enumerate([0,9,12,18]):
  im=Image.fromarray(rgb[j]);d=ImageDraw.Draw(im);p=pts[f'p{j}'];ii=set(r['pnp'][j].get('inlier_indices',[]))
  for i,xy in enumerate(p):
   x,y=map(float,xy);d.ellipse((x-1.2,y-1.2,x+1.2,y+1.2),fill='lime' if i in ii else 'orange')
  x=n%2*504;y=n//2*408;sheet.paste(im.resize((504,378)),(x,y+24));dd.text((x+3,y+3),f'{key} p{j} n={len(p)} PnP={r["pnp"][j]["valid"]}',fill='black')
 sheet.save(O/'validated/evidence'/(key+'.jpg'))
summary={'windows':8,'frame_pairs':160,'final_correspondence_observations':sum(s['points'] for r in rows for s in r['stats']),'point_sufficient_pairs':sum(r['point_sufficient_pairs'] for r in rows),'point_sufficient_windows':[r['key'] for r in rows if r['point_sufficient_window']],'PnP_valid_pairs':sum(r['pnp_valid_pairs'] for r in rows),'PnP_usable_windows':[r['key'] for r in rows if r['pnp_quality']],'score_measured':False,'classifier_fits':0,'new_depth_calls':0,'downloads':0,'status':'Partial point-supply recovery; full8-window bottleneck NOT resolved; no ready motion feature.'}
write(O/'summary.json',summary)
# Link every usable pair, preserve rejection reasons and provenance in the full manifest.
manifest=read(O/'validated/manifest.json');write(O/'validated/sufficient_pairs.json',[r for r in manifest if r['point_supply_sufficient']]);write(O/'validated/PnP_valid_pairs.json',[r for r in manifest if r['PnP_valid']])
write(O/'visual_review.json',{'reviewer':'AI; not human or official correspondence truth','reviewed':'40 RGB anchor frames; all168 expanded-mask overlay frames;24 source/target pair overlay comparisons; expanded09 bonnet crop0/10/19. Previous168 RGB review reused.','pre_measurement_corrections':['Remove expanded19 distant roof patch uncertain sky boundary','Remove expanded14 median strip containing vegetation'], 'post_measurement_semantic_correction':'expanded09 bonnet top seen in source-pair overlay and confirmed in crops. All raw attempts preserved, cutoff y>=.69H, corresponding source/destination points removed and results recomputed in corrected/. Final uses corrected masks.','remaining_limits':['Road shadows, wet-road reflections, windshield reflections and glass facade appearance can move despite static underlying surfaces.','PnP fit and DIS/LK agreement do not prove physically correct correspondences.','Manual mask coverage remains incomplete; thin poles and small far buildings not confidently annotated are omitted.'], 'scope':'Development-diagnostic masks only, no production segmentation.'})
shutil.copyfile(S/'SOURCE_LICENSE.txt',O/'SOURCE_LICENSE.txt');shutil.copyfile(S/'DA3_LICENSE',O/'DA3_LICENSE')
with (O/'SOURCE_LICENSE.txt').open('a') as f:f.write('\nCurrent changes: expanded AI-reviewed masks, bonnet correction, dense/adaptive source sampling, DIS/LK correspondence comparisons and PnP diagnostics. Original source lineage: ../stage3_depth_pnp_20260920/manifest.json. No official or human labels claimed.\n')
raw_table='\n'.join(f"| {r['key']} | {float(np.median([x['retained_points'] for x in old[r['key']]['selection']])):g} | {next(z for z in raw_pair if z['key']==r['key'])['methods']['DIS']['median_points']:g} | {r['median_points']:g} | {r['point_sufficient_pairs']}/20 | {r['pnp_valid_pairs']}/20 |" for r in rows)
flow_reasons=Counter(p['reason'] for r in rows for p in r['pnp'])
report=f'''# Stage3 배경 대응점 확보 — 2026-09-21

**판정: 일부 확보. 8구간 전체의 병목 해결에는 실패했다.** 최종 엄격한 후보는 {summary['final_correspondence_observations']:,}개 대응점 관측, 점 수·분포 통과24/160쌍, 기존 PnP 통과22/160쌍이다. ZOD000026 한 구간만 전체 창의 점 공급 기준을 통과했고, 완성된 가감속 특징의 구간 품질을 통과한 것은0/8이다. 점을 많이 확보한 것, 정지 배경에 정확히 대응하는 것, PnP 안정성, S3 개선을 구분한다. S3 평가·분류기 학습·운영 변경은 하지 않았다.

## 비교 순서와 고정 조건

1. 이전 마스크 그대로 격자8px→4px: 샘플링 간격만 변경.
2. 같은4px 격자에서 RGB로 검수한 도로·건물·분리대·가드레일 영역 추가: 영역만 변경.
3. 같은 확장 영역에서2px 후보를 만들고 모서리 응답 우선으로 반경4px 중복 억제: 샘플링 정책만 변경. 모서리 응답은 순위이지 정답 보증이 아니다.
4. 목적지에서 탈락하는 영향을 분리하려고 동일한 원천 선정점을 DIS와 LK에 각각 추적. 두 방법에 동일한 추가 밝기차 검사(MAE≤20)를 적용했다. 이 대조 내부만 추적기 효과를 분리한 비교이며,3번 대비 변화 전부를 추적기 탓으로 돌리지 않는다.
5. 검수 중 본넷 오염을 찾아 RAV4 expanded09의 y≥.69H 점을 제거. 원본/이전 결과는 보존하고 정정 결과를 별도 저장했다.
6. 2D 추적점의 공급과 DA3 깊이 신뢰도 탈락을 분리해 계수. 최종 후보는 **두 추적기의 FB≤1px·밝기차·질감·양끝 마스크 조건 통과, 목적지 좌표 차이≤1px, 기존 깊이 신뢰도 기준 통과** 교집합이다. 낮은 깊이 신뢰도 점은 PnP 입력으로 올리지 않았다.

기존 영상8×21프레임, DIS_FAST, 저장 깊이/K/시각, 7px 질감 표준편차≥4, 원래8px 격자에서 정한 깊이 신뢰도 중앙값, PnP·RANSAC/LM, 최소30인라이어, 가로40%·세로20% 분포, 구간16/20 유효쌍 및 양끝 각3/4 조건은 유지했다. DIS 밀집광류는 한 번 재계산해 모든 대조에서 재사용했다. 추가 깊이 추론·다운로드·패키지 설치·모델 학습은0회.

방법 근거는 [OpenCV 모서리 검출 문서](https://docs.opencv.org/4.10.0/dd/d1a/group__imgproc__feature.html)와 [광류 문서](https://docs.opencv.org/4.10.0/d4/dee/tutorial_optical_flow.html)다. 설치된 함수로 구현했으며, 문서가 대회 성능을 보장한다는 의미는 아니다.

## 수량·분포와 실제 사용 가능성

아래 점 수는 프레임 쌍별 중앙값이다. 마지막 열은 종합 구간 통과가 아니라 개별 PnP 쌍 통과 수다. 서로 가까운 관측과 반복 프레임을 독립3D점 수로 세지 않는다.

| 구간 | 이전8px 점 수 | 새 공급 DIS 점 수 | 최종 교집합 점 수 | 최종 수·분포 통과 | 최종 PnP 통과 |
|---|---:|---:|---:|---:|---:|
{raw_table}

expanded09 이전 중앙값21에는 이번에 발견한 본넷 오염이 포함되어 있다. 같은 의미상 정정을 적용한 이전 기준의 중앙값은 {baseline09['median_points']:g}다. 다른 방법의 새 수치는 모두 정정 후 결과다. 이 사례의 이전/정정 전 ‘분포 확보’ 결과를 성공 근거로 사용하지 않는다.

### 확보한 구간

`zod_000026_161`은 와이퍼에 가려진 f5/f11 때문에4쌍을 제외했다. 나머지16쌍 모두 최종 교집합에서72~164개 점을 확보하고 수·분포 기준을 통과했다. 이16쌍 중15쌍이 기존 PnP 검사를 통과했다. 따라서 **이 구간의 가시 프레임에서는 배경점 수·분포 부족을 해소했다.**

남은 pair18은 점129개로 수·분포가 충분하지만, 앞뒤 이동 추정의 일관성 비율1.487이 기존≤0.25를 넘었다. 점 수 부족으로 분류하지 않는다. 최소16개 유효 PnP쌍 조건은 유지하여 전체 구간 품질은 실패다. 미통과 q를 학습 특징으로 사용하지 않는다. 기존 ZOD 카메라 왜곡·깊이 시간 안정성 문제도 해결되지 않았다.

### 해결하지 못한 구간

- comma 일부는 도로 표시·연석에 점이 집중되어 화면 상하/좌우 지지가 부족하다. 시작점 수를 늘려도 독립적인 구조가 자동으로 추가되지 않는다.
- `expanded_19`는 깊이 신뢰도를 적용하지 않으면 DIS/LK 각각20/20쌍에서2D 수·분포 조건을 통과했다. 그러나 두 추적기가 동의하는 점으로 제한하면1/20쌍만 남았다. 점 공급, 대응 위치 일치, 깊이 사용 가능성을 같은 것으로 취급하면 안 된다.
- ZOD000002는 점 수가 늘어도 차선 표시 부근의 좁은 영역에 집중되어 공간 분포가 실패했다. 초반 트럭이 우측 배경을 가린다.
- 서로 다른 추적기 동의와 밝기차 통과는 오류를 줄이는 진단 조건이며 정확한 대응점 정답의 대체 증명이 아니다. 도로 그림자·반사와 수동 영역 누락도 남아 있다.

## 검수 정정과 한계

expanded09의 본넷 상단을 이전 마스크가 배경으로 포함한 것은 이전 AI 검수의 오류다. 상세크롭 `hood_check.jpg`에서 확인했고 센서/q와 무관하게 제외했다. 정정 전 실험은 삭제하지 않고 `corrected/`와 `masks_corrected/`로 대체 판정을 남겼다. 기존 static_background 결과를 그대로 운영/학습에 쓰면 안 된다.

확장 마스크168프레임을 검수했고, 대표24개 원천/목적지 프레임쌍을 추가로 보았다. 최종 교집합은 이 영역의 부분집합이다. 사람 검수, 모든 대응점의 물리적 정지 여부, 자동 분할 일반화, 독립 평가를 주장하지 않는다. 모든 자료는 반복 노출된 개발 진단 자료다.

## 재사용 가능한 산출물

- `validated/manifest.json`: 전체160쌍의 원본 출처·시각·좌표 배열 경로·통과 여부·실패 사유.
- `validated/sufficient_pairs.json`: 수·분포를 확보한24쌍. **가감속 학습 준비 완료 목록이 아니다.**
- `validated/PnP_valid_pairs.json`: 기존 기하 검사까지 통과한22쌍. 원래 구간 품질/깊이 안정성을 대체하지 않는다.
- `validated/*.npz`: 최종 대응점 좌표. `validated/results.json`: 품질 검사와 미채택 진단값.
- `mask_annotations.json`, `masks_corrected/`, `mask_review/`, `visual_review.json`, `hood_check.jpg`: AI 검수 및 정정 근거.
- 각 대조의 freeze·results·log: 설정·원인 비교·exit상태. `validated/checks.json`:160쌍 양끝마스크·추적기합의·점 간격·보호해시 검사 통과.

실행 파일: `run.py grid4_oldmask`, `expand_masks.py`, `run.py grid4_expanded`, `adaptive.py`, `source_pair.py`, `hood_correction.py`, `decouple_depth.py`, `validate_supply.py`, `final_report.py`. 기존 Mac 실험 가상환경 사용. 완료 결과 덮어쓰기를 피하려면 별도 새 폴더에서 재현한다.

## 결론

사용자가 요청한 **전체 병목을 해결할 만큼 충분한 배경점 확보**는 완료하지 못했다. 한 구간의 가시16쌍에서만 수·분포 병목을 해소했다. 자동 배경 분할·분류기 학습·제출로 확대할 근거는 아직 없다. 확인된 개선은 대응점 확보와 원인 분리이며 점수 개선이 아니다.

같은 저해상도 영상에서 격자/임계값을 계속 바꾸는 탐색은 중단한다. 남은 대응 위치 불일치와 구조 부족을 해결하려면 추가 영상 정보(원본 해상도에서 식별되는 정지 구조 또는 여러 프레임에 걸친 같은 점의 위치 제약)를 검증해야 한다. 그 방법이 성공할지는 현재 확인되지 않았다. 이번에는 그 검증이나 새 모델 다운로드까지 확장하지 않았다.
'''
(O/'REPORT.md').write_text(report)
write(O/'exit_status.json',{n:0 for n in ['grid4_oldmask','expand_masks','grid4_expanded','adaptive','source_pair','hood_correction','depth_separated','validate_supply','evidence','final_report']})
print(json.dumps(summary,ensure_ascii=False,indent=2))
