# Spatial coordinate reference B

Exact supplied 1152x768 canvas. Origin (0,0) top-left, x right, y down. No source/native coordinate conversion is needed for these coordinates.

tolerance_px is a conservative manual image-space uncertainty radius/normal-to-line width, not a statistical confidence interval. Null means unsupported; absence of a coordinate is not evidence of absence.

Prior case states and entry interval are known; new spatial predictions and peer coordinates were not read. This is an AI reference, not human-certified ground truth.

| Frame | Vehicle bbox | Supported wheel contact points | Counterpart-side boundary |
|---|---|---|---|
| 31 | [650, 449, 746, 513] | rear [680, 505] +/-7 | short observed dash + uncertain extension |
| 39 | [625, 431, 836, 541] | front [640, 520] +/-7, rear [720, 536] +/-6 | UNKNOWN / occluded; null polyline |

f31: 관찰한 점선과 제한된 연장은 보이는 가까운 뒷바퀴가 바깥쪽에 있는 관계와 양립한다. 앞바퀴와 반대편 바퀴 접지점은 식별하지 못했으므로 '모든 바퀴가 바깥'을 완전한 좌표 측정으로 독립 재증명하지는 못한다.

자차 오른쪽의 흰 점선 도색 띠 중심을 대표하는 polyline. 관찰점은 가까운 한 도색 구간에서 직접 지정. 두 위쪽 점은 그 구간 방향의 제한적 연장으로 관찰점이 아님. 타이어-띠 접촉 판정 시 도색 반폭과 tolerance를 무시하면 안 됨.

- SUV 상부 황색 마스크는 접지점을 덮지 않지만 원본 압축과 작은 타이어 때문에 좌표 정확도 제한.
- 경계는 짧은 한 점선 구간에 기반한다. 긴 연장 구간에는 큰 오차를 부여했으며 차로 폭/과거 상태로 선을 맞추지 않았다.
- bbox는 보이는 외곽에 대한 근사 박스이고 픽셀 분할 정답이 아니다.

f39: 보이는 바퀴 좌표는 제공할 수 있으나 상대쪽 경계 polyline이 미확정이므로 현재 좌표 참조만으로 과거 INSIDE 판정을 검증할 수 없다. 기존 상태 참조는 변경하지 않고 기하 참조의 제한으로 분리한다.

자차 상대쪽 경계를 요구하지만 이 입력의 개별 픽셀 기하로 지원되는 연속 선은 미확정. 차체 그림자/검은 노면 자국을 도색 경계로 대체하지 않음.

- 안쪽 기준점은 방호벽/황색 가장자리 오른쪽의 자차 진행 통로에서 선택했지만 그 점 하나로 가려진 반대쪽 경계를 결정할 수 없다.
- 과거 INSIDE 또는 다른 프레임의 진입 구간을 이용해 가려진 경계나 접지점 좌표를 생성하지 않았다.
- 가려진 경계 좌표 null은 평가에서 미지원으로 취급해야 하며 오답 좌표나 경계 없음으로 해석하면 안 된다.

Exact input images directly viewed. Reviewer-only crops [570,420,970,630] saved without resize; a separate nearest-neighbor 3x copy with coordinate grid aided reading. Grid labels refer to original 1152x768 coordinates. These aids add no source information and are not model inputs.

No model calls. Existing model inputs and prior references unchanged. New review JSON/MD and reviewer-only image aids created; editing stopped after completion.
