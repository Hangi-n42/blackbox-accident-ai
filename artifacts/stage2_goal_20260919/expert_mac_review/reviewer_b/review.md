# Stage2 독립 AI 전문가 시각 검수 B

검수 대상은 00024~00029 원본 MP4와 그 PTS 매핑이다. 다른 검수자 기록, 기존 AI screening/REPORT, 인간 초안, 모델 예측/점수/후보/이벤트 라벨은 읽지 않았고 모델 추론도 실행하지 않았다. `human_review=false`, `type=ai_expert_independent_review`이다.

전체 2Hz 공유시트 444개 표본을 실제 이미지 도구로 판독했고, 각 영상의 근접·진입 검토 구간은 추가 원본 프레임으로 판독했다. 전체 원본을 연속 재생하거나 오디오를 청취하지 않았다. 모든 원본의 SHA256 및 순차 디코드 PTS 목록이 제공된 PTS JSON과 일치한다. 프레임 번호는 0-based이며 00028은 time_base=1/19584, 나머지는 1/15360이다.

**결론: 여섯 영상 모두 이 시각 증거만으로 실제 자차 접촉을 확정하지 않았다. 이는 비충돌 판정이 아니다.** 따라서 최초 접촉 프레임과 실제 접촉 당시 공간 0/1, 실제 사고 상대를 전제로 하는 확정 대회 정답은 미상으로 보존한다. 아래 근접 검수 범위는 접촉의 불확실성 구간으로 사용할 수 없다.

| 원본 | 추적 대상 | 조건부 진입 | 실제 접촉 / space |
|---|---|---|---|
| 00024 | 노란색 박스형 택시 | 미상 | 미상 / 미상 |
| 00025 | 검은색 세단 | 미상 | 미상 / 미상 |
| 00026 | 파란색 Prime 표기가 있는 대형 트레일러 | 미상 | 미상 / 미상 |
| 00027 | 전방의 짙은 청색 승용차 | 미상 | 미상 / 미상 |
| 00028 | 첫 프레임부터 전방에 있는 짙은색 Nissan SUV | 규칙상 f0 / 물리 진입은 시작 전 미상 | 미상 / 미상 |
| 00029 | 우측 차로에서 접근하는 은색 소형 해치백/MPV | 추론 구간 f570~585, RIGHT (상대 확정 전 조건부) | 미상 / 미상 |

## 00024

추적 차량: 노란색 박스형 택시. f0의 왼쪽 인접 차량을 f570~660까지 차체 형상·색·창문·후미등의 연속성으로 추적했다. 첫 화면 전방의 청색 SUV와는 다른 차량이다.

사고 상대 확정 여부: **미확정**. f600~627에서 택시 후면/우측 차체와 자차 후드가 화면상 가까워지나 실제 양 차량 접촉면은 후드 아래에 가려진다. 택시 변형·접촉 지점은 보이지 않는다. 그 뒤 택시는 계속 주행한다. f945~990의 흰색 세단 우측 근접도 별도로 확인했으나 접촉면은 보이지 않는다. 이 두 근접을 실제 사고 상대 확정으로 사용하지 않았다.

원본 연속 정밀검수 중심 범위: f600~f627, native PTS 307200~321024, 20.000000~20.900000초. 이 범위는 실제 충돌 시각의 bounds가 아니다.

진입: f570,600,615에서 택시 우측 바퀴는 화면에 보이는 점선 좌측에 남아 있다. 교차로에서 자차 방향이 바뀌어 상대 바퀴가 자차 차로 경계에 최초로 닿는 순간을 특정할 수 없다. 화면상 접근과 차로 진입을 구분했다.

방향: 주요 추적 택시는 화면 왼쪽에서 보인다. 그러나 자차 차로 진입 자체와 사고 상대가 확정되지 않아 대회 LEFT 정답으로 승격하지 않았다.

공간: 실제 접촉 시각이 미확정이므로 접촉 당시 진행·회피 공간 0/1은 미상이다.

직접 본 범위: 전체 약 2Hz 시트와 다음 원본 프레임 세트. 합집합 135프레임.
- f570~f660, step=3: [00024_0570_0660_s3](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_0570_0660_s3/manifest.json)
- f600~f627, step=1: [00024_0600_0627_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_0600_0627_s1/manifest.json)
- f945~f990, step=3: [00024_0945_0990_s3](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_0945_0990_s3/manifest.json)

볼 수 없던 정보: 후드 아래 자차 전방 범퍼와 택시 접촉면, 가림 구간에서의 실제 두 차량 간 거리, 오디오 및 측후방 영상, 교차로 통과 중 바퀴와 자차 차로 경계의 최초 접점.

대표 근거 이미지:
- [00024_f0000.png (f0, PTS 0, 0.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_f0000.png)
- [f0570.png (f570, PTS 291840, 19.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_0570_0660_s3/f0570.png)
- [f0600.png (f600, PTS 307200, 20.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_0570_0660_s3/f0600.png)
- [f0615.png (f615, PTS 314880, 20.500000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_0570_0660_s3/f0615.png)
- [sheet_01.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_0600_0627_s1/sheet_01.jpg)
- [sheet_00.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00024_0945_0990_s3/sheet_00.jpg)

## 00025

추적 차량: 검은색 세단. 첫 화면 왼쪽 인접 차로의 세단을 f510~645까지 창문 윤곽·차체·후미등을 통해 추적했다. f597~620에서 자차 전방 왼쪽 차체가 근접하고 f624 이후 앞을 가로질러 진행한다.

사고 상대 확정 여부: **미확정**. f597~620에서 세단 우측 전면/측면이 후드 왼쪽 아래와 겹쳐 보이고 자차 시야가 우측 보도·교통섬으로 돌아간다. 실제 범퍼/차체 접촉점은 화면 하단에 가려진다. 보행자 앞 정지와 급격한 시야 변화만으로 접촉을 확정하지 않았다. 세단은 이후 전방으로 떠난다.

원본 연속 정밀검수 중심 범위: f597~f620, native PTS 305664~317440, 19.900000~20.666667초. 이 범위는 실제 충돌 시각의 bounds가 아니다.

진입: f510~549 연속 원본에서 우측 바퀴와 왼쪽 차로 경계 근처를 확인했다. 교차로 진입 이후 차로 표시가 끊기고 자차 회전과 와이퍼/후드 가림이 겹친다. 첫 바퀴 접점을 확인할 수 없고 보수적인 물리 진입 구간도 확정하지 않았다.

방향: 추적 세단은 화면 왼쪽에서 자차 앞쪽으로 이동한다. 조건부 화면상 출발 방향만 LEFT이며 실제 사고 상대의 대회 방향은 미상이다.

공간: 실제 접촉 시각이 미확정이므로 접촉 당시 진행·회피 공간 0/1은 미상이다.

직접 본 범위: 전체 약 2Hz 시트와 다음 원본 프레임 세트. 합집합 160프레임.
- f510~f549, step=1: [00025_0510_0549_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0510_0549_s1/manifest.json)
- f540~f645, step=3: [00025_0540_0645_s3](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0540_0645_s3/manifest.json)
- f597~f620, step=1: [00025_0597_0620_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0597_0620_s1/manifest.json)

볼 수 없던 정보: 후드 아래 세단 바퀴 및 자차 전방/좌측 접촉면, 끊긴 교차로 차선 연장선의 정확한 위치, 자차와 세단의 실제 거리, 오디오 및 측후방 영상.

대표 근거 이미지:
- [00025_f0000.png (f0, PTS 0, 0.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_f0000.png)
- [f0510.png (f510, PTS 261120, 17.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0510_0549_s1/f0510.png)
- [f0529.png (f529, PTS 270848, 17.633333s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0510_0549_s1/f0529.png)
- [f0535.png (f535, PTS 273920, 17.833333s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0510_0549_s1/f0535.png)
- [f0570.png (f570, PTS 291840, 19.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0540_0645_s3/f0570.png)
- [f0609.png (f609, PTS 311808, 20.300000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0540_0645_s3/f0609.png)
- [sheet_01.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00025_0597_0620_s1/sheet_01.jpg)

## 00026

추적 차량: 파란색 Prime 표기가 있는 대형 트레일러. f300~465에서 후면에서 좌측 측면으로 이어지는 같은 트레일러를 추적할 수 있다. 앞서 보이는 흰색 트럭과는 구분한다.

사고 상대 확정 여부: **미확정**. 처음부터 카메라가 하늘을 향하고 하단 넓은 영역은 블러 처리되어 있다. f340 이후 파란 트레일러 측면이 화면 오른쪽을 크게 차지하지만, 도로·바퀴·자차 차체 접점은 보이지 않는다. 트럭 측면의 확대와 카메라 움직임은 접촉의 직접 증거가 아니다.

원본 연속 정밀검수 중심 범위: f360~f375, native PTS 184320~192000, 12.000000~12.500000초. 이 범위는 실제 충돌 시각의 bounds가 아니다.

진입: 전체 0~539 표본에서 상대 바퀴와 자차 차로 경계를 동시에 판독할 수 없다. 측면이 오른쪽에서 나타난 사실을 바퀴의 차선 최초 접촉으로 대체할 수 없다.

방향: 파란 트레일러는 화면 오른쪽에 보인다. 차로 경계가 보이지 않아 진입 방향 RIGHT로 확정하지 않았다.

공간: 실제 접촉 시각이 미확정이므로 접촉 당시 진행·회피 공간 0/1은 미상이다.

직접 본 범위: 전체 약 2Hz 시트와 다음 원본 프레임 세트. 합집합 71프레임.
- f300~f465, step=5: [00026_0300_0465_s5](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00026_0300_0465_s5/manifest.json)
- f360~f375, step=1: [00026_0360_0375_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00026_0360_0375_s1/manifest.json)

볼 수 없던 정보: 도로와 자차 차로 경계, 트레일러 바퀴, 실제 접촉점, 진행 및 회피 공간, 하단 블러 내부 내용, 오디오 및 측후방 영상.

대표 근거 이미지:
- [00026_f0000.png (f0, PTS 0, 0.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00026_f0000.png)
- [sheet_00.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00026_0300_0465_s5/sheet_00.jpg)
- [sheet_02.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00026_0300_0465_s5/sheet_02.jpg)
- [sheet_00.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00026_0360_0375_s1/sheet_00.jpg)

## 00027

추적 차량: 전방의 짙은 청색 승용차. f0 오른쪽에 보이는 차량과 f60 이후 전방 차량의 연결은 어둠·차량 중첩 때문에 확정하지 않았다. f60~1199의 전방 차체와 후미등은 일관되게 추적할 수 있다.

사고 상대 확정 여부: **미확정**. f573~596에서 앞차가 후드 가까이 확대되고 정지한다. f581~589 주변에도 실제 자차 범퍼와 앞차의 접촉점은 보이지 않는다. 이후 사람이 차량 부근으로 나오는 모습은 확인되지만 접촉의 직접 증거로 사용하지 않았다.

원본 연속 정밀검수 중심 범위: f573~f596, native PTS 293376~305152, 19.100000~19.866667초. 이 범위는 실제 충돌 시각의 bounds가 아니다.

진입: 첫 화면에서 자차는 회전 중이고 대상 동일성/차로 경계가 불분명하다. f60부터 전방 같은 통행 경로에 있는 사실은 확인되지만 최초 바퀴 진입은 보이지 않는다. 첫 이미지부터 차로 안이라는 조건이 입증되지 않아 competition entry=0을 부여하지 않았다.

방향: 정밀검수 시점에는 이미 전방이다. 화면 좌우에서의 최초 진입은 관찰되지 않는다.

공간: 실제 접촉 시각이 미확정이므로 접촉 당시 진행·회피 공간 0/1은 미상이다.

직접 본 범위: 전체 약 2Hz 시트와 다음 원본 프레임 세트. 합집합 125프레임.
- f0~f120, step=10: [00027_0000_0120_s10](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00027_0000_0120_s10/manifest.json)
- f540~f615, step=3: [00027_0540_0615_s3](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00027_0540_0615_s3/manifest.json)
- f573~f596, step=1: [00027_0573_0596_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00027_0573_0596_s1/manifest.json)

볼 수 없던 정보: 처음 f0~60 대상 차량 동일성의 확실한 연결, 최초 진입 당시 차로 경계 및 바퀴, 후드 아래 실제 접촉면, 야간 측방 빈 공간의 전체 폭, 오디오 및 측후방 영상.

대표 근거 이미지:
- [00027_f0000.png (f0, PTS 0, 0.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00027_f0000.png)
- [sheet_00.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00027_0000_0120_s10/sheet_00.jpg)
- [sheet_01.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00027_0000_0120_s10/sheet_01.jpg)
- [f0582.png (f582, PTS 297984, 19.400000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00027_0540_0615_s3/f0582.png)
- [sheet_01.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00027_0573_0596_s1/sheet_01.jpg)

## 00028

추적 차량: 첫 프레임부터 전방에 있는 짙은색 Nissan SUV. f0~1224 전체 시간축에서 같은 후면 차체·엠블럼·등화의 연속성을 확인했다.

사고 상대 확정 여부: **미확정**. f585~613에서 앞차가 가까워지고 f614~621에서 시야와 앞차 위치가 변한다. 그러나 실제 자차 범퍼/앞차 접촉면은 하단 가림 아래에 있다. 근접·등화 변화·시야 흔들림만으로 실제 접촉 또는 최초 접촉 프레임을 확정하지 않았다.

원본 연속 정밀검수 중심 범위: f606~f621, native PTS 387840~397440, 19.803922~20.294118초. 이 범위는 실제 충돌 시각의 bounds가 아니다.

진입: 동일 추적 SUV가 f0에서 노란 중앙선 오른쪽과 우측 연석 사이의 자차 통행 차로 안에 이미 있다. 이 차량이 실제 사고 상대라고 추가 확정될 경우 대회 규칙값은 첫 원본번호 0이다. 물리적 진입은 영상 시작 전 미상이며 0초 진입을 뜻하지 않는다.

방향: 첫 화면부터 이미 전방 차로 안이어서 LEFT/RIGHT 물리 진입 방향은 관찰되지 않는다.

공간: 실제 접촉 시각이 미확정이므로 접촉 당시 진행·회피 공간 0/1은 미상이다.

직접 본 범위: 전체 약 2Hz 시트와 다음 원본 프레임 세트. 합집합 113프레임.
- f585~f635, step=2: [00028_0585_0635_s2](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00028_0585_0635_s2/manifest.json)
- f606~f621, step=1: [00028_0606_0621_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00028_0606_0621_s1/manifest.json)

볼 수 없던 정보: 영상 시작 전 실제 진입 시각/방향, 하단 자차 전방 범퍼와 접촉점, 실제 접촉 순간, 접촉 순간을 전제로 하는 회피 공간, 오디오 및 측후방 영상.

대표 근거 이미지:
- [00028_f0000.png (f0, PTS 0, 0.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00028_f0000.png)
- [f0611.png (f611, PTS 391040, 19.967320s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00028_0585_0635_s2/f0611.png)
- [sheet_00.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00028_0606_0621_s1/sheet_00.jpg)
- [sheet_01.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00028_0606_0621_s1/sheet_01.jpg)

## 00029

추적 차량: 우측 차로에서 접근하는 은색 소형 해치백/MPV. f480~645에서 후면 램프·검은 뒤창·차체를 연속 추적했다. 시작 프레임의 회색 세단 및 좌측의 흰색 세단과 구분했다.

사고 상대 확정 여부: **미확정**. f597~620에서 은색 차량 좌측면이 화면 오른쪽을 크게 차지하고 왼쪽에 흰색 세단이 병행한다. 자차 측면과 은색 차량의 실제 접촉점은 우측/하단 시야 밖이다. 차체 변형이나 두 차체 접촉을 직접 볼 수 없고 이후 두 차량은 계속 주행한다.

원본 연속 정밀검수 중심 범위: f597~f620, native PTS 305664~317440, 19.900000~20.666667초. 이 범위는 실제 충돌 시각의 bounds가 아니다.

진입: f570에서 은색 차량의 좌측 바퀴는 우측 인접 차로 쪽에 남아 있다. f585에서는 같은 바퀴가 보이는 점선의 연장선보다 자차 차로 쪽에 들어와 있다. 그 사이 점선 공백·차량 가림·자차 방향 변화 때문에 최초 접점을 한 프레임으로 정할 수 없어 조건부 진입 구간을 f570~585로 보존했다.

방향: 추적 은색 차량은 화면 오른쪽 인접 차로에서 접근한다. 조건부 candidate entry 방향은 RIGHT이며 실제 사고 상대의 대회 정답은 접촉 미확정으로 보류한다.

공간: 실제 접촉 시각이 미확정이므로 접촉 당시 진행·회피 공간 0/1은 미상이다.

직접 본 범위: 전체 약 2Hz 시트와 다음 원본 프레임 세트. 합집합 143프레임.
- f555~f645, step=3: [00029_0555_0645_s3](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0555_0645_s3/manifest.json)
- f563~f578, step=1: [00029_0563_0578_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0563_0578_s1/manifest.json)
- f578~f593, step=1: [00029_0578_0593_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0578_0593_s1/manifest.json)
- f597~f620, step=1: [00029_0597_0620_s1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0597_0620_s1/manifest.json)

볼 수 없던 정보: 우측/하단 시야 밖 자차 측면과 상대 차체 접점, 점선 공백 및 상대 하부 가림의 정확한 최초 바퀴 접점, 실제 접촉 시각, 자차 좌우 전체 측방 여유 폭, 오디오 및 측후방 영상.

대표 근거 이미지:
- [00029_f0000.png (f0, PTS 0, 0.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_f0000.png)
- [f0555.png (f555, PTS 284160, 18.500000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0555_0645_s3/f0555.png)
- [f0570.png (f570, PTS 291840, 19.000000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0555_0645_s3/f0570.png)
- [f0578.png (f578, PTS 295936, 19.266667s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0563_0578_s1/f0578.png)
- [f0585.png (f585, PTS 299520, 19.500000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0555_0645_s3/f0585.png)
- [f0603.png (f603, PTS 308736, 20.100000s)](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0555_0645_s3/f0603.png)
- [sheet_01.jpg](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/00029_0597_0620_s1/sheet_01.jpg)

## 사용 제한

여기에 기록한 조건부 차량 추적·진입 관찰을 충돌 확정 GT로 사용해서는 안 된다. 00028의 f0 규칙값은 같은 앞차가 실제 사고 상대라고 별도 확정될 때에만 적용된다. 00029의 f570~585 구간은 차선/바퀴 기하에 대한 보수적 추론이며 정확한 한 프레임이 아니다. 00024~00027의 진입은 미상이다. 원본 영상의 비가림 접촉 장면, 추가 측면 영상 또는 동기화된 접촉 증거가 없으면 정확도를 숫자로 강제하지 않는다.

모든 항목의 status, native PTS, 관찰 범위, 근거 경로는 [records.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/records.json)에 구조화했다.
