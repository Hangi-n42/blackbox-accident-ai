# Stage2 신규 소량 취득·AI 적격성 선별 결과

2026-09-19. **6개 원본 취득 완료, 정확 접촉 GT 0개, 사람 검수 0개, 새 독립 평가 인증 0개.** 영상 확보 성공을 정답 확보 또는 점수 개선으로 바꾸지 않는다.

## 취득과 확인

- 원천: Nexar 공식 공개 `nexar-ai/nexar_collision_prediction`, 고정 revision `aa97deda5a59f00bb7187739053b7c72e14374df`의 `train/positive/00024.mp4`부터 `00029.mp4`까지 6개.
- 원본 합계 **99,413,074바이트**, 전체 **6,572프레임**. 취득 전에 두 배치의 ID·기대 크기·배포SHA를 각각 동결했다. 첫 배치4개의 가시성 부족을 확인한 뒤 번호순 다음2개만 추가했다. 모델 예측을 보고 표본을 고르거나 교체하지 않았다.
- 배포 LFS SHA와 실제 원본 SHA 6/6 일치. 원본 전부 디코딩, 원시 PTS/time_base/연속 decoded index 보존, 누락·비증가 시각 0. 00028은30.6Hz, 나머지는30Hz다.
- 사전69개 숫자형 원천ID 제외. 현재 로컬96개 MP4의 SHA를 재계산해 신규6개와 대조: 동일바이트 중복0. 신규6개끼리도0. 재인코딩·다른 절단의 동일 사고 및 주행원천 독립성은 인증하지 않았다.
- 기존 Nexar 라이선스를 보존하고 같은 고정 revision의 LICENSE를 다시 받아 바이트 일치를 확인했다. 라이선스와 출처는 `NEXAR_LICENSE.txt`, `license_check.json`, 각 acquisition record에 있다.
- 다운로드·디코딩·SHA/PTS 계약 검사 실행 exit0. 모델 추론·학습·제출·외부 연락은0회. 새 파일은 이 acquisition 폴더 아래에만 작성했다.

## 시각 검수 범위와 결론

AI 한 명이 overview6장, 공급자 event 힌트 주변 시트6장, 원본크기PNG7장을 직접 판독했다. 전체6572프레임을 육안 관찰하거나 연속재생·오디오 검수한 것은 아니다. nativePNG는10장 생성했지만 직접 판독은7장이다.

|ID|관측 요약|정밀 평가 판정|
|---|---|---|
|00024|노란 택시가 좌전방으로 근접. 접촉 가능 부위는 후드·하단 경계 뒤.|접촉 여부/시점 unknown|
|00025|검은 승용차가 좌전방에 근접하고 자차 시야가 우측으로 이동. 접촉면 하단 가림.|접촉 여부/시점 unknown|
|00026|카메라가 하늘을 향해 도로·바퀴·접촉면이 대부분 보이지 않음.|대회 항목 주석 가시성 부족|
|00027|앞차와 근접, 우측 다른 차량 존재. 접촉 가능 부위가 후드·블러 뒤.|접촉 여부/시점 unknown|
|00028|시작부터 같은 Nissan 앞차가 차로 내부. 사건 주변 근접·시야변화는 있으나 접촉면 가림.|접촉 여부/시점 unknown; 시작부터 진입은 사고상대 확정 전 조건부|
|00029|우측 Chevrolet이 근접. 아래 접촉면은 블러·화면 밖.|접촉 여부/시점 unknown|

`ai_screening.json`은 AI 관측이다. 공급자 event 시점을 최초접촉 정답으로 복사하지 않았고 unknown을 비충돌/0/LEFT로 채우지 않았다. 네 출력의 정확 GT는 전부 미상이다. 00028의 물리 진입시점은 시작이전 불관측이고, 대회 첫프레임 규약은 실제 사고상대가 확인될 때만 적용 가능한 조건부 정보로 분리했다.

## 다음 가능한 한 단계

`review_packet.md`의 모델출력 없는 원본·overview·event·native PTS와 `human_review_templates/`의 빈 양식6개를 사람 검수자에게 제공할 수 있다. 양식은 `unfilled_human_review_template`이며 실제 사람검수로 집계되지 않는다. 기존 AI 판독문을 보지 않고 접촉 관측 가능성을 독립 확인할 수 있지만, 원본에서도 가려진 시점을 강제로 확정해서는 안 된다. 현 자료의 정확 성능 비교는 보류한다.

추가 다운로드는100MB 상한에 근접해 중단했다. 다른 경로도 재확인했다. [CCD 공식 저장소](https://github.com/Cogito2012/CarCrashDataset)는 YouTube 유래10FPS 클립·ego 관여표시·시간주석을 설명하지만 이번 확인으로 데이터 자체의 별도 권리범위와 소량 정밀GT 적격성을 확정하지 못했다. [MM-AU 공식 owner 답변](https://huggingface.co/datasets/JeffreyChou/MM-AU/discussions/1)은 원영상/FPS를 제공할 수 없다고 현재도 명시한다. 두 경로를 새 ±0.3초 검증자료 확보 성공으로 기록하지 않았고 추가 영상은 받지 않았다.

## 재현과 주요 파일

- `selection.json`, `selection_round2.json`: 예측 미노출 원천선정 동결.
- `acquisition.json`, `acquisition_round2.json`: 실제 취득·전체 디코딩 기록.
- `integrity.json`, `prior_byte_inventory.json`: SHA·원본 PTS·중복 검사 결과.
- `ai_screening.json`: 정답 미확보·조건부 관측·검수 범위.
- `review_packet.md`, `human_review_templates/`: 사용자 사람검수용 묶음.
- `acquire.py`, `acquire_round2.py`: 신규 취득 재현 코드. 원본이 이미 존재하면 중단하며 덮어쓰지 않는다.
- `verify_intake.py`: 기존 산출물 재검사. 프로젝트 Python환경에서 실행 가능하다.
