# V5 새 사고 영상 검증 자료: DADA

2026-09-13. 모델 예측을 읽기 전에 공식 주석으로 원천 선택 규칙과 목록을 동결했다. 6개 차량 관련 범주에서 해시 순서로 각각 2개, 총 12개를 우선 취득한다. 각 범주의 예비 2개도 미리 고정했다. 파일 목록만으로 자차 관여와 최초 물리 접촉을 확정하지 않으며, 영상 확인 전에는 적격 검증 표본 수를 확정하지 않는다.

원 출처는 [저자 DADA 저장소](https://github.com/JWFangit/LOTVS-DADA)와 그 저장소가 직접 연결한 [Google Drive 배포 폴더](https://drive.google.com/drive/folders/1l1_xOMWfs2eSoh0771ZJOcS2tcKwhh-C)다. 저장소의 Benchmark download 절은 이 benchmark를 공개하며 “sincerely invite to use and share it”라고 명시한다. 이 배포자의 명시적 사용·공유 안내에 근거해 비영리 모델 연구의 로컬 검증에 사용한다. 별도 표준 데이터 라이선스의 이름이나 제3자 영상 전체에 대한 권리보증 문구는 확인하지 못했다. MIT 코드 라이선스나 논문 저작권을 영상의 허락으로 대체하지 않는다. 원 RGB/주석은 제출 ZIP에 포함하지 않는다.

[저자 논문](https://arxiv.org/html/1904.12634)은 DADA 영상의 30fps 시간 기준과 자차 관여/비관여 구분, 사고 시간 구간을 설명한다. 이번 공식 XLSX에는 abnormal start, accident frame, abnormal end, total frames가 별도로 있다. accident frame을 대회 최초 물리 접촉과 자동 동일시하지 않으며, 원본 번호와 시각적 의미를 확인한다. 진입·방향·회피 공간의 대회 정답은 이 주석에서 얻었다고 주장하지 않는다.

전체 대용량 영상 묶음 대신 ZIP64 중앙 목록 482,459,595 bytes와 작은 XLSX 257,393 bytes를 취득해 파일 위치를 확인했다. 선택 RGB만 원 ZIP 볼륨의 Range 요청으로 읽고, 각 파일의 중앙/로컬 헤더·원본 경로·크기·CRC32를 대조한다. 다운로드 범위와 SHA256은 별도 JSON에 보존한다. 원래 프레임 번호와 픽셀을 유지하며 실제 추론에는 주석을 넘기지 않는다.

같은 온라인 원천의 재편집 영상이 있을 수 있고 원 채널 ID는 제공되지 않았다. 공개 5개와 영상 유사도를 점검하더라도 모든 중복이 없다고 보장할 수 없다. 새로운 clip 12개를 독립 카메라/운전자 12개라고 부르지 않는다. 예측 미열람 시각 검토는 AI 관찰로 표시하고, 불명확한 항목을 정답으로 강제하지 않는다. 검증 결과에 따라 표본을 골라내거나 제출 코드에 ID별 답을 넣지 않는다.

취득 코드: `scripts/probe_v5_dada.py`, `fetch_v5_dada_directory.py`, `select_v5_dada.py`, `extract_v5_dada.py`. 선택 동결 SHA256: `c615cc166cfcf7eaf04ef0734e72797742945f5b2f3874abea46ea1b0837221f`.
