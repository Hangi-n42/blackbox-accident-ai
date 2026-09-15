# PSAD 후속 공개 메타데이터 확인

2026-09-15. 기존 `independent_data_options.md`에서 미확인으로 남은 공개 시간 표와 scene annotation 링크를 직접 확인했다. 영상·가중치 다운로드, 학습, 모델 예측은 하지 않았다.

공식 [시간 표 원문](https://raw.githubusercontent.com/Shun-Gan/PSAD-dataset/main/Temporal_annotations_for_PRT.csv)은 행 번호, Pre-crash scenario, Appearance, First glance, Anomaly start, Evasive maneuver, Collision, Anomaly end 열을 가진다. Collision은 실수값으로 제공되지만 이 표에는 영상 파일명·원본 URL·사고 상대 식별자가 없다. Pre-crash scenario 값을 원본 영상 ID로 임의 해석하거나 행 번호와 영상 파일 순서를 임의로 연결하지 않는다. 따라서 이 표만으로 알려진 원본의 물리 접촉 정답과 PTS를 연결하지 못했다.

[공식 저장소](https://github.com/Shun-Gan/PSAD-dataset)는 stimuli 영상 30fps와 object_labels의 10fps 번호를 별도로 설명한다. 이 규칙을 Collision 필드에도 그대로 적용할 근거는 아직 없다. [scene annotations 링크](https://drive.google.com/file/d/1fUTxLj-XqQJexKzlz6iJWYRIiw0awlPS/view?usp=sharing)는 웹 텍스트 도구에서 로그인 링크만 반환해 RAR 내용은 확인하지 못했다. 실제 다운로드 불가능이나 접근 금지가 입증됐다는 뜻은 아니다.

원천 [DoTA 저장소](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)는 배포 프레임을 10fps로 추출하는 명령과 anomaly_start/end를 설명한다. 원저자가 공유 권한을 확보했다는 안내는 있으나, 이를 PSAD의 별도 라벨 정의·이용조건·첫 물리 접촉 정답의 증명으로 확대하지 않는다.

이 후속 확인으로 새 시간 정답을 확보하지 못했다. 직접 이용하려면 실제 scene annotation의 영상 ID·Collision 단위와 원본 시퀀스 대응, 접촉 정의를 확인해야 한다. 임의 FPS·파일 순서·anomaly 시작으로 보충하지 않는다. 현재 V7 채택 조건과 점수 전망은 바뀌지 않았다.
