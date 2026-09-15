# MM-AU 공식 충돌 주석 및 소량 취득 가능성

조사일: 2026-09-14. 기존 제출·모델·추론 코드를 수정하지 않았다. GPU·모델 예측·신규 영상 주석 생성은 하지 않았다.

**판정: 공식 collision-start 프레임 주석을 이용한 외부 프레임 단위 감사 후보로는 조건부 viable이다. 현재 공식 배포만으로 대회의 ±0.3초 정확도를 계산하는 경로는 확보하지 못했다.** 시간 자료가 없는 이유는 추측이 아니라 저장소 관리자의 명시적 답변이다. 자료 전체를 배제하지 않고 실제 소량 시퀀스와 기존 사람 주석을 확보했다. 독립성·ego 접촉 여부를 아직 확인하지 않았으므로 검증용 확정 cohort에는 등록하지 않았다.

## 확인된 주석 근거

CVPR 2024 논문 §3은 사고 시간창을 5명의 자원자가 주석하고 최종 프레임 인덱스를 평균으로 결정한다고 설명한다. t_ai/t_ae는 사고창의 시작/끝이고, t_co는 충돌 시작이다. 이는 사고 위험이 보이기 시작하는 시점과 충돌을 분리한 **기존 사람 주석**이다. 5명 평균이라는 이유로 주석 자체를 무효로 취급하지 않는다. 다만 최초의 물리적 접촉을 판정하는 세부 매뉴얼, 검수자 간 편차, 가려진 접촉의 조정 규칙은 확인하지 못했다. 논문은 CCD/A3D/DoTA/DADA-2000와 영상 플랫폼 자료를 재수집했다고 명시하므로 데이터셋 이름이 달라도 원천 중복이 가능하다. [공식 CVPR 논문 §3](https://openaccess.thecvf.com/content/CVPR2024/papers/Fang_Abductive_Ego-View_Accident_Video_Understanding_for_Safe_Driving_Perception_CVPR_2024_paper.pdf)

공식 GitHub의 `video_metadata.json`과 README도 t_co를 collision start frame으로 설명한다. weather/light/scenes/type, t_ai/t_co/t_ae, total_frames 등이 있으며 FPS/PTS/영상별 시간표 필드는 없다. 현재 고정 revision의 JSON은 11,730개 레코드이며 논문 소개의 11,727과 다르므로 그대로 동일 규모라고 쓰지 않는다. 현재 공개 예제의 ego 차량 차로 진입·LEFT/RIGHT·회피공간과 일치하는 정답 필드는 없다. [공식 GitHub](https://github.com/jeffreychou777/LOTVS-MM-AU/tree/2b205a48d81f78b0c09d387d04a333aca9fc5949), [공식 원본 주석](https://github.com/jeffreychou777/LOTVS-MM-AU/blob/2b205a48d81f78b0c09d387d04a333aca9fc5949/video_metadata.json)

## 시간 변환의 구체적인 미확인

2025-06-08 공식 HF의 FPS 질문에 저장소 owner JeffreyChou는 수집 영상마다 원래 FPS가 다르며, 원본 영상이 현재 없고, 자신은 이미지 변환 과정에 참여하지 않아 원본 영상이나 FPS를 제공할 수 없다고 답했다. 일정 FPS를 임의 지정해 자연스럽게 재생하라는 제안도 있으나 이는 원래 시간축의 근거가 아니다. **원본 이미지 시퀀스라도 공식 샘플링률/번호 대응이 입증되면 시간 평가에 사용할 수 있지만 이 배포의 해당 근거는 확인되지 않았다.** 원본 PTS를 요구한다는 이유만으로 배제하는 것이 아니다. [공식 owner 답변](https://huggingface.co/datasets/JeffreyChou/MM-AU/discussions/1)

따라서 지금 확보한 자료로 가능한 값은 원본 프레임 번호의 충돌 위치 오차와 순서 기반 후보 포함 여부다. ±3/±9프레임을 임의로 ±0.3초라고 부르지 않는다. 향후 실제 원천 동영상 또는 공식 영상별 sampling map을 찾아 대응시키면 충돌-only 시간 검증 경로를 다시 열 수 있다. 프레임 단위 감사도 t_co의 번호 기준(0/1 시작 여부 및 추출과의 관계)이 확인되어야 정확한 오차로 해석할 수 있다.

## 라이선스

논문 공식 GitHub에서 연결하는 HF 데이터 카드에는 `cc-by-nc-4.0`이 명시되어 있다. GitHub README는 academic use만 무료라고 표현한다. 따라서 데이터 사용 조건을 코드 라이선스로 추정할 필요는 없다. 다만 두 문구가 있다고 상금 대회 이용·재배포까지 허용된다고 단정하지 않는다. 현재 취득본은 자료 확보 가능성 및 비상업적 연구 검토용이며 제출 ZIP에 포함하지 않았다. [고정 HF 데이터 카드](https://huggingface.co/datasets/JeffreyChou/MM-AU/blob/540cb1277cb70e91a7022abe852decb3ee9adb0a/README.md)

CC BY-NC 4.0은 출처·라이선스·변경 표시와 비상업적 이용 조건을 둔다. 원천 영상의 기타 권리까지 보증하지 않는다. 학술 연구용 충돌 감사 후보라는 판단과 대회 사용 권리의 최종 판단을 구분한다. [Creative Commons 공식 조건](https://creativecommons.org/licenses/by-nc/4.0/)

## 1GB 이내 취득: 실제 성공

공식 HF tree 조회 결과 전체 265개 파일 합계는 525,823,428,143 bytes이다. 원본 CAP/DADA는 2GiB 조각으로 나뉜 gzip tar 스트림이며, 개별 파일을 임의 위치에서 바로 꺼낼 수 있는 독립 ZIP 조각이 아니다. 그러나 **첫 gzip 조각 앞부분을 순서대로 해제해 초반의 완전한 영상 폴더를 취득하는 방식**은 실제 성공했다. 영상 내용·모델 출력에 따른 선택 없이 압축 순서 첫 source만 선택했다. 이것은 무작위·대표 표본 추출이 아니다. [공식 배포 구조](https://huggingface.co/datasets/JeffreyChou/MM-AU/tree/540cb1277cb70e91a7022abe852decb3ee9adb0a)

| 확인 항목 | 실제 결과 |
|---|---|
| HF revision | `540cb1277cb70e91a7022abe852decb3ee9adb0a` |
| GitHub revision | `2b205a48d81f78b0c09d387d04a333aca9fc5949` |
| 원본 조각 | `CAP-DATA_chunks/1-10/1-10.part_aa` |
| 첫 HTTP 요청 | 206, bytes 0–16777215/2147483648 |
| 추가 HTTP 요청 | 206, bytes 16777216–67108863/2147483648 |
| 취득한 압축 prefix | 총 67,108,864 bytes (64MiB) |
| metadata+prefix 측정 body 합계 | 77,077,921 bytes, 별도 최초 tree 조회는 약 0.12MB 이하; 합계 1GB 미만 |
| 완전 추출된 영상 폴더 | `CAP-DATA/1-10/8/009509/images` |
| 원본 이미지 번호 | 000001.jpg … 000223.jpg, 총 223장, 빠진 번호 없음 |
| 보존 위치 | `research/v6_stage2/mmau_probe/cap_8_009509/images/` |
| 기존 주석 연결 | video_hashcode=9ca724b1, video_name=8_9509, total_frames=223, t_co=90 |
| 파일 검증 | 각 파일 길이·SHA 기록, JPEG 223장 PIL verify 통과, 다음 source 경계 확인 |

`mmau_hf_tree.json`, `mmau_probe/probe_report.json`, `mmau_probe/extraction_report.json`에 조회·요청·파일 해시·범위를 보존했다. 코드 `mmau_probe.py`, `mmau_extract_one.py`는 206 응답과 정확한 최대 전송량을 확인하고 초과 시 실패한다. 전체 gzip 다운로드나 임의 tar 경로 추출을 하지 않았다. 기존 영상의 새 GT를 만든 것이 아니라 공식 metadata 한 행을 `official_annotation.json`으로 복사했다.

## 남은 조건과 다음에 가능한 최소 작업

1. **시간:** 현재는 공식 sampling map/원본 영상이 없으므로 초 단위 collision-only 점수는 계산하지 않는다. 확인되지 않은 FPS를 GT schema에 넣지 않는다.
2. **동일 사고 중복:** 수집 출처가 겹치므로 CAP 이름만으로 기존 공개 5개/DADA/Nexar와 독립이라고 선언할 수 없다. 현재 source가 내용상 새 사고인지는 확인하지 않았다. 모델 추론 전에 원천 정보·프레임 지문·검수로 확인해야 한다. 이 시점에서는 `independence_verified=false`다.
3. **대상:** ego-view 촬영과 ego 차량의 실제 충돌은 다르다. 현재 주석에는 명시적 ego_contact boolean이 없다. 확보 행의 texts와 causes는 각각 motorbike/truck을 언급하므로 대상 검토 없이 대회의 자동차 충돌 상대라고 확정하지 않는다.
4. **활용:** 신규 사람이 충돌 시점을 다시 찍지 않아도 기존 사람 t_co를 이용한 충돌 위치 감사 경로는 남아 있다. 먼저 공식 번호 기준과 대상/중복을 확인하고, 시간 근거가 복원되면 기존 고정 모델의 collision-only 독립 검증을 제안할 수 있다. 나머지 세 항목 및 전체 S2는 계속 null이다. 이번 조사에서는 예측을 실행하지 않았다.
