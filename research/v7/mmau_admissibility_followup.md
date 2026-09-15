# MM-AU 보유 표본의 프레임 대응·대회 사용 적합성 후속 확인

2026-09-15. 기존 `independent_data_options.md`, `mmau_feasibility.md`, 실제 추출본과 공식 주석을 먼저 읽었다. 추가 확인은 연결된 원저자 CAP 데이터 로더·평가기와 기존 owner 답변·라이선스에 한정했다. 영상/모델 신규 다운로드, 추론, 학습, 주석 변경은 없다.

**결론: 보유 표본의 배포 시퀀스와 공식 주석 행은 연결되지만, 이를 정확한 자차 접촉의 초 단위 독립 검증으로 바로 사용할 수는 없다.** 막힌 조건은 영상 부족만이 아니다. `t_co`와 JPEG 번호의 0/1 기준, 원래 시간 대응, 해당 행의 실제 접촉 대상, 대회에서의 구체적 이용 범위가 남아 있다. 새로운 대용량 다운로드 없이 이 경계를 확인했다.

## 실제 보유 자료의 연결

|항목|확인 결과|
|---|---|
|배포 원천|공식 HF `JeffreyChou/MM-AU`, revision `540cb1277cb70e91a7022abe852decb3ee9adb0a`|
|압축 내부 시퀀스|`CAP-DATA/1-10/8/009509/images/`|
|보유본|`research/v6_stage2/mmau_probe/cap_8_009509/images/000001.jpg` … `000223.jpg`|
|주석 원천|공식 GitHub `jeffreychou777/LOTVS-MM-AU`, revision `2b205a48d81f78b0c09d387d04a333aca9fc5949`, `video_metadata.json`|
|행 식별|`video_hashcode=9ca724b1`, `video_name=8_9509`, `id=8281`, `type=8`|
|기존 공식 값|`total_frames=223`, `t_ai=20`, `t_co=90`, `t_ae=200`, `accident_frame=90`|
|이번 실제 무결성 검사|추출 manifest의 JPEG 223개 SHA를 전수 재계산: 불일치 0개|
|시간 필드|메타데이터 전체 레코드의 키 합집합에 FPS·PTS·원천 URL·시간표 없음|

자료 연결은 추출 경로의 type/일련번호 정규화와 유일한 `video_name`, 동일 프레임 수에 근거한다. **파일 존재와 행 연결의 증거이지, 모든 의미 주석이 정확하다는 증거는 아니다.** [공식 번호·필드 설명](https://github.com/jeffreychou777/LOTVS-MM-AU), [고정 배포 카드](https://huggingface.co/datasets/JeffreyChou/MM-AU/blob/540cb1277cb70e91a7022abe852decb3ee9adb0a/README.md)

재현용 로컬 해시:

- `video_metadata.json`: `493f9c95acb6f59f0b9717270707bde405c2d568e56face90395076e1f76dea8`
- `extraction_report.json`: `68cbd6d3785afb2282a83081596c695b90d29dc7ff43c76527db413d1cbd1f84`
- `cap_8_009509/official_annotation.json`: `ad2c231a9e001eb12e09aabe6d4a358815d9b9a4c1fb0b343eada38228c60700`

## 새로 확인한 원저자 코드가 해결하는 것과 해결하지 않는 것

공식 MM-AU에서 연결된 CAP 저장소의 실제 구현까지 확인했다. `src/dataset.py` 59–60행은 정렬된 이미지 목록을 `video_file[fid]`로 읽고, 88행은 설정 파일의 `toa`에서 clip start를 빼서 내부 인덱스를 만든다. 즉 **이 로더가 받는 별도 txt 설정의 시작/충돌 번호는 목록 인덱스로 동작**한다. 그러나 로더는 MM-AU의 `video_metadata.json`을 읽지 않는다. 보유 행 `9ca724b1`이 포함된 해당 txt 설정과 JSON→txt 변환 규칙을 확보하지 못했다. 따라서 이 코드만으로 `t_co=90`이 `000090.jpg`인지 목록 인덱스 90의 `000091.jpg`인지 확정하면 안 된다. [원저자 데이터 로더](https://raw.githubusercontent.com/JWFanggit/LOTVS-CAP/main/src/dataset.py)

`dataset.py` 18행의 `self.fps=30`, `Test.py` 102·104행의 `fps=30`도 확인했다. 이는 **평가 코드에 설정된 시간 환산 상수**다. 해당 JPEG의 실제 획득률·보존률이 30Hz였다는 영상별 provenance는 없다. 원본 FPS가 다양하고 원본 영상/FPS를 제공할 수 없다는 owner의 2025-06-08 답변과 함께 읽어야 한다. 상수 30을 채택하면 CAP 구현의 관행을 재현할 수는 있어도 DACON의 실제 ±0.3초를 입증하지 못한다. [원저자 평가기](https://raw.githubusercontent.com/JWFanggit/LOTVS-CAP/main/Test.py), [owner 답변](https://huggingface.co/datasets/JeffreyChou/MM-AU/discussions/1)

이번에는 보유 `000090.jpg`도 직접 보았다. AI 관찰상 후드 바로 앞에 흰색 픽업이 보인다. 공식 `texts`는 motorbike를, `causes`는 truck을 언급한다. **단일 프레임 관찰로 공식 정답이 틀렸다고 판정하지 않지만, 대상·문장 정합성을 확인하지 않은 상태에서 ego-contact GT로 승격할 수 없다는 구체적 이유다.** 이 관찰은 새 사람 GT가 아니다. 화면 하단의 초 단위 시계가 보이지만, 이를 OCR해 새로운 정확 FPS/PTS 정답으로 만들지 않았다. 표본 1·90·91·223의 EXIF DateTimeOriginal도 없다.

## 라이선스 판단

공식 HF 카드는 CC BY-NC 4.0이며 MM-AU GitHub는 학술 이용을, 연결된 CAP GitHub는 연구 이용을 명시한다. 공개 다운로드 가능성은 확인됐다. 로컬 대회 통합 문서 213–214행은 최소 비영리 이용이 허용된 공개 자원을 허용하되 참가자가 조건을 준수해야 한다고 기록한다. **NC라는 이유만으로 대회 전면 금지라고 판정하는 것도 잘못이다.** 다만 NC의 비상업 정의는 이용의 목적과 성격에 관한 조건이며 상금 대회라는 이름만으로 충족 또는 위반을 자동 확정하지 않는다. 이 대회의 모델 선택·수상 산출물 이용까지 포함한 별도 허락은 보유 자료에서 확인하지 못했다. [HF 데이터 조건](https://huggingface.co/datasets/JeffreyChou/MM-AU/blob/main/README.md), [CAP 연구 이용 문구](https://github.com/JWFanggit/LOTVS-CAP), [CC 공식 법문 §1(i), §2(a)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)

현재의 한정적 출처·무결성 감사와, 이 자료로 제출 모델을 학습·선택하거나 원본을 ZIP에 재배포하는 행위를 구분한다. 원저자의 데이터 공개 조건은 제3자 원천 영상의 모든 권리까지 보증하지 않는다. 이번 보고서는 대회 사용에 대한 확정 법률 허가도, 금지 판정도 아니다. 기존 모델/제출에 MM-AU를 추가하지 않았다.

## 바로 실행 가능한 후속의 범위

이미 223장을 갖고 있으므로 **추가 영상 취득 계획은 필요 없다(추가 payload 0 bytes)**. source/label binding 및 무결성 확인은 이번에 끝냈다. 이 상태에서 할 수 있는 것은 JPEG 순서와 주석 행을 보존하는 자료 감사이며, 정확 접촉 시간 채택 검증은 아니다. `t_co` 오프셋 두 가설을 임의로 하나 선택하거나 둘 중 모델에 유리한 값을 선택하지 않는다.

다음으로 결손을 실제 해소할 수 있는 좁은 행동은 공식 연락처 `lotvsmmau@gmail.com`에 **이 한 표본을 식별한 질문**을 보내는 것이다. 이번 과제에는 외부 발송 승인이 없어 보내지 않았다. 질문 내용은 다음처럼 고정할 수 있다.

1. `9ca724b1 / CAP-DATA/1-10/8/009509`, 223장, `t_co=90`이 JPEG `000090`과 `000091` 중 어느 것에 대응하는지, 그리고 그 근거인 원주석/설정 행을 요청한다.
2. 이 한 시퀀스의 원천 URL 또는 이미지 추출 sampling map이 남아 있는지 요청한다. 이미 owner가 전체 원본 FPS 제공 불가라고 답했으므로 같은 일반 FPS 질문을 반복하지 않는다. 해당 자료도 없으면 **초 단위 검증 경로는 종료**한다.
3. 이 주석이 자차와 픽업의 최초 접촉인지, 텍스트의 motorbike/truck 불일치가 무엇인지 확인한다. 답변 없이 AI 시각 관찰로 기존 사람 주석을 고쳐 쓰지 않는다.
4. 정확한 대회명과 '원본 비재배포, 외부 독립 검증 및 제출 모델 선택에만 사용, 상금·수상 산출물 귀속 규칙 존재'라는 사용 범위를 밝히고 허용 여부를 요청한다. 필요하면 이후 허용 범위 안에서만 배포 출처·라이선스 고지를 유지한다.

이는 이메일 답변을 받았다는 주장이 아니다. 답변을 기다리지 않고 같은 압축 파일을 더 받거나 다른 FPS를 시험해도 현재 결손은 해결되지 않는다. 프레임 기준이 공식적으로 확인되더라도 시간 대응이 없으면 frame-only 오차만 가능하고, entry/side/space 또는 전체 Stage2 점수는 계속 산출 불가다. 실제 원천을 찾더라도 MM-AU JPEG와 순서·누락·중복의 대응 및 기존 DADA/Nexar/public 원천 중복 검사를 별도로 통과해야 한다.
