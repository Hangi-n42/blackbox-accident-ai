# DLC-2021 제한된 외부 진단 표본

이 폴더는 DACON 제출 자산이 아니라 별도 영역의 연구 진단 자료다. 이번 동결 진단에서는 가중치 학습·혼합 비율 조정·임계값 조정을 하지 않았다. 배포 JPEG를 선택 취득했으며 동영상 파일을 만들지 않았다. 이후 주행 원본 오탐 진단과 함께 TPO 단독 변경 여부를 검토했고, 현재 혼합 모델을 유지했다. 이는 개발 단계의 판단 근거이며 독립적인 최종 대회 성능 평가가 아니다.

원자료: Polevoy, D. V.; Sigareva, I. V.; Ershova, D. M.; Arlazarov, V. V.; Nikolaev, D. P.; Ming, Z.; Luqman, M. M.; Burie, J.-C. **Document Liveness Challenge Dataset (DLC-2021)**. Journal of Imaging 2022, 8, 181. [논문](https://doi.org/10.3390/jimaging8070181).

취득 버전: [물리적 원본 Part 1, 7467028](https://zenodo.org/records/7467028), [화면 재촬영 Part 2, 6466770](https://zenodo.org/records/6466770). 분류·촬영 조건은 Part 1의 공식 정정 CSV를 사용했다. Part 2 원래 CSV의 lva_passport 한 행 오기를 수정한 배포본이며, 미디어 체크섬을 변경한 것으로 보고하지 않는다.

양쪽 배포물의 `license.txt`를 읽고 원문을 각각 보존했다. 데이터는 **Creative Commons Attribution-ShareAlike 2.5 Generic** 조건으로 제공된다. [라이선스](https://creativecommons.org/licenses/by-sa/2.5/). 얼굴 이미지 출처는 [Generated Photos](https://generated.photos/)이며 제공자의 권장에 따라 기여를 표시한다.

변경: 공개 압축 파일에서 일부 JPEG만 선택했다. JPEG 바이트는 수정하지 않았으며 추출 후 CRC32를 검증했다. 선택 목록과 추가 메타데이터·측정치는 이번 연구에서 작성했다. 모델 입력 처리 외에 원본 저장 JPEG의 자르기·재압축·색 보정은 하지 않았다.

`selection_plan.json`은 모델 평가 전에 고정한 선택 기준과 목록이다. `acquisition_manifest.json`은 영상·문서·카메라·화면 표기, 정확한 압축 내부 경로, 다운로드 URL/바이트 범위, CRC32, SHA-256 및 라이선스를 기록한다. `network_log.json`은 응답 길이와 범위만 기록하며 인증정보를 저장하지 않는다. 누적 응답 데이터는 200,000,000바이트 이내로 제한한다.

검증 단위는 **한 원영상에 해당하는 배포 JPEG의 시간순 부분집합**이다. 원본 영상 디코딩 결과와 동일한 평가가 아니다. DLC의 `or`은 물리적 문서 촬영이라는 뜻이며 TPO는 화면뿐 아니라 인쇄 공격도 학습했다. 따라서 물리적 문서를 attack으로 분류하는 현상만으로 블랙박스 원본 오탐의 원인을 입증할 수 없다. 문서 영역의 결과를 대회 점수로 외삽하지 않는다.
