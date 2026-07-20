# UDIAS Data Card (템플릿)

보고서 3장: "Dataset documentation should include a clear data card."
아래 [ ] 항목을 실제 수치·사실로 채워서 데이터셋과 함께 배포한다.
scripts/01, 02가 출력하는 align_report.json과 split 분포표를 그대로 인용하면 된다.

## 1. 개요
- 이름: UDIAS — 저고도 RGB–IR 해상 선박 탐지 정렬 데이터셋
- 버전 / 배포일: [v2.0 / YYYY-MM-DD]
- 목적: RGB–IR 페어 기반 (1) 크로스모달 정렬, (2) 단일/이중 모달 선박 탐지,
  (3) 융합 모델 평가. IR은 SAR의 대리(proxy) 모달리티로 사용됨 → §7 한계 참조.

## 2. 수집
- 플랫폼/센서: [드론 기종, RGB 카메라 모델, IR 카메라 모델, 해상도, FOV]
- 촬영 지역: [항만/연안 명칭, 좌표 범위] / 기간: [ ]
- 영상 수 / 추출 프레임 수 / 페어 수: [ ] (FRAME_INTERVAL=10)

## 3. 구성 및 분할
- split 방식: 영상(세션) 단위 층화 분할 (주/야 × 장면유형), seed=42
- 누수 검사: video-level 검사 통과, pHash near-duplicate 검사 [통과/예외 N건 처리]
- 분포표: [scripts/02 summarize() 출력 붙여넣기]

## 4. 정렬 (Alignment)
- 방법: SIFT + ratio test 0.75 + RANSAC(5.0px), min_inliers=15
- 성공률: 전체 [ ]%, 주간 [ ]%, 야간 [ ]%  (align_report.json)
- 평균 재투영 오차: [ ]px / 랜드마크 검증셋 오차: [ ]px (페어 [ ]개, 대응점 [ ]개)
- 실패 페어 처리: aligned=false로 명시 기록, early fusion 학습에서 제외

## 5. 라벨
- 클래스: Ship (1클래스)
- 절차: YOLO11x COCO 초벌 → Label Studio 전수 검수 ([ ]명, 검수율 [ ]%)
- 타깃별 플래그: visible_in{rgb, ir, both}, uncertain
- 검수자 간 일치도(선택): [IoU 기준 agreement]
- 타깃 크기 분포: small [ ]% / medium [ ]% / large [ ]%

## 6. 벤치마크 베이스라인
- 프로토콜: COCO mAP@[.5:.95], 크기별 AP, 장면유형별 분리 리포트, 3 seeds mean±std
- 표: [runs/benchmark.json → format_benchmark_table 출력 붙여넣기]
- 재현 정보: 이미지 640px, epochs [ ], batch [ ], GPU [ ], 추론속도 [ ] FPS

## 7. 한계 및 의도된 용도
- IR은 SAR과 물리적 생성 원리가 다름(수동 열복사 vs 능동 마이크로파 산란).
  본 데이터셋의 정렬/융합 결론이 위성 SAR–EO로 그대로 일반화된다고 주장하지 않음.
- 지리적 편향: [특정 항만/계절에 편중된 정도]
- 야간 EO 품질 저하로 야간 페어의 정렬 성공률이 낮음: [수치]
- 감시 목적 오남용 방지: 의도된 용도는 [해상 안전/연구], 접근 제한: [ ]

## 8. 라이선스 / 인용
- 라이선스: [ ] / 인용 형식: [ ]
