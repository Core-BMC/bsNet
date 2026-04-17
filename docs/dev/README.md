# Dev Log 기록 지침

이 디렉토리(`docs/dev/`)는 bsNet 프로젝트의 모든 개발 세션 기록을 보관한다.  
Claude Code 인스턴스는 **컨텍스트 시작 시 이 README를 읽고**, 세션 내 작업을 반드시 기록해야 한다.

---

## 워크플로우

> 상위 프로토콜: `CLAUDE.md` (또는 `AGENTS.md`)의 **Session Protocol** 섹션 참조.
> 이 문서는 dev log **기록 형식**에 대한 상세 지침이다.

```
[세션 시작 — Recall]
a) CLAUDE.md / AGENTS.md 읽기  →  정책, 현재 상태, Session Protocol 확인
b) docs/dev/ 최근 로그 읽기  →  직전 세션 TODO를 시작점으로 삼기
c) git status + git log  →  미반영 변경 파악

[작업 중]
d) 오늘 날짜의 로그 파일 확인 또는 생성
e) 프로젝트 작업 수행 + 병행하여 기록

[세션 종료 — Relay]
f) dev log TODO 섹션 갱신 (다음 세션 시작점)
g) CLAUDE.md / AGENTS.md Current Status 간략 갱신
h) 두 파일 동기화 확인
```

---

## 로그 파일 규칙

### 파일명
```
docs/dev/{YY-MM-DD-HH-MM}_dev_logs.md
```
- `YY-MM-DD-HH-MM` : 해당 날(day)의 **첫 세션** 시작 시각
- 예: `26-04-04-13-36_dev_logs.md`

### 같은 날 여러 세션
- 파일을 새로 만들지 않는다.
- **같은 날짜 파일에 새 세션 섹션을 추가**한다.
- 확인 방법: `docs/dev/` 에서 오늘 날짜(`YY-MM-DD`)로 시작하는 파일이 있으면 그 파일에 추가.

---

## 로그 파일 형식

CHANGELOG.md 와 동일한 스타일을 따른다 ([Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 기반).

```markdown
# Dev Log — YYYY-MM-DD

## Session 1 — HH:MM (간단한 세션 주제)

### Added
- **파일명 또는 기능명**: 추가한 내용 요약.

### Changed
- **파일명 또는 기능명**: 변경 내용 및 이유.

### Fixed
- **파일명 또는 기능명**: 수정한 버그 또는 문제.

### Experiments / Results
- 실험 조건, 결과 수치, 관찰 사항 (해당 시에만 작성).

### TODO
- [ ] 다음 세션에서 이어서 할 작업
- [ ] 미완성 항목 또는 검토 필요 사항
```

### 규칙 요약

| 항목 | 규칙 |
|------|------|
| 최소 기록 | 세션당 한 줄 이상 (어떤 작업을 했는지 요약) |
| 섹션 선택 | Added / Changed / Fixed / Experiments 중 해당하는 것만 사용 |
| TODO | 반드시 작성 — 다음 컨텍스트의 시작점이 됨 |
| 언어 | 한국어 또는 영어 무관, 혼용 가능 |
| 수치/경로 | 실험 결과는 구체적인 수치와 파일 경로를 포함 |

---

## 예시

```markdown
# Dev Log — 2026-04-04

## Session 1 — 13:36 (xcpd sweep 로그 분석 + CLAUDE.md 초기화)

### Added
- **CLAUDE.md**: 프로젝트 전반 지침 초기화 (빌드/테스트/아키텍처).
- **docs/dev/README.md**: 개발 로그 기록 지침 문서 신규 작성.

### Changed
- **CLAUDE.md**: 세션 로그 기록 지침 항목 추가 (짧은 버전, 상세는 이 파일 참고).

### TODO
- [ ] `logs/sweep_xcpd_4s256.log` 분석 후 이상치 원인 파악
- [ ] Figure 1 Panel B atlas 투명도 재조정 검토
```

---

## 이 디렉토리의 파일 목록

로그 파일이 쌓이면 이 섹션을 업데이트한다.

| 파일 | 날짜 | 주요 내용 |
|------|------|-----------|
| [26-03-26-00-00_dev_logs.md](26-03-26-00-00_dev_logs.md) | 2026-03-26 | PoC 파이프라인 최적화 (LW+SB+Prior), N=9 코호트 86.8%, 실데이터 스케일업 착수 |
| [26-03-27-00-00_dev_logs.md](26-03-27-00-00_dev_logs.md) | 2026-03-27 | N=100 실증 완료 (R²=0.488, Pass 91%), Figure 2 KDE 편향 보정, Topology N=400 |
| [26-03-28-00-00_dev_logs.md](26-03-28-00-00_dev_logs.md) | 2026-03-28 | 방어 실험 Track A–F 완료, Ceiling 53.6% 발견, Fisher z 채택 (Pilot ceiling 0%) |
| [26-03-29-00-00_dev_logs.md](26-03-29-00-00_dev_logs.md) | 2026-03-29 | ABIDE N=468 ρ̂T=0.843 (ceiling 0%), ADHD-200 N=40 파이프라인 구축 및 검증 |
| [26-03-30-00-00_dev_logs.md](26-03-30-00-00_dev_logs.md) | 2026-03-30 | Track E 배치 리팩토링, Track H 분류 (BS-NET Acc=0.720), Figure 3/4/5/6/7 통합 |
| [26-04-01-00-00_dev_logs.md](26-04-01-00-00_dev_logs.md) | 2026-04-01 | Figure 1–7 스타일 완성 (ATLAS_META), ds000243 파이프라인 인프라 구축 |
| [26-04-04-13-36_dev_logs.md](26-04-04-13-36_dev_logs.md) | 2026-04-04 | CLAUDE.md 초기화, 세션 로그 체계 구축 (docs/dev/ 신설) |
| [26-04-10-00-00_dev_logs.md](26-04-10-00-00_dev_logs.md) | 2026-04-10 | Fig4 sliding-window 확장, legend 전면 개편, 스토리라인 확정, 논문 집필 준비 문서 3건 |
| [26-04-17-13-48_dev_logs.md](26-04-17-13-48_dev_logs.md) | 2026-04-17 | Fig 3를 N=468 메인 검증으로 재정의, strict+CONSORT를 Fig S3로 분리, ABIDE supplementary legend 정렬 |
