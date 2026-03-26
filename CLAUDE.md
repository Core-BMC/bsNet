# CLAUDE.md — BS-NET Project

## Co-Author
이 프로젝트의 리팩토링 작업은 Claude (Anthropic)의 지원을 받아 수행되었습니다.

## Git Policy
- 커밋 메시지에 Co-Authored-By 트레일러를 포함하지 않는다.
- Co-author 정보는 이 문서에만 기록한다.

## Conventions
- Python 3.9+, PEP 8, Black (88 cols), isort, ruff
- Type hints, Google Style docstrings, snake_case
- N806 exemption: neuroimaging domain variables (fc_true_T, G, C, L)
- Tests: pytest with mock-based dry-run (nilearn/nibabel not required)
