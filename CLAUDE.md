# CLAUDE.md — BS-NET Project

## Co-Author
이 프로젝트의 리팩토링 작업은 Claude (Anthropic)의 지원을 받아 수행되었습니다.

## Conventions
- Python 3.9+, PEP 8, Black (88 cols), isort, ruff
- Type hints, Google Style docstrings, snake_case
- N806 exemption: neuroimaging domain variables (fc_true_T, G, C, L)
- Tests: pytest with mock-based dry-run (nilearn/nibabel not required)
