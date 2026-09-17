## Correction Method Selection Guide
BS-NET `correct_attenuation()` 함수의 `method` 파라미터:

| Method | 코드 | 설명 | 논문 근거 | Ceiling 해소 |
|--------|------|------|-----------|-------------|
| Original | `"original"` | 표준 CTT 보정 + hard clip | Spearman (1904) | ✗ (85%) |
| **Fisher z** | `"fisher_z"` | z-space에서 가법 보정 → tanh 역변환 | Shou (2014), Teeuw (2021) | **✓ (0%)** |
| Partial | `"partial"` | α=0.5 감쇠 보정 | Zimmerman (2007) | ✓ (0%) |
| Soft clamp | `"soft_clamp"` | tanh 압축 (순위 보존) | — | ✓ (0%) |

**권장**: `"fisher_z"` (학술적으로 가장 방어 가능, ceiling 완전 해소, 의미 있는 improvement 유지)
