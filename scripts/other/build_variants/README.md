# Build Variants Test Scripts

Scripts to test different compilation and optimization variants.

## Available Scripts

### `test_o0_quick.sh`
Test with **no optimization** (`-O0`).

Useful for:
- Debugging (more readable code)
- Performance comparison
- Correctness verification without aggressive optimizations

```bash
bash other/build_variants/test_o0_quick.sh
```

**Build variant**: `o0`

### `test_o1_quick.sh`
Test with **optimization level 1** (`-O1`).

Balance between:
- Better performance than `-O0`
- Faster compilation than `-Ofast`
- Still debuggable

```bash
bash other/build_variants/test_o1_quick.sh
```

**Build variant**: `o1`

### `test_noarch_quick.sh`
Test **without architecture-specific optimizations** (no `-march=native`).

Useful for:
- Portability (generic x86-64 code)
- Comparison with optimized builds
- Compatibility verification

```bash
bash other/build_variants/test_noarch_quick.sh
```

**Build variant**: `noarch`

## Build Variants

| Variant | Flags | Performance | Portability | Debug |
|:--------|:------|:------------|:------------|:------|
| `o0` | `-O0 -fopenmp` | Low | High | Easy |
| `o1` | `-O1 -flto -fopenmp` | Medium | High | Medium |
| `noarch` | `-Ofast -flto -fopenmp` | High | High | Difficult |
| `ofast` | `-Ofast -flto -fopenmp -march=native` | Maximum | Low | Difficult |

## Output

Results are saved in standard CSV files with specific `BUILD_VARIANT`:
- `data/SM3800083_strong_parallel_results.csv`
- `data/SM3800083_weak_parallel_results.csv`
- `data/SM3800083_omp_serial_results.csv`

**`BuildVariant` column**: Identifies the build variant used.

## Performance Comparison

To compare performance:

```bash
# Run all tests
bash other/build_variants/test_o0_quick.sh
bash other/build_variants/test_o1_quick.sh
bash other/build_variants/test_noarch_quick.sh

# Analyze CSV
grep "o0\|o1\|noarch\|ofast" data/SM3800083_*_results.csv
```

## When to Use

- **During development**: `test_o0_quick.sh` for debugging
- **Comparison**: All three to see optimization impact
- **Portability**: `test_noarch_quick.sh` to verify compatibility

## Note

Complete tests in the report use `ofast_omp_improved` (fully optimized variant). These scripts are for comparison and development.
