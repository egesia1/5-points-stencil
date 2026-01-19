# Legacy Scaling Scripts

Legacy scripts for scalability tests, replaced by more comprehensive parametric versions.

## Available Scripts

### `strong_scaling.sh`
Legacy version of strong scaling test.

**Replaced by**: `../parametric_strong_scaling.sh`

- Fixed configuration: 8 tasks/node × 14 cpus/task
- Single energy source
- 5 node levels (1, 2, 4, 8, 16)

```bash
bash other/legacy/strong_scaling.sh
```

### `weak_scaling.sh`
Legacy version of weak scaling test.

**Replaced by**: `../parametric_weak_scaling.sh`

- Fixed configuration: 8 tasks/node × 14 cpus/task
- Single energy source
- 5 node levels (1, 2, 4, 8, 16)

```bash
bash other/legacy/weak_scaling.sh
```

### `omp_scaling.sh`
Legacy version of OpenMP scaling test.

**Replaced by**: `../parametric_omp_scaling.sh`

- 1 node, 1 task
- Variable thread counts
- Single energy source

```bash
bash other/legacy/omp_scaling.sh
```

## Differences with Parametric Versions

| Feature | Legacy | Parametric |
|:--------|:-------|:-----------|
| Configurations | 1 fixed | 3 variants |
| Energy Sources | 1 fixed | 3 levels |
| Total jobs | ~15 | 45-117 |
| Flexibility | Low | High |

## When to Use

Use these scripts only for:
- Quick tests with standard configuration
- Comparison with previous results
- Debugging specific issues

For complete tests, always use the parametric versions.
