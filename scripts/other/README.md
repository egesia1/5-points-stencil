# Legacy and Utility Scripts

This directory contains legacy scripts, utilities, and support scripts that **were not used to generate the main report results**.

The scripts used for results production are located in the main `scripts/` directory.

## Structure

This directory is organized into subdirectories by type:

- **[`legacy/`](legacy/)** - Legacy scaling scripts (replaced by parametric versions)
- **[`quick_tests/`](quick_tests/)** - Scripts for quick local tests
- **[`visualization/`](visualization/)** - Python scripts for results visualization
- **[`build_variants/`](build_variants/)** - Tests for different compilation variants
- **[`utility/`](utility/)** - Utility scripts for job management and summaries

## Quick Reference

### Legacy Scaling Scripts
```bash
bash other/legacy/strong_scaling.sh
bash other/legacy/weak_scaling.sh
bash other/legacy/omp_scaling.sh
```

### Quick Tests
```bash
bash other/quick_tests/quick_test_all.sh
bash other/quick_tests/quick_config_test.sh 8 14 1
```

### Visualization
```bash
python3 other/visualization/visualize_heat_diffusion.py
python3 other/visualization/visualize_heat_3d.py
bash other/visualization/run_viz.sh
```

### Build Variants
```bash
bash other/build_variants/test_o0_quick.sh
bash other/build_variants/test_o1_quick.sh
bash other/build_variants/test_noarch_quick.sh
```

### Utility
```bash
bash other/utility/relaunch_failed_jobs.sh
bash other/utility/boost_summary.sh
bash other/utility/dcgp_summary.sh
```

## Notes

These scripts are maintained for:
- **Reference**: Comparison with previous versions
- **Development**: Quick tests during development
- **Debug**: Diagnostics and troubleshooting
- **Visualization**: Visual analysis of results

For production tests and report results, use the scripts in the main `scripts/` directory.
