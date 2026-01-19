# Utility Scripts

Utility scripts for job management, summaries, and support operations.

## Available Scripts

### `relaunch_failed_jobs.sh`
**Relaunch failed jobs** on Leonardo.

Identifies failed jobs and relaunches them automatically.

```bash
bash other/utility/relaunch_failed_jobs.sh
```

**Typical usage**:
1. Check failed jobs: `squeue -u $USER`
2. Check logs: `tail -f slurm-*.out`
3. Relaunch: `bash other/utility/relaunch_failed_jobs.sh`

### `boost_summary.sh`
**Summary for boost partition** on Leonardo.

Generates a summary of jobs executed on the `boost_usr_prod` partition.

```bash
bash other/utility/boost_summary.sh
```

**Output**: `boost_summary.txt`

**Content**:
- Number of jobs executed
- Timing statistics
- Tested configurations
- Success rate

### `dcgp_summary.sh`
**Summary for dcgp partition** on Leonardo.

Generates a summary of jobs executed on the `dcgp_usr_prod` partition.

```bash
bash other/utility/dcgp_summary.sh
```

**Output**: `dcgp_summary.txt`

**Content**:
- Number of jobs executed
- Timing statistics
- Tested configurations
- Success rate

## Output Files

### `boost_summary.txt`
Text summary of jobs on boost partition.

### `dcgp_summary.txt`
Text summary of jobs on dcgp partition.

## When to Use

- **After complete tests**: Generate summary for analysis
- **Debug**: Relaunch failed jobs
- **Monitoring**: Check job status

## Notes

These scripts are specific to the Leonardo environment and may require adaptations for other clusters.
