# Local topology stress in tree-based portfolios

Reproducible code and recorded computational results accompanying the working
manuscript **Local topology stress in tree-based portfolios: an exact
edge-exchange algorithm**, by Marek Gałązka and Hanna Wdowicka.

The algorithm evaluates portfolio variance over a reference covariance tree
and every allowed single-edge exchange, reusing directed exposure messages
and path prefix/suffix sums. The uncertainty family is a deterministic local
stress set; it is not a statistical confidence region.

## Contents

- `src/tree_risk.py`: path-product covariance, exchange queries and cost-aware allocation.
- `src/verify.py`: independent dense-covariance and small optimization checks.
- `src/experiments.py`: seeded simulations, historical allocation and timing.
- `src/download_data.py`: public source-data retrieval and provenance checks.
- `src/analyze.py`: paired summaries, tables and computational figures.
- `src/check_release.py`: integrity checks and independent accounting replay.
- `results/`: replication results, daily strategy series, weights and solver diagnostics.
- `data/`: source metadata and hashes; original third-party return files are downloaded separately.
- `PROTOCOL.txt`: internal design fixed before the experiments, not an external preregistration.
- `SHA256SUMS`: hashes for the recorded code, metadata and result files.

## Reproduce

Tested with Python 3.12.14. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/download_data.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/experiments.py simulation --reps 100
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/experiments.py empirical --n 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/experiments.py empirical --n 30
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/experiments.py benchmark
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/analyze.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/check_release.py
```

To regenerate tables and figures from the recorded results, only the dependency
installation and `python src/analyze.py` are needed. The script creates its output
directories. For `check_release.py`, download the original return panels first.
Run benchmarks after other computations finish; shared-machine timing varies.

`data/archived_provenance.json` preserves the original experiment's data vintage.
The provider may revise historical returns; a different download hash can produce
different results. No point-in-time vintage claim is made.

## Recorded validation and interpretation

- 18,932 dense-covariance scenario comparisons across 245 reference trees;
  largest absolute risk discrepancy 5.69e-14.
- 800 independently simulated samples in eight settings; 2,400 optimization fits.
- Two overlapping US industry panels (10 and 30 assets), 2000–2025:
  6,539 daily observations and 312 monthly decisions per series, at three cost levels.
- All 8,016 simulation/historical optimization fits meet the stated numerical
  objective-gap threshold. Binary64 diagnostics are not rounding-verified certificates.
- Historical fees and returns independently reconstructed from archived weights.

Local robustness has small and setting-dependent financial effects. Shrinkage
allocation has lower realized volatility in both principal historical panels.
Industry research returns are not directly tradeable securities, and modeled
costs exclude trades within each industry portfolio.

## Assistance and status

ChatGPT assisted development and debugging of the code and preparation of
programmatic plots. Numerical outputs were produced by executing the documented
calculations. This repository is a computational companion to a working paper;
it is not a claim of journal acceptance or independent validation of novelty.

The manuscript, cover letter and author-review notes are maintained separately.
A reuse license has not yet been selected for this repository.
