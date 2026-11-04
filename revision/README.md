# Local topology stress: revision 0.2.1 (4 November 2026)

This release implements the supplied major-revision assessment. It preserves the negative findings: cut rerooting is faster on all tested complete graphs; the path oracle wins on the tested sparse 4-NN graphs. Robust allocation has higher volatility than linear shrinkage in all reported historical specifications.

Round two evaluates each unchanged simulated allocation with the same population objective, `52.5*population_variance + cost*two_way_turnover`, including Equal and HRP. Sample has the lowest mean relative variance in all eight 10 bps settings, but the lowest mean risk-plus-cost objective in only three (one a near tie with LW). These are descriptive simulation rankings. No additional portfolios or data were fitted.

## Reproduce the revised study

Run commands from this `revision` directory. The original 3 November release remains in the repository root and is pinned at commit 621731f62a1f155bea1614418674dbcaf10d01f0.

```bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python src/download_data.py
python src/verify.py
python src/revision_benchmark.py
python src/run_revision.py
python src/revision_inference.py
python src/revision_report.py
python src/revision_integrity.py
python src/revision_solver_audit.py
```

Run timing separately from other numerical jobs. `run_revision.py` uses four independent processes, each with one BLAS/OpenMP thread. It records every job status; any nonzero exit is a failure, not an omitted replication. `--retry` reruns only failed jobs. Seeds are indexed by asset count, setting and replication.

The public [full_results.zip](full_results.zip) archive contains all recorded simulation replications, historical daily strategy ledgers, holdings, solver records, inference, benchmark and verification outputs. It contains no original third-party return files. Its SHA-256 is `c1516c979e6bba92cd70c84f9976310db19fa2a3037bc3ac1b9dd52ec10a2515` (23,103,343 bytes), also recorded in `MANIFEST.json`.

To generate tables and figures from the recorded outputs without rerunning the experiments, install the requirements and run:

```bash
python -m zipfile -e full_results.zip .
python src/revision_report.py
```

The aggregate `results/simulation.csv` is reconstructed from the eight `sim_n_setting.csv` files. Full recomputation and independent ledger replay require source data and use the complete command sequence above.

The archive is unchanged from version 0.2.0. Round-two derived summaries are in `results/round2_simulation_summary.csv` and paired comparisons in `results/round2_simulation_paired.csv`. `src/round2_scoring.py` regenerates them and manuscript Table 3 directly from the eight archived simulation files; `revision_report.py` invokes the same calculation. `results/round2_scoring_verification.json` records every input hash and the arithmetic checks. Cost penalties and objectives use basis-point units (10,000 times wealth units); the cost term is the target-weight penalty in the decision problem, not the exact self-financing historical fee. Paired intervals are descriptive t499 intervals without multiplicity adjustment. The source records allow every derived score to be reconstructed.

The data download scripts obtain French's 10/30/49 industry and daily risk-free files. This study uses the archived November 2026 hashes in `data/archived_provenance.json`. Newer French downloads can revise history: the download date alone is not a point-in-time data guarantee. Raw downloaded source files are not redistributed here. Cost accounting concerns only allocation across research industry series.

## Methods and verification

- `tree_risk.py`: direct division-free query, original branch-prefix cross-check, convex allocation and mathematical lower-bound formulas.
- `cut_baseline.py`: O(nK+L) comparator, where K counts active cuts.
- `conic_master.py`: native Clarabel QP/SOCP; restored-domain quadratic dual lower bound.
- `revision_experiments.py`: 4,000 training samples, zero/10 bps simulation comparisons, 18 historical specifications.
- `revision_inference.py`: paired studentized circular blocks, log variance ratio/excess-return Sharpe, natural bootstrap block-sum SEs; Bartlett original-sample HAC and fixed 21/63/126-day blocks. This is an explicit adaptation, not the full Ledoit–Wolf prewhitening/calibration implementation. Holm adjustment covers 12 tests per block setting. Allocations are not refitted in resamples.
- `verify.py` and `revision_benchmark.py`: exhaustive/dense cross-checks, conic versus full-scenario SLSQP comparisons, and signed/zero-edge checks.
- `revision_integrity.py`: independent dollar-trade replay of every historical row.
- `revision_solver_audit.py`: reconstructs all 17 originally flagged fits using the original root results and pinned original source.

All 53,472 revised allocation fits meet a numerically evaluated normalized global gap at most 2e-7. Clarabel status warnings remain visible; there is no claim of an interval-arithmetic certificate. Requested Clarabel tolerances are 1e-9, refined only if the independent global gap requires it. Tangent LPs retry numerical presolve failures using HiGHS interior point without presolve. The code records all such calls.

Computational figures are generated from actual numerical outputs. No investment recommendation or general performance advantage is claimed. No reuse license has been selected by the authors; no MIT license is asserted for this release.
