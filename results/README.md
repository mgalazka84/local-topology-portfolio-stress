# Recorded outputs

The CSV and JSON files preserve the computational run reported in the working
paper. `verification.json` and `release_integrity.json` record the checks;
`environment.json` records package versions and thread limits.

`simulation.csv` contains one row per replication and method.
`empirical_*_daily.csv` contains one row per date, cost and method.
`empirical_*_holdings.json` contains target weights at rebalancing dates.
`*_solvers.csv` retains objective bounds, timing and solver status.
Historical panels are overlapping representations of US equities.

Return values are decimal fractions; 0.001 is 10 basis points. Derived strategy
returns are provided; the underlying third-party return files are retrieved with
`src/download_data.py`. See the root README for the interpretation limits.
