# Source provenance

The JSON files identify the Kenneth French daily value-weighted industry return
panels, download date, selected range and original-file SHA-256 digests.
`archived_provenance.json` is the immutable reference for the reported run.

Run `python src/download_data.py` from the repository root to obtain the original
files. They are not redistributed here and are ignored by Git. The script warns
when the downloaded vintage differs from the archived experiment.
