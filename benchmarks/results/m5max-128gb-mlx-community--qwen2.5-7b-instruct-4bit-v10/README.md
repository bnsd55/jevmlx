# Bench results folder

Produced by `jevmlx bench`. One subfolder per (machine-model, track,
scorer, dataset) combination, each holding `predictions.jsonl`,
`run.json`, `report.json`, and `report.md`; dataset lock files sit at
the top level. `SUMMARY.md` is the one-glance table.

Commit this folder and nothing else (no caches, no model weights).

Predictions were gzipped (`predictions.jsonl.gz`) to keep the folder
under the 5 MB commit limit; decompress with `gunzip` to inspect.
