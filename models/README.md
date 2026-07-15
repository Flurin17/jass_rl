# Model artifacts and evidence

The compact JSON qualification reports, their input shards, and the cited run
manifests are tracked so a clone can inspect the evidence and rerun the strict
merges. Training checkpoints remain ignored because binary model distribution
belongs in a release or artifact store rather than Git history.

The qualified advisor archive used by the reports is currently local at
`models/experiments/ppo_full_consolidation_v1/20260715_095439/model_final.zip`.
Its SHA-256 is
`513e96d900144a3e1ccc80f748561538ce2430a0068795716d16061dfd9accfa`
and its recorded size is 24,016,834 bytes. A fresh clone can inspect and merge
the evidence, but cannot run that exact advisor until this archive is supplied
at the documented path or a new model is trained. Do not imply otherwise; add a
release URL here before publishing the checkpoint externally.
