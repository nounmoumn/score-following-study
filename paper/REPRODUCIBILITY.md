# Reproducibility guide

## Environment

- Repository root: this file's parent directory is `paper/`; run commands from its parent.
- Frozen Matchmaker commit used by the audio experiments: `40fd6e7602bd9e596cea1beafdcee4853b9f2c98`.
- Frozen validator SHA-256: `233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff`.
- Python dependencies are recorded in `pyproject.toml` and `uv.lock`; the completed experiments used `.venv_matchmaker`.

## Fast paper audit

```powershell
& .\.venv_matchmaker\Scripts\python.exe scripts/audit_relocation_research_package.py
```

Expected result: top-level `status` is `PASS`. This means the stored evidence reproduces the declared boundary—MIDI lockbox PASS, both audio validations FAIL—not that every experiment passed.

Output: `results/relocation_research_package_audit.json`.

## Regenerate paper figures and table

```powershell
& .\.venv_matchmaker\Scripts\python.exe scripts/build_relocation_paper_assets.py
```

Outputs are listed in `paper/asset_manifest.json`. The script reads only the four frozen aggregate JSON files.

## Independent component audits

The component audits are write-once by design and refuse to overwrite an existing audit artifact. Their verified outputs are already included in each result directory. To execute them again, use a fresh copy of the package in which only the corresponding `independent_audit.json` has been omitted; do not delete the archived evidence in the primary workspace.

```powershell
& .\.venv_matchmaker\Scripts\python.exe scripts/audit_online_relocation_validation.py
& .\.venv_matchmaker\Scripts\python.exe scripts/audit_online_relocation_lockbox.py
& .\.venv_matchmaker\Scripts\python.exe scripts/audit_online_relocation_vienna_audio.py
& .\.venv_matchmaker\Scripts\python.exe scripts/audit_online_relocation_audio_mean_evidence.py
```

The component audits independently reconstruct model decisions and gates. Their stored outputs report zero maximum numeric residual and no frozen-source hash drift.

## Tests and static validation

```powershell
& .\.venv_matchmaker\Scripts\python.exe tests/test_matchmaker_ambiguity_rescue.py -v
& .\.venv_matchmaker\Scripts\python.exe -m py_compile scripts/audit_relocation_research_package.py scripts/build_relocation_paper_assets.py
```

## Interpretation of failure

An audit script returning `PASS` for an audio experiment means it successfully reproduced the experiment's predeclared `FAIL` verdict. Keep implementation validity separate from scientific success.
