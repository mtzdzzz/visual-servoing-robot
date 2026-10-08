# GitHub Release Checklist

Status legend: `[x]` locally verified; `[ ]` requires a final human/GitHub-side action.

- [ ] README rendered correctly in the GitHub preview
- [x] `requirements.txt` verified in the existing Python 3.10 environment
- [x] Demo commands present in `python .\src\simulation.py --help`
- [x] README image links resolve to files under `docs/assets/`
- [x] No local absolute paths in README or docs
- [x] No private machine paths in README or docs
- [x] `.venv/` and Python caches are ignored
- [x] Curated `outputs/final_report/`, `outputs/final_tables/`, and `docs/` are not ignored
- [x] No oversized README assets detected
- [x] Stage 21 automated regression tests pass for the three demos
- [x] Stage 23 metrics are referenced from `outputs/final_report/final_metrics.csv`
- [x] Git status checked from the actual repository root and scoped to this project
- [x] Installation snippet uses the configured GitHub remote URL
- [ ] Perform final interactive visual check of all three GUI demos before recording/release
- [ ] Review staged files, commit, and push

## Pre-Push Commands

```powershell
python .\src\simulation.py --help
python -m unittest discover -s src -p "test_*.py"
python -m compileall src
git status --short -- visual-servoing-robot
```

Interactive checks:

```powershell
python .\src\simulation.py --demo-tracking
python .\src\simulation.py --demo-multitarget
python .\src\simulation.py --demo-obstacle
```

Confirm `Q`, `Esc`, `Ctrl+C`, and PyBullet GUI closure exit cleanly. Do not commit `.venv/`, cache directories, or private machine-specific files.
