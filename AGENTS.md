# AGENTS.md

## Purpose
Agent operating guide for this repository.
`.py` files are the primary editing targets; `.ipynb` files are synced from them via jupytext after changes are confirmed working.
All Python-dependent commands must be run via `uv run` to ensure the correct environment is used.

## Project Layout
- `01_bbox_crops.*`: crop extraction from VIA CSV
- `02_yolo_oneclass_from_via.*`: YOLO dataset creation + one-class training
- `05_vit_behavior_classifier.*`: ViT training + evaluation
- `06_cow_detection_and_behavior_pipeline.*`: detect + classify inference
- `06a_botsort_pipeline.*`: tracking-enhanced inference
- `README.md`: run order and project context
- `dataset.md`: dataset provenance/constraints

Generated/local directories (ignored):
- `data/` (large dataset files)
- `workdir/` (intermediate data)
- `artifacts/` (models, runs, figures, demo outputs)

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ultralytics opencv-python numpy matplotlib scikit-learn pandas tqdm
pip install torch transformers datasets evaluate
```

Optional dev tools:
```bash
pip install pytest ruff black mypy
```

## Build and Run Commands
No package build system exists (no `pyproject.toml`, no Makefile).
Run workflow stages directly (always via `uv run`):
```bash
uv run python 01_bbox_crops.py
uv run python 02_yolo_oneclass_from_via.py
uv run python 05_vit_behavior_classifier.py
uv run python 06_cow_detection_and_behavior_pipeline.py
uv run python 06a_botsort_pipeline.py
```

Notebook usage:
```bash
uv run jupyter notebook
```

Headless notebook smoke run:
```bash
uv run jupyter nbconvert --to notebook --execute 01_bbox_crops.ipynb --output /tmp/01_bbox_crops.executed.ipynb
```

## Lint and Format
Use conservative defaults:
```bash
uv run ruff check .
uv run black --check .
```

Format only touched files when needed:
```bash
uv run black 01_bbox_crops.py 02_yolo_oneclass_from_via.py
```

## Test Commands
There is no committed `tests/` suite yet.
Use these commands once tests exist:

## Code Style Guidelines

### Imports
- Group imports: standard library, third-party, local.
- Prefer one import per line in new edits.
- Remove unused imports and avoid repeated imports across cells.

### Formatting
- Target Python 3.10+.
- Keep line length roughly 88-100.
- Prefer readable multi-line calls over compressed one-liners.
- Preserve notebook markers in mirrors: `# %%` and `# %% [markdown]`.

### Types
- Add type hints for new helper functions and non-trivial return values.
- Prefer concrete containers (`list[str]`, `dict[str, float]`).
- Keep typing practical; avoid heavy abstractions in notebook-style code.

### Naming
- Constants: `UPPER_SNAKE_CASE`.
- Variables/functions: `snake_case`.
- Use consistent labels: `stand`, `lying down`, `foraging`, `drinking water`, `rumination`.
- Path variables should end with `_path` or `_dir`.

### Error Handling
- Do not add bare `except:` blocks.
- Prefer `except Exception as e` with concise context.
- Fail fast for missing required inputs/models (`Path.exists()`).
- For per-item failures (single frame/crop), continue and report skip counts.

### Paths and Outputs
- Prefer `pathlib.Path`.
- Use `mkdir(parents=True, exist_ok=True)` for output directories.
- Do not write generated outputs to repository root.
- Keep outputs under `artifacts/` and intermediates under `workdir/`.

### ML/Pipeline Conventions
- Keep reproducibility controls (`torch`, `numpy`, `random` seeds).
- Keep class mappings consistent across stages.
- Preserve video-based splits for detection to reduce leakage.
- Keep thresholds/frame limits as named constants.

## Notebook and Mirror Rules
- **Always edit `.py` files**, never `.ipynb` files directly.
- Treat `.py` files as the source of truth; `.ipynb` files are derived outputs.
- Do not add argparse/CLI wrappers unless explicitly requested.
- Keep mirror code order aligned with notebook execution order.
- Once changes in a `.py` file are confirmed working, sync to the corresponding notebook using jupytext:
  ```bash
  uv run jupytext --sync <filename>.py
  ```
- To sync all mirrors at once:
  ```bash
  uv run jupytext --sync *.py
  ```

## Runtime/Data Constraints
- `data/` is required locally and not committed.
- GPU is strongly preferred for training stages.
- Avoid committing generated models/runs/media unless explicitly requested.
- Follow `.gitignore` for `data/`, `workdir/`, and `artifacts/`.

## Cursor/Copilot Rules
No repo-specific rule files were found:
- `.cursorrules` (not present)
- `.cursor/rules/` (not present)
- `.github/copilot-instructions.md` (not present)

If added later, treat them as higher-priority instructions and update this file.
