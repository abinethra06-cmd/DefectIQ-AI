# From Detection to Disposition — Defect Inspector (HorizonXT PS19)

AI visual inspection that detects steel surface defects, scores their severity, and recommends
`ACCEPT`, `REWORK`, `REJECT`, or `MANUAL_REVIEW` with a plain-language explanation.

## Flow
Image -> Member A detector -> `src/severity.py` -> `src/storage.py` (SQLite) -> Member C Streamlit dashboard.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest tests -v
```
If PowerShell blocks activation, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
and `.\.venv\Scripts\python.exe -m pytest tests -v`.

## Important integration contract
Member A's detector must return:
`(defects, annotated_image, inference_ms)`, where each defect has `type`, `confidence`,
`box`, `box_norm`, and `area_ratio`. Class names must exactly match `config.yaml`.

Member C should call `inspect(image_bgr, image_name, profile="Standard")` from `src.pipeline`
and display the returned `decision`, `max_score`, `reason`, `defects`, `annotated`,
`inference_ms`, and `id`.

## Run tests
```powershell
python -m pytest tests -v
```

## Model and dataset
Place model weights at `models/best.pt` locally. Do not commit model weights or the dataset.
Record the dataset source, license, exact requested citation, model and library licenses in
`ACKNOWLEDGEMENTS.md`.

## Results and limitations
Add only measured held-out test metrics from the team's actual evaluation.
Severity weights are transparent team-defined rules, not learned factory acceptance labels.
The project is initially trained on one public dataset; real production deployment requires
plant-specific validation.
