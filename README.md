# TESS XGBoost Exoplanet Prototype

This repo contains a simple Kaggle-ready notebook for the Bharatiya Antariksh
Hackathon 2026 exoplanet challenge: detecting exoplanet-like transits from noisy
astronomical light curves.

## Files

- `tess_xgboost_exoplanet_prototype.ipynb` - upload this directly to Kaggle.
- `tess_xgboost_exoplanet_prototype.py` - editable percent-cell source for the
  notebook.
- `tools/make_ipynb.py` - regenerates the notebook from the source file.

## Kaggle Data Layout

Add a Kaggle dataset containing any mix of:

- TESS light-curve FITS files, such as SPOC `*_lc.fits` files.
- CSV light curves with columns like `time,flux` or `time,pdcsap_flux`.
- Optional `labels.csv` with either `tic_id,label` or `file_name,label`.
- Optional compact TIC/xCTL metadata CSV extract with `tic_id` or `ID`.

Example `labels.csv`:

```csv
tic_id,label
123456789,1
987654321,0
```

Labels can be numeric (`1`/`0`) or text such as `planet`, `candidate`,
`confirmed`, `false_positive`, `eclipsing_binary`, `noise`, or `systematic`.

## What the Notebook Does

1. Scans `/kaggle/input` for TESS light curves.
2. Reads FITS/CSV time and flux data.
3. Flattens the light curve with a rolling median.
4. Extracts simple statistics plus Box Least Squares transit features.
5. Trains an XGBoost classifier when two-class labels are available.
6. Falls back to synthetic transit/no-transit data when no labels are attached.
7. Writes:
   - `tess_xgboost_predictions.csv`
   - `tess_xgboost_model.joblib`

This is a prototype baseline, not a publication-grade transit vetting pipeline.
Use labeled TESS Objects of Interest, confirmed planets, and false positives for
a stronger submission.

## React GUI Demo

The interactive dashboard lives in `xgboost-gemma-demo/`. It reads
`outputs/tess_xgboost_predictions.csv`, visualizes score distribution/top
candidates, and sends selected rows to a Gemma 4 2B demo-analysis bridge.

Run it locally:

```powershell
cd xgboost-gemma-demo
pnpm install
pnpm dev
```

By default, the Gemma bridge uses a deterministic local demo adapter. To connect
a real backend, set `VITE_GEMMA_ENDPOINT` before starting Vite; the endpoint
should accept `{ model, prompt, rows, stats, threshold }` and return `{ text }`.
