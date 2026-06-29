# TESS XGBoost to Gemma Demo

Interactive React prototype for exploring `tess_xgboost_predictions.csv`.

The app loads XGBoost prediction rows, lets you filter/select targets, visualizes
score distribution and top candidates, then sends the selected rows to a Gemma
4 2B analysis bridge. By default the bridge is a local deterministic demo so the
site works without API keys. To connect a real model backend, run Vite with:

```bash
VITE_GEMMA_ENDPOINT=http://localhost:8000/gemma npm run dev
```

The endpoint should accept JSON with `{ model, prompt, rows, stats, threshold }`
and return `{ text: "..." }`.
