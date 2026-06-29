import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  Download,
  FileUp,
  Filter,
  Gauge,
  LineChart,
  RefreshCw,
  Search,
  Send,
  Sparkles,
  Star,
  TableProperties,
  XCircle
} from "lucide-react";
import "./styles.css";

const DEFAULT_CSV_PATH = "/tess_xgboost_predictions.csv";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      value += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        i += 1;
      }
      row.push(value);
      if (row.some((item) => item.trim() !== "")) {
        rows.push(row);
      }
      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  if (value || row.length) {
    row.push(value);
    rows.push(row);
  }

  if (!rows.length) {
    return [];
  }

  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((items) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = (items[index] ?? "").trim();
    });
    return normalizePrediction(record);
  });
}

function normalizePrediction(record) {
  const score = Number.parseFloat(record.exoplanet_score ?? record.score ?? "0");
  const label = Number.parseInt(record.predicted_label ?? record.label ?? "0", 10);
  return {
    tic_id: String(record.tic_id ?? record.tic ?? record.id ?? "unknown"),
    file_name: String(record.file_name ?? record.filename ?? ""),
    source: String(record.source ?? "uploaded"),
    exoplanet_score: Number.isFinite(score) ? score : 0,
    predicted_label: Number.isFinite(label) ? label : 0
  };
}

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function scoreBand(score) {
  if (score >= 0.85) return "High";
  if (score >= 0.55) return "Watch";
  if (score >= 0.25) return "Low";
  return "Reject";
}

function bandClass(score) {
  if (score >= 0.85) return "bandHigh";
  if (score >= 0.55) return "bandWatch";
  if (score >= 0.25) return "bandLow";
  return "bandReject";
}

function computeStats(rows) {
  if (!rows.length) {
    return {
      total: 0,
      candidates: 0,
      rejected: 0,
      meanScore: 0,
      topScore: 0,
      highConfidence: 0
    };
  }

  const candidates = rows.filter((row) => row.predicted_label === 1).length;
  const topScore = Math.max(...rows.map((row) => row.exoplanet_score));
  const meanScore =
    rows.reduce((sum, row) => sum + row.exoplanet_score, 0) / rows.length;
  const highConfidence = rows.filter((row) => row.exoplanet_score >= 0.85).length;

  return {
    total: rows.length,
    candidates,
    rejected: rows.length - candidates,
    meanScore,
    topScore,
    highConfidence
  };
}

function buildBins(rows) {
  const bins = Array.from({ length: 10 }, (_, index) => ({
    label: `${index * 10}-${index * 10 + 10}`,
    min: index / 10,
    max: (index + 1) / 10,
    count: 0
  }));

  rows.forEach((row) => {
    const index = Math.min(9, Math.max(0, Math.floor(row.exoplanet_score * 10)));
    bins[index].count += 1;
  });

  return bins;
}

function buildGemmaPrompt(selectedRows, stats) {
  const rows = selectedRows.slice(0, 12).map((row) => ({
    tic_id: row.tic_id,
    score: Number(row.exoplanet_score.toFixed(5)),
    predicted_label: row.predicted_label,
    band: scoreBand(row.exoplanet_score)
  }));

  return [
    "You are Gemma 4 2B acting as a compact astronomy assistant.",
    "Interpret XGBoost exoplanet screening output for a demo dashboard.",
    "Explain confidence, caveats, and next observational checks.",
    "",
    `Dataset summary: ${stats.total} rows, ${stats.candidates} candidate predictions, mean score ${stats.meanScore.toFixed(3)}.`,
    `Rows to analyze: ${JSON.stringify(rows, null, 2)}`
  ].join("\n");
}

async function runGemmaBridge({ rows, stats, threshold }) {
  const prompt = buildGemmaPrompt(rows, stats);
  const endpoint = import.meta.env.VITE_GEMMA_ENDPOINT;

  if (endpoint) {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemma-4-2b-demo",
        prompt,
        rows,
        stats,
        threshold
      })
    });

    if (!response.ok) {
      throw new Error(`Gemma endpoint returned ${response.status}`);
    }

    const data = await response.json();
    return {
      mode: "Live endpoint",
      prompt,
      text: data.text ?? data.response ?? JSON.stringify(data, null, 2)
    };
  }

  await new Promise((resolve) => setTimeout(resolve, 650));
  const topRows = [...rows]
    .sort((a, b) => b.exoplanet_score - a.exoplanet_score)
    .slice(0, 3);
  const high = rows.filter((row) => row.exoplanet_score >= 0.85);
  const watch = rows.filter(
    (row) => row.exoplanet_score >= threshold && row.exoplanet_score < 0.85
  );
  const low = rows.filter((row) => row.exoplanet_score < threshold);
  const strongest = topRows[0];

  const text = [
    `Gemma 4 2B demo interpretation: ${rows.length} selected XGBoost outputs were reviewed.`,
    strongest
      ? `Strongest candidate is ${strongest.tic_id} with score ${pct(strongest.exoplanet_score, 2)}.`
      : "No candidate row is currently selected.",
    `${high.length} rows are high-confidence transit-like candidates, ${watch.length} sit in the review band, and ${low.length} fall below the active threshold.`,
    `Recommended next step: inspect phase-folded light curves for the top ${Math.min(3, Math.max(1, topRows.length))} TICs, then cross-check TIC/xCTL stellar radius and disposition metadata before claiming a planet candidate.`,
    `Caveat: this is a prototype explanation layer over XGBoost scores; it is not a scientific validation report.`
  ].join("\n\n");

  return {
    mode: "Local demo adapter",
    prompt,
    text
  };
}

function App() {
  const fileInputRef = useRef(null);
  const [rows, setRows] = useState([]);
  const [status, setStatus] = useState("Loading bundled predictions CSV...");
  const [query, setQuery] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [sortKey, setSortKey] = useState("score_desc");
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [activeRowId, setActiveRowId] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    fetch(DEFAULT_CSV_PATH)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Bundled CSV not found");
        }
        return response.text();
      })
      .then((text) => {
        const parsed = parseCsv(text);
        setRows(parsed);
        setSelectedIds(new Set(parsed.slice(0, 8).map((row) => row.tic_id)));
        setActiveRowId(parsed[0]?.tic_id ?? null);
        setStatus(`Loaded ${parsed.length} rows from bundled predictions CSV.`);
      })
      .catch((error) => {
        setStatus(`Upload a predictions CSV to begin. ${error.message}`);
      });
  }, []);

  const stats = useMemo(() => computeStats(rows), [rows]);

  const filteredRows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    let next = rows.filter((row) => {
      const matchesQuery =
        !needle ||
        row.tic_id.toLowerCase().includes(needle) ||
        row.file_name.toLowerCase().includes(needle) ||
        row.source.toLowerCase().includes(needle);
      return matchesQuery && row.exoplanet_score >= threshold;
    });

    next = [...next].sort((a, b) => {
      if (sortKey === "score_asc") return a.exoplanet_score - b.exoplanet_score;
      if (sortKey === "tic") return a.tic_id.localeCompare(b.tic_id);
      if (sortKey === "label") return b.predicted_label - a.predicted_label;
      return b.exoplanet_score - a.exoplanet_score;
    });

    return next;
  }, [rows, query, sortKey, threshold]);

  const selectedRows = useMemo(() => {
    const byId = new Map(rows.map((row) => [row.tic_id, row]));
    return [...selectedIds].map((id) => byId.get(id)).filter(Boolean);
  }, [rows, selectedIds]);

  const activeRow = useMemo(
    () => rows.find((row) => row.tic_id === activeRowId) ?? filteredRows[0] ?? rows[0],
    [activeRowId, filteredRows, rows]
  );

  const bins = useMemo(() => buildBins(rows), [rows]);
  const maxBin = Math.max(1, ...bins.map((bin) => bin.count));
  const topRows = useMemo(
    () => [...rows].sort((a, b) => b.exoplanet_score - a.exoplanet_score).slice(0, 8),
    [rows]
  );

  function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseCsv(String(reader.result ?? ""));
      setRows(parsed);
      setSelectedIds(new Set(parsed.slice(0, 8).map((row) => row.tic_id)));
      setActiveRowId(parsed[0]?.tic_id ?? null);
      setAnalysis(null);
      setStatus(`Loaded ${parsed.length} rows from ${file.name}.`);
    };
    reader.readAsText(file);
  }

  function toggleSelected(row) {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(row.tic_id)) {
        next.delete(row.tic_id);
      } else {
        next.add(row.tic_id);
      }
      return next;
    });
    setActiveRowId(row.tic_id);
  }

  async function handleAnalyze() {
    const inputRows = selectedRows.length ? selectedRows : filteredRows.slice(0, 8);
    setIsAnalyzing(true);
    setAnalysis(null);
    try {
      const result = await runGemmaBridge({
        rows: inputRows,
        stats,
        threshold
      });
      setAnalysis(result);
    } catch (error) {
      setAnalysis({
        mode: "Gemma bridge error",
        prompt: buildGemmaPrompt(inputRows, stats),
        text: error.message
      });
    } finally {
      setIsAnalyzing(false);
    }
  }

  function downloadSelection() {
    const header = "tic_id,file_name,source,exoplanet_score,predicted_label\n";
    const body = selectedRows
      .map((row) =>
        [
          row.tic_id,
          row.file_name,
          row.source,
          row.exoplanet_score,
          row.predicted_label
        ]
          .map((value) => `"${String(value).replaceAll('"', '""')}"`)
          .join(",")
      )
      .join("\n");

    const blob = new Blob([header + body], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "selected_tess_predictions.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="appShell">
      <header className="topBar">
        <div className="brandBlock">
          <div className="brandMark">
            <Sparkles size={22} />
          </div>
          <div>
            <h1>TESS XGBoost to Gemma 4 2B</h1>
            <p>{status}</p>
          </div>
        </div>
        <div className="topActions">
          <button className="iconButton" type="button" onClick={() => fileInputRef.current?.click()}>
            <FileUp size={18} />
            <span>CSV</span>
          </button>
          <button className="iconButton primary" type="button" onClick={handleAnalyze}>
            {isAnalyzing ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}
            <span>{isAnalyzing ? "Analyzing" : "Gemma"}</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="hiddenInput"
            onChange={handleUpload}
          />
        </div>
      </header>

      <section className="metricStrip" aria-label="Prediction metrics">
        <Metric icon={<TableProperties size={19} />} label="Rows" value={stats.total} />
        <Metric icon={<Star size={19} />} label="Candidates" value={stats.candidates} />
        <Metric icon={<XCircle size={19} />} label="Rejected" value={stats.rejected} />
        <Metric icon={<Gauge size={19} />} label="Mean Score" value={pct(stats.meanScore)} />
        <Metric icon={<CheckCircle2 size={19} />} label="High Confidence" value={stats.highConfidence} />
      </section>

      <section className="workspaceGrid">
        <section className="leftPane" aria-label="Prediction explorer">
          <div className="toolbar">
            <label className="searchBox">
              <Search size={17} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search TIC, file, source"
              />
            </label>
            <label className="selectBox">
              <Filter size={17} />
              <select value={sortKey} onChange={(event) => setSortKey(event.target.value)}>
                <option value="score_desc">Score high to low</option>
                <option value="score_asc">Score low to high</option>
                <option value="tic">TIC ID</option>
                <option value="label">Predicted label</option>
              </select>
              <ChevronDown size={16} />
            </label>
          </div>

          <div className="thresholdRow">
            <div>
              <span>Threshold</span>
              <strong>{threshold.toFixed(2)}</strong>
            </div>
            <input
              type="range"
              min="0"
              max="0.99"
              step="0.01"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
          </div>

          <div className="tableHeader">
            <span>{filteredRows.length} visible rows</span>
            <button
              className="ghostButton"
              type="button"
              onClick={downloadSelection}
              disabled={!selectedRows.length}
            >
              <Download size={17} />
              <span>Selection</span>
            </button>
          </div>

          <div className="predictionTable" role="table">
            <div className="tableRow headerRow" role="row">
              <span>Pick</span>
              <span>TIC</span>
              <span>Score</span>
              <span>Band</span>
              <span>Label</span>
            </div>
            {filteredRows.slice(0, 60).map((row) => (
              <button
                className={`tableRow ${activeRow?.tic_id === row.tic_id ? "active" : ""}`}
                key={row.tic_id}
                type="button"
                role="row"
                onClick={() => toggleSelected(row)}
              >
                <span className={`pickDot ${selectedIds.has(row.tic_id) ? "picked" : ""}`} />
                <span className="mono">{row.tic_id}</span>
                <span>{pct(row.exoplanet_score, 2)}</span>
                <span className={`bandPill ${bandClass(row.exoplanet_score)}`}>
                  {scoreBand(row.exoplanet_score)}
                </span>
                <span>{row.predicted_label ? "Candidate" : "Reject"}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="rightPane" aria-label="Analysis and visualizations">
          <div className="panelHero">
            <div>
              <h2>{activeRow ? activeRow.tic_id : "No row selected"}</h2>
              <p>{activeRow?.file_name || "Upload or select a prediction row"}</p>
            </div>
            <div className={`scoreBadge ${activeRow ? bandClass(activeRow.exoplanet_score) : ""}`}>
              {activeRow ? pct(activeRow.exoplanet_score, 2) : "0.0%"}
            </div>
          </div>

          <div className="vizGrid">
            <div className="vizBlock">
              <div className="blockTitle">
                <BarChart3 size={18} />
                <span>Score Distribution</span>
              </div>
              <div className="histogram">
                {bins.map((bin) => (
                  <div className="histColumn" key={bin.label}>
                    <div
                      className="histBar"
                      style={{ height: `${Math.max(4, (bin.count / maxBin) * 100)}%` }}
                      title={`${bin.label}%: ${bin.count}`}
                    />
                    <span>{bin.label}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="vizBlock">
              <div className="blockTitle">
                <LineChart size={18} />
                <span>Top Candidates</span>
              </div>
              <div className="rankList">
                {topRows.map((row, index) => (
                  <button
                    type="button"
                    className="rankItem"
                    key={row.tic_id}
                    onClick={() => {
                      setActiveRowId(row.tic_id);
                      setSelectedIds((current) => new Set(current).add(row.tic_id));
                    }}
                  >
                    <span>{index + 1}</span>
                    <strong>{row.tic_id}</strong>
                    <div>
                      <i style={{ width: `${Math.max(3, row.exoplanet_score * 100)}%` }} />
                    </div>
                    <em>{pct(row.exoplanet_score, 1)}</em>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <section className="gemmaPanel">
            <div className="gemmaHeader">
              <div className="blockTitle">
                <BrainCircuit size={19} />
                <span>Gemma 4 2B Analysis Bridge</span>
              </div>
              <span>{selectedRows.length || filteredRows.slice(0, 8).length} rows queued</span>
            </div>

            <div className="promptBox">
              <pre>{buildGemmaPrompt(selectedRows.length ? selectedRows : filteredRows.slice(0, 8), stats)}</pre>
            </div>

            <div className="answerBox">
              {isAnalyzing ? (
                <div className="thinking">
                  <RefreshCw className="spin" size={20} />
                  <span>Sending XGBoost results to the Gemma demo adapter...</span>
                </div>
              ) : analysis ? (
                <>
                  <div className="modeLine">
                    <Activity size={17} />
                    <span>{analysis.mode}</span>
                  </div>
                  <p>{analysis.text}</p>
                </>
              ) : (
                <p>
                  Press the Gemma button to generate an explanation from the selected
                  XGBoost predictions. Set `VITE_GEMMA_ENDPOINT` to call a live
                  backend; otherwise this demo uses a local adapter.
                </p>
              )}
            </div>
          </section>
        </section>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }) {
  return (
    <div className="metricItem">
      <div>{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
