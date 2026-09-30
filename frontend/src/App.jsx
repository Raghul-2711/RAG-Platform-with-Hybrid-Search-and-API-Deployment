import { useEffect, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5000";

function Result({ hit, index }) {
  const [open, setOpen] = useState(false);
  const text = hit.text || "";
  const long = text.length > 350;
  const shown = open || !long ? text : text.slice(0, 350) + "…";
  const meta = hit.metadata || {};
  const where = meta.page_number
    ? `Page ${meta.page_number}`
    : meta.slide_number
      ? `Slide ${meta.slide_number}`
      : null;

  return (
    <div className="result">
      <div className="result-head">
        <span>
          <b>[{index + 1}]</b> {hit.source || "unknown source"}
          {where ? ` · ${where}` : ""}
        </span>
        <span>score {Number(hit.rerank_score ?? hit.score ?? 0).toFixed(3)}</span>
      </div>
      <p className="result-text">{shown}</p>
      {long && (
        <button className="toggle" onClick={() => setOpen(!open)}>
          {open ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [generate, setGenerate] = useState(true);
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [online, setOnline] = useState(false);

  useEffect(() => {
    const check = () =>
      fetch(`${API_BASE}/health`)
        .then((r) => setOnline(r.ok))
        .catch(() => setOnline(false));
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  async function ask() {
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError("");
    setData(null);
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, top_k: topK, generate }),
      });
      const json = await res.json().catch(() => null);
      if (!res.ok) {
        const msg =
          json?.error?.message ||
          json?.message ||
          (typeof json?.error === "string" ? json.error : null) ||
          `Server returned ${res.status}`;
        throw new Error(msg);
      }
      setData(json);
    } catch (e) {
      setError(
        e.message === "Failed to fetch"
          ? "Cannot reach the backend. Is app.py running on port 5000?"
          : e.message
      );
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <span className="eyebrow">ENTERPRISE RAG</span>
          <h1 className="title">RAG Assistant</h1>
          <p className="subtitle">Grounded knowledge responses from your documents</p>
        </div>
        <div className={`status ${online ? "on" : "off"}`}>
          <span className="dot" />
          API {online ? "CONNECTED" : "OFFLINE"}
        </div>
      </header>

      <section className="card">
        <h2>Ask your documents</h2>
        <p className="muted">
          Ask a question and the system will retrieve relevant document context
          before generating a grounded answer.
        </p>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question about your documents…"
        />
        <div className="controls">
          <div className="options">
            <label>
              <input
                type="checkbox"
                checked={generate}
                onChange={(e) => setGenerate(e.target.checked)}
              />
              Generate answer
            </label>
            <label>
              Passages
              <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                {[3, 5, 8, 10].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
          </div>
          <button className="btn" onClick={ask} disabled={loading || !question.trim()}>
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>
      </section>

      {error && <div className="card error">⚠ {error}</div>}

      {data && (
        <>
          {data.answer && (
            <section className="card">
              <h2>Answer</h2>
              <p className="answer">{data.answer}</p>
              <div className="stats">
                {data.generation?.model && <span className="pill">{data.generation.model}</span>}
                {data.retrieval?.retrieval_time_seconds != null && (
                  <span className="pill">retrieval {data.retrieval.retrieval_time_seconds}s</span>
                )}
                {data.generation?.generation_time_seconds > 0 && (
                  <span className="pill">generation {data.generation.generation_time_seconds}s</span>
                )}
                {data.total_time_seconds != null && (
                  <span className="pill">total {data.total_time_seconds}s</span>
                )}
              </div>
            </section>
          )}

          {!data.answer && generate && (
            <div className="card error">
              The backend returned no answer. Check the backend terminal for errors.
            </div>
          )}

          {data.sources?.length > 0 && (
            <section className="card">
              <h2>Sources</h2>
              <div className="source-list">
                {data.sources.map((s, i) => (
                  <span className="chip" key={i}>
                    [{s.rank}] {s.source}
                    {s.location ? ` · ${s.location}` : ""}
                  </span>
                ))}
              </div>
            </section>
          )}

          {data.results?.length > 0 && (
            <section className="card">
              <h2>Retrieved passages</h2>
              {data.results.map((hit, i) => (
                <Result key={hit.chunk_id || i} hit={hit} index={i} />
              ))}
            </section>
          )}

          {data.results?.length === 0 && (
            <div className="card error">No matching passages were found for this question.</div>
          )}
        </>
      )}

      <footer className="footer">Enterprise RAG · {API_BASE}</footer>
    </div>
  );
}
