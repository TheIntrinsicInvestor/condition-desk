"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Docket from "@/components/Docket";
import { IconAlert, IconDownload, IconUpload, SUBSYSTEM_ICONS } from "@/components/icons";

const API = process.env.NEXT_PUBLIC_API_BASE || "";

const SUBSYSTEMS = [
  { id: "shm", label: "Structural Health", accepts: ".csv",
    brief: "A single-column stress recording from a bogie strain gauge.",
    unit: "cumulative damage" },
  { id: "acv", label: "Air Conditioning", accepts: ".xlsx",
    brief: "An Excel workbook of per-car temperature and running-mode logs.",
    unit: "faulty car" },
  { id: "door", label: "Doors", accepts: ".csv",
    brief: "A continuous stream of door motor readings, many cycles back to back.",
    unit: "abnormal cycles" },
  { id: "rail", label: "Rail Corrugation", accepts: ".csv",
    brief: "A one-second, 10 kHz recording from the axle-box accelerometers.",
    unit: "rail condition" },
];

function verdictOf(item) {
  if (!item.result) return null;
  // A door stream yields many rows, so the first row's label is not the
  // finding. Use the headline the docket itself shows.
  if (item.result.rows.length !== 1) return item.result.headline.value;
  const r = item.result.rows[0];
  return r.prediction ?? r.ranked_cars?.split("|")[0] ?? item.result.headline.value;
}

function toneOf(item) {
  if (!item.result) return "clear";
  return item.result.headline.status === "high" ? "fault" : "clear";
}

export default function Page() {
  const [subsystem, setSubsystem] = useState("rail");
  const [items, setItems] = useState({ shm: [], acv: [], door: [], rail: [] });
  const [activeIndex, setActiveIndex] = useState(0);
  const [over, setOver] = useState(false);
  const [exporting, setExporting] = useState(false);
  const queueRef = useRef(false);
  const pendingRef = useRef([]);
  const inputRef = useRef(null);

  const meta = SUBSYSTEMS.find((s) => s.id === subsystem);
  const list = items[subsystem];
  const done = list.filter((i) => i.result);
  const active = done[activeIndex] ?? done[done.length - 1] ?? null;

  useEffect(() => {
    setActiveIndex(Math.max(done.length - 1, 0));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done.length, subsystem]);

  const push = useCallback((sub, updater) => {
    setItems((prev) => ({ ...prev, [sub]: updater(prev[sub]) }));
  }, []);

  // The queue is held in a ref and drained in order. It must NOT be read by
  // side-effecting inside a setState updater: React may invoke an updater more
  // than once, and the second pass finds nothing pending, which silently
  // strands the in-flight item on "working" forever.
  const runQueue = useCallback(async () => {
    if (queueRef.current) return;
    queueRef.current = true;
    try {
      while (pendingRef.current.length) {
        const job = pendingRef.current.shift();
        push(job.sub, (arr) =>
          arr.map((i) => (i.id === job.id ? { ...i, status: "working" } : i))
        );

        const body = new FormData();
        body.append("file", job.file);
        try {
          const res = await fetch(`${API}/api/predict/${job.sub}`, {
            method: "POST",
            body,
          });
          const json = await res.json();
          if (!res.ok) throw new Error(json.detail || `Request failed (${res.status})`);
          push(job.sub, (arr) =>
            arr.map((i) =>
              i.id === job.id ? { ...i, status: "done", result: json } : i
            )
          );
        } catch (err) {
          push(job.sub, (arr) =>
            arr.map((i) =>
              i.id === job.id
                ? { ...i, status: "error", error: String(err.message || err) }
                : i
            )
          );
        }
      }
    } finally {
      queueRef.current = false;
    }
  }, [push]);

  const addFiles = useCallback(
    (fileList) => {
      const files = Array.from(fileList || []);
      if (!files.length) return;
      const sub = subsystem;
      const jobs = files.map((file, n) => ({
        id: `${Date.now()}-${n}-${file.name}`,
        sub,
        file,
        name: file.name,
        status: "queued",
        result: null,
        error: null,
      }));
      pendingRef.current.push(...jobs);
      push(sub, (arr) => [...arr, ...jobs]);
      runQueue();
    },
    [subsystem, push, runQueue]
  );

  const rows = useMemo(
    () => done.flatMap((i) => i.result.rows),
    [done]
  );

  async function exportCsv() {
    if (!rows.length) return;
    setExporting(true);
    try {
      const res = await fetch(`${API}/api/export/${subsystem}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows }),
      });
      const text = await res.text();
      const blob = new Blob([text], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${subsystem}_predictions.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }

  const working = list.some((i) => i.status === "working" || i.status === "queued");

  return (
    <div className="shell">
      <div className="head">
        <header className="topbar">
          <div className="wordmark">
            Condition Desk <span>train condition monitoring</span>
          </div>
          <div className="topbar-spacer" />
          <button
            className="btn"
            onClick={exportCsv}
            disabled={!rows.length || exporting}
            title={rows.length ? undefined : "Process a file first to export"}
          >
            <IconDownload />
            {exporting
              ? "Preparing"
              : rows.length
              ? `Export ${rows.length} ${rows.length === 1 ? "row" : "rows"}`
              : "Export"}
          </button>
        </header>

        <div className="tabbar">
          <div className="tabs" role="tablist" aria-label="Subsystem">
            {SUBSYSTEMS.map((s) => {
              const Icon = SUBSYSTEM_ICONS[s.id];
              const n = items[s.id].filter((i) => i.result).length;
              return (
                <button
                  key={s.id}
                  role="tab"
                  aria-selected={subsystem === s.id}
                  className="tab"
                  onClick={() => setSubsystem(s.id)}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
                    <Icon width="14" height="14" />
                    {s.label}
                    {n > 0 && <span className="mono" style={{ color: "var(--ink-faint)" }}>{n}</span>}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="main">
        <aside className="rail">
          <p className="lede">{meta.brief}</p>

          <label
            className="dropzone"
            data-over={over}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              addFiles(e.dataTransfer.files);
            }}
          >
            <IconUpload width="20" height="20" />
            <p className="dropzone-title">Drop {meta.accepts} files here</p>
            <p className="dropzone-hint">or click to browse. Several at once is fine.</p>
            <input
              ref={inputRef}
              type="file"
              accept={meta.accepts}
              multiple
              onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }}
            />
          </label>

          {list.length > 0 && (
            <>
              <div className="batch-head">
                <h2>This batch</h2>
                <span className="batch-count mono">
                  {done.length}/{list.length}
                </span>
              </div>
              <ul className="spines">
                {list.map((item) => {
                  const idx = done.indexOf(item);
                  const tone = toneOf(item);
                  return (
                    <li key={item.id}>
                      <button
                        className="spine"
                        aria-current={item === active}
                        onClick={() => idx >= 0 && setActiveIndex(idx)}
                        disabled={!item.result}
                      >
                        {item.status === "done" ? (
                          <span
                            aria-hidden="true"
                            style={{
                              width: 9, height: 9, borderRadius: 2,
                              background: tone === "fault" ? "var(--fault)" : "var(--clear)",
                              display: "inline-block",
                            }}
                          />
                        ) : item.status === "error" ? (
                          <IconAlert width="12" height="12" style={{ color: "var(--fault)" }} />
                        ) : (
                          <span className="spinner" />
                        )}
                        <span className="spine-name">{item.name}</span>
                        <span className={item.result ? "spine-verdict" : "spine-pending"}>
                          {item.status === "done"
                            ? verdictOf(item)
                            : item.status === "error"
                            ? "failed"
                            : item.status === "working"
                            ? "reading"
                            : "queued"}
                        </span>
                      </button>
                      {item.error && (
                        <div className="errorbox" style={{ margin: "6px 0 10px" }}>
                          <IconAlert width="14" height="14" />
                          <span>{item.error}</span>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </aside>

        <main className="stage">
          {active ? (
            <Docket result={active.result} index={done.indexOf(active)} />
          ) : (
            <div className="empty">
              <h1 className="serif">
                {working ? "Reading your file" : "Every finding, with the evidence behind it."}
              </h1>
              <p>
                Condition Desk reads the four condition-monitoring feeds that come off a
                train and its track, and turns each file into a maintenance record: what
                was found, what the model actually measured, and what to do next.
              </p>
              <p>
                Choose a subsystem above, then drop in a file. {meta.brief}
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
