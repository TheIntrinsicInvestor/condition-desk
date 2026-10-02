"use client";

import { AcvCars, DoorCycles, RailSpectrum, ShmBands } from "./Evidence";
import { IconAlert, IconCheck, IconFault } from "./icons";

const STATUS_CLASS = { high: "status-fault", normal: "status-clear" };

function Chip({ tone, children }) {
  const Icon = tone === "clear" ? IconCheck : tone === "caution" ? IconAlert : IconFault;
  return (
    <span className={`chip chip-${tone}`}>
      <Icon width="12" height="12" />
      {children}
    </span>
  );
}

/* Plain-language reading of the result. This is the part a planner acts on,
   so it names the decision, not the model. */
function interpret(res) {
  const e = res.explain;

  if (e.kind === "rail") {
    const label = res.rows[0].prediction;
    const conf = Math.max(...Object.values(e.confidence));
    const low = conf < 0.6;
    if (label === "Normal") {
      return {
        lines: [
          "No corrugation signature on either rail. Nothing to schedule from this recording.",
          low
            ? "The margin here is narrow, so treat it as inconclusive rather than clean if this stretch has a history."
            : "The reading is unambiguous.",
        ],
        caution: low,
      };
    }
    const side = label;
    // Only claim a dominant wavelength when the contrast actually favours the
    // predicted side. It often does not: the classifier reads 335 features
    // across both vibration and shock channels, and the vibration contrast
    // alone can point the other way. Saying otherwise would be inventing a
    // rationale the data does not support.
    const dir = side === "Side I" ? 1 : -1;
    const best = e.bands
      .filter((b) => b.contrast !== null)
      .reduce((a, b) => (b.contrast * dir > a.contrast * dir ? b : a));
    const supports = best.contrast * dir > 0.3;
    return {
      lines: [
        supports
          ? `Vibration energy runs higher on the ${side} rail around ${Math.round(
              best.wavelength_mm
            )} mm wavelength, which is the length scale corrugation forms at. A rough wheel would raise both rails together, so a one-sided difference points at the rail.`
          : `The ${side} verdict comes from the combined pattern across both rails, including the shock channels and overall levels, rather than from one stand-out wavelength. On the vibration channels alone the two rails are close, so do not expect this chart to show an obvious spike.`,
        `Schedule rail grinding on ${side}. Measured at ${e.speed_mps?.toFixed(1)} m/s.`,
        low
          ? "Confidence is below 60%, so confirm with a second pass before booking a possession."
          : null,
      ].filter(Boolean),
      caution: low || !supports,
    };
  }

  if (e.kind === "shm") {
    const d = res.headline.value;
    const pct = (d * 100).toFixed(1);
    return {
      lines: [
        `This recording accounts for ${pct}% of the detail's modelled fatigue life under Miner's rule. A value of 1.0 means the life is fully consumed.`,
        d > 0.5
          ? "That is over half the budget from one recording. Flag this detail for inspection and check how representative the recording period is."
          : "Within the expected range for a single recording period.",
      ],
      caution: d > 0.5,
    };
  }

  if (e.kind === "door") {
    const ab = res.rows.filter((r) => r.prediction !== "Normal").length;
    const n = res.rows.length;
    const b = e.borderline.length;
    return {
      lines: [
        ab === 0
          ? `All ${n} cycles in this stream drew normal motor effort.`
          : `${ab} of ${n} cycles drew more motor effort than any healthy training cycle, which points to added resistance in the door mechanism. Inspect the runners and check for obstruction.`,
        b > 0
          ? `${b} further ${b === 1 ? "cycle sits" : "cycles sit"} between the two training ranges (${e.borderline.join(", ")}). They are called Normal, but nothing in the training data covers that region, so they are the ones to check by hand.`
          : null,
      ].filter(Boolean),
      caution: b > 0,
    };
  }

  const top = e.cars[0];
  const second = e.cars[1];
  const tight = e.margin !== null && e.spread > 0 && e.margin < e.spread * 0.15;
  return {
    lines: [
      `Car ${top.car} runs the warmest relative to its own cooling setpoint, which is what a unit losing refrigerant does: it cannot reject heat, so it sits above the target it was given.`,
      `Check car ${top.car} first${
        second ? `, then car ${second.car}` : ""
      }. All eight cars are ranked below so the list stays useful if the first is clean.`,
      tight
        ? `The gap to second place is narrow, so treat the top two as joint candidates rather than a single answer.`
        : null,
    ].filter(Boolean),
    caution: tight,
  };
}

export default function Docket({ result, index }) {
  const e = result.explain;
  const read = interpret(result);
  const statusClass = STATUS_CLASS[result.headline.status] ?? "";
  const conf =
    e.kind === "rail" ? Math.max(...Object.values(e.confidence)) : null;

  return (
    <article className="docket docket-enter" key={result.filename}>
      <header className="docket-head">
        <span className="docket-no mono">
          Docket {String(index + 1).padStart(3, "0")}
        </span>
        <span className="docket-file mono">{result.filename}</span>
      </header>

      <div className="finding">
        <h2 className={`verdict serif ${statusClass}`}>
          {result.headline.value}
        </h2>
        <p className="verdict-unit">{result.headline.unit}</p>

        {conf !== null && (
          <div className="confidence">
            <div className="confidence-row">
              <span>Model confidence</span>
              <span className="mono">{(conf * 100).toFixed(0)}%</span>
            </div>
            <div
              className="meter"
              role="meter"
              aria-valuenow={Math.round(conf * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Model confidence"
            >
              <div
                className="meter-fill"
                style={{
                  width: `${conf * 100}%`,
                  background: conf < 0.6 ? "var(--caution)" : "var(--mark)",
                }}
              />
            </div>
            <div style={{ marginTop: 9 }}>
              {conf < 0.6 ? (
                <Chip tone="caution">Marginal call, verify before acting</Chip>
              ) : (
                <Chip tone="clear">Clear margin</Chip>
              )}
            </div>
          </div>
        )}
      </div>

      <section className="section">
        <h3>What this means</h3>
        {read.lines.map((l, i) => (
          <p key={i}>{l}</p>
        ))}
      </section>

      <section className="section">
        <h3>Evidence</h3>

        {e.kind === "rail" && (
          <>
            <div className="readings">
              <div>
                <p className="reading-label">Train speed</p>
                <p className="reading-value mono">
                  {e.speed_mps?.toFixed(1)} m/s
                </p>
              </div>
              {Object.entries(e.confidence).map(([k, v]) => (
                <div key={k}>
                  <p className="reading-label">{k}</p>
                  <p className="reading-value mono">{(v * 100).toFixed(0)}%</p>
                </div>
              ))}
            </div>
            {e.order_valid ? (
              <RailSpectrum bands={e.bands} side={result.rows[0].prediction} />
            ) : (
              <p className="method">{e.note}</p>
            )}
          </>
        )}

        {e.kind === "shm" && (
          <>
            <div className="readings">
              <div>
                <p className="reading-label">Samples</p>
                <p className="reading-value mono">
                  {e.samples.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="reading-label">Rainflow cycles</p>
                <p className="reading-value mono">
                  {e.total_cycles?.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="reading-label">Stress range</p>
                <p className="reading-value mono">
                  {e.stress_range[0]?.toFixed(1)} to{" "}
                  {e.stress_range[1]?.toFixed(1)} MPa
                </p>
              </div>
            </div>
            <ShmBands bands={e.bands} />
          </>
        )}

        {e.kind === "door" && (
          <DoorCycles
            cycles={e.cycles}
            threshold={e.threshold}
            trainNormal={e.train_normal}
            trainAbnormal={e.train_abnormal}
          />
        )}

        {e.kind === "acv" && <AcvCars cars={e.cars} />}
      </section>

      <section className="section">
        <h3>Method</h3>
        <p className="method">{e.method}</p>
      </section>
    </article>
  );
}
