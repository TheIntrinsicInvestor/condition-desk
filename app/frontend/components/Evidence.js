// Evidence charts. Authored SVG driven by the API response, no chart library.
// Each one shows the quantity the model actually read, not a decorative proxy.

const INK = "var(--ink)";
const SOFT = "var(--ink-soft)";
const FAINT = "var(--ink-faint)";
const RULE = "var(--rule)";

function Axis({ x1, x2, y }) {
  return <line x1={x1} x2={x2} y1={y} y2={y} stroke={RULE} strokeWidth="1" />;
}

/* ------------------------------------------------------------------ rail */

export function RailSpectrum({ bands, side }) {
  const usable = bands.filter((b) => b.contrast !== null);
  if (!usable.length) return null;

  // The discriminative quantity is the difference between the two rails at the
  // same wavelength, not either rail's level, which rises and falls with speed
  // and track. Plotting the difference directly is both more legible than two
  // near-identical curves and more honest about what separates the classes.
  const W = 680, H = 210, PAD_L = 46, PAD_R = 10, PAD_T = 16, PAD_B = 38;
  const mag = Math.max(...usable.map((b) => Math.abs(b.contrast)), 0.1);
  const bw = (W - PAD_L - PAD_R) / usable.length;
  const mid = PAD_T + (H - PAD_T - PAD_B) / 2;
  const half = (H - PAD_T - PAD_B) / 2;
  const y = (v) => mid - (v / mag) * half;
  const ticks = [0, 6, 12, 18, 24].filter((i) => i < usable.length);
  const peak = usable.reduce((a, b) =>
    Math.abs(b.contrast) > Math.abs(a.contrast) ? b : a
  );

  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
           aria-label="Difference in vibration energy between the two rails, by wavelength">
        {usable.map((b, i) => {
          const yy = y(b.contrast);
          const h = Math.abs(yy - mid);
          const up = b.contrast > 0;
          return (
            <rect key={i} x={PAD_L + bw * i + 1} y={up ? yy : mid}
                  width={Math.max(bw - 2, 1)} height={Math.max(h, 0.5)}
                  fill={up ? "var(--fault)" : "var(--mark)"}
                  opacity={b === peak ? 0.95 : 0.42}>
              <title>{`${Math.round(b.wavelength_mm)} mm: ${
                up ? "Side I" : "Side II"} higher by ${Math.abs(b.contrast).toFixed(2)}`}</title>
            </rect>
          );
        })}
        <line x1={PAD_L} x2={W - PAD_R} y1={mid} y2={mid} stroke={INK} strokeWidth="1" />
        {ticks.map((i) => (
          <text key={i} x={PAD_L + bw * (i + 0.5)} y={H - PAD_B + 16}
                fontSize="10.5" fill={FAINT} textAnchor="middle" className="mono">
            {usable[i].wavelength_mm >= 100
              ? Math.round(usable[i].wavelength_mm)
              : usable[i].wavelength_mm.toFixed(1)}
          </text>
        ))}
        <text x={PAD_L - 9} y={PAD_T + 9} fontSize="10.5" fill="var(--fault)" textAnchor="end">
          Side I
        </text>
        <text x={PAD_L - 9} y={H - PAD_B - 2} fontSize="10.5" fill="var(--mark)" textAnchor="end">
          Side II
        </text>
        <text x={PAD_L} y={H - 6} fontSize="10.5" fill={FAINT}>
          wavelength (mm), long to short
        </text>
      </svg>
      <figcaption style={{ fontSize: 12.5, color: SOFT, marginTop: 4, maxWidth: "68ch" }}>
        Bars above the line mean the Side I rail carries more energy at that wavelength, below
        means Side II. Largest difference: {Math.round(peak.wavelength_mm)} mm, favouring{" "}
        {peak.contrast > 0 ? "Side I" : "Side II"} by {Math.abs(peak.contrast).toFixed(2)} in
        log energy. This shows the vibration channels only; the classifier also reads the shock
        channels and absolute levels, so it does not decide on this view alone.
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------- shm */

export function ShmBands({ bands }) {
  const W = 680, H = 198, PAD_L = 52, PAD_R = 8, PAD_T = 16, PAD_B = 40;
  const hi = Math.max(...bands.map((b) => b.damage_pct), 1);
  const bw = (W - PAD_L - PAD_R) / bands.length;

  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
           aria-label="Share of total fatigue damage contributed by each stress amplitude band">
        <Axis x1={PAD_L} x2={W - PAD_R} y={H - PAD_B} />
        {bands.map((b, i) => {
          const h = (b.damage_pct / hi) * (H - PAD_T - PAD_B);
          return (
            <rect key={i} x={PAD_L + bw * i + 1.5} y={H - PAD_B - h}
                  width={Math.max(bw - 3, 1)} height={h}
                  fill={b.damage_pct > 20 ? "var(--fault)" : "var(--ink)"}
                  opacity={b.damage_pct > 20 ? 1 : 0.5} />
          );
        })}
        {bands.map((b, i) =>
          i % 3 === 0 ? (
            <text key={i} x={PAD_L + bw * (i + 0.5)} y={H - PAD_B + 15}
                  fontSize="10.5" fill={FAINT} textAnchor="middle" className="mono">
              {Math.round(b.from)}
            </text>
          ) : null
        )}
        <text x={PAD_L} y={H - 6} fontSize="10.5" fill={FAINT}>
          stress amplitude (MPa)
        </text>
        <text x={PAD_L - 9} y={PAD_T + 4} fontSize="10.5" fill={FAINT} textAnchor="end">
          {Math.round(hi)}%
        </text>
        <text x={PAD_L - 9} y={H - PAD_B} fontSize="10.5" fill={FAINT} textAnchor="end">
          0%
        </text>
        <text x={12} y={(PAD_T + H - PAD_B) / 2} fontSize="10.5" fill={FAINT}
              textAnchor="middle" transform={`rotate(-90 12 ${(PAD_T + H - PAD_B) / 2})`}>
          share of damage
        </text>
      </svg>
      <figcaption style={{ fontSize: 12.5, color: SOFT, marginTop: 4, maxWidth: "68ch" }}>
        Damage rises as the fifth power of amplitude, so a small number of large cycles
        accounts for most of the total. Bands contributing over 20% are marked.
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ door */

export function DoorCycles({ cycles, threshold, trainNormal, trainAbnormal }) {
  const W = 680, H = 186, PAD_L = 8, PAD_R = 8, PAD_T = 26, PAD_B = 56;
  const vals = cycles.map((c) => c.value);
  const lo = Math.min(...vals, trainNormal[0]) - 60;
  const hi = Math.max(...vals, trainAbnormal[1]) + 60;
  const x = (v) => PAD_L + ((v - lo) / (hi - lo)) * (W - PAD_L - PAD_R);
  const band = (a, b) => ({ x: x(a), w: Math.max(x(b) - x(a), 1) });
  const nb = band(trainNormal[0], trainNormal[1]);
  const ab = band(trainAbnormal[0], trainAbnormal[1]);
  const top = PAD_T, h = H - PAD_T - PAD_B;

  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
           aria-label="Every cycle placed against the decision threshold, with the training ranges behind">
        <rect x={nb.x} y={top} width={nb.w} height={h} fill="var(--clear-wash)" />
        <rect x={ab.x} y={top} width={ab.w} height={h} fill="var(--fault-wash)" />
        <text x={nb.x + nb.w / 2} y={top - 9} fontSize="10.5" fill="var(--clear)"
              textAnchor="middle">training Normal</text>
        <text x={ab.x + ab.w / 2} y={top - 9} fontSize="10.5" fill="var(--fault)"
              textAnchor="middle">training Abnormal</text>
        <line x1={x(threshold)} x2={x(threshold)} y1={top - 3} y2={top + h + 3}
              stroke={INK} strokeWidth="1.5" strokeDasharray="3 3" />
        <text x={x(threshold)} y={H - PAD_B + 26} fontSize="10.5" fill={INK}
              textAnchor="middle" className="mono">threshold {Math.round(threshold)}</text>
        {cycles.map((c) => {
          const marginal = Math.abs(c.margin) < 0.5;
          return (
            <circle key={c.index} cx={x(c.value)}
                    cy={top + h / 2 + ((c.index * 37) % 11) - 5} r="3.6"
                    fill={c.prediction === "Normal" ? "var(--clear)" : "var(--fault)"}
                    stroke={marginal ? "var(--caution)" : "none"}
                    strokeWidth={marginal ? 2 : 0}
                    opacity="0.85">
              <title>{`Cycle ${c.index} (${c.operation}) — ${c.prediction}`}</title>
            </circle>
          );
        })}
        <Axis x1={PAD_L} x2={W - PAD_R} y={top + h + 10} />
        <text x={PAD_L} y={H - 4} fontSize="10.5" fill={FAINT}>
          motor voltage integral, one dot per cycle
        </text>
      </svg>
      <figcaption style={{ fontSize: 12.5, color: SOFT, marginTop: 4, maxWidth: "68ch" }}>
        Cycles ringed in amber fall inside the gap between the two training ranges, a region
        containing none of the 110 labelled training cycles. They are the calls worth checking.
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------- acv */

export function AcvCars({ cars }) {
  const vals = cars.map((c) => c.score).filter((v) => v !== null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1;

  return (
    <table className="tbl">
      <thead>
        <tr>
          <th style={{ width: 46 }}>Rank</th>
          <th style={{ width: 62 }}>Car</th>
          <th>Excess over its own setpoint</th>
          <th className="num" style={{ width: 86 }}>Reading</th>
        </tr>
      </thead>
      <tbody>
        {cars.map((c) => {
          const pct = c.score === null ? 0 : ((c.score - lo) / span) * 100;
          const top = c.rank === 1;
          return (
            <tr key={c.car} className={top ? "row-top" : undefined}>
              <td className="mono">{c.rank}</td>
              <td className="mono">{c.car}</td>
              <td>
                <div style={{ background: "var(--rule-soft)", borderRadius: 2, height: 10 }}>
                  <div style={{
                    width: `${Math.max(pct, 1.5)}%`, height: "100%", borderRadius: 2,
                    background: top ? "var(--fault)" : "var(--ink)",
                    opacity: top ? 1 : 0.34,
                  }} />
                </div>
              </td>
              <td className="num mono">
                {c.score === null ? "n/a" : `${c.score > 0 ? "+" : ""}${c.score.toFixed(2)}`}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
