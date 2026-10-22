import { useStore } from './store';
import type { NodeRate, LinkBaseline } from './types';

function rateColor(rate: number, expected: number): string {
  if (expected === 0) return '#666';
  const ratio = rate / expected;
  if (ratio < 0.3) return '#ef4444';
  if (ratio < 0.6) return '#f59e0b';
  return '#22c55e';
}

function captureColor(rate: number): string {
  if (rate < 0.3) return '#ef4444';
  if (rate < 0.6) return '#f59e0b';
  return '#22c55e';
}

function pctBar(fraction: number, color: string) {
  const width = Math.min(100, fraction * 100);
  return (
    <div style={{ height: 8, borderRadius: 3, background: '#f0f0f0', overflow: 'hidden', minWidth: 40 }}>
      <div style={{ height: '100%', width: `${width}%`, background: color, borderRadius: 3, transition: 'width 0.3s' }} />
    </div>
  );
}

interface LinkPair {
  a: string;  // sorted first MAC
  b: string;  // sorted second MAC
  ab: LinkBaseline | null;  // a->b direction
  ba: LinkBaseline | null;  // b->a direction
}

function LinkCalibrationTable({ baselines, target, macLabel }: {
  baselines: Record<string, LinkBaseline>;
  target: number;
  macLabel: (mac: string) => string;
}) {
  const entries = Object.values(baselines);
  if (entries.length === 0) return null;

  // Group bidirectional links
  const pairMap = new Map<string, LinkPair>();
  for (const bl of entries) {
    const [a, b] = [bl.tx, bl.rx].sort();
    const key = `${a}|${b}`;
    if (!pairMap.has(key)) pairMap.set(key, { a, b, ab: null, ba: null });
    const pair = pairMap.get(key)!;
    if (bl.tx === a) pair.ab = bl; else pair.ba = bl;
  }

  // Stable lexicographic sort by the human-readable labels so rows do not
  // reshuffle every time a count ticks over during calibration.
  const pairs = [...pairMap.values()].sort((x, y) => {
    const xa = macLabel(x.a), xb = macLabel(x.b);
    const ya = macLabel(y.a), yb = macLabel(y.b);
    return xa.localeCompare(ya) || xb.localeCompare(yb);
  });

  const thBase = { padding: '4px 6px', fontSize: 11, color: '#888', whiteSpace: 'nowrap' as const };
  const thR = { ...thBase, textAlign: 'right' as const };
  const thC = { ...thBase, textAlign: 'center' as const };
  const mono = { fontFamily: 'monospace', fontSize: 12 };
  const tdR = { textAlign: 'right' as const, padding: '4px 6px', ...mono };
  const tdC = { textAlign: 'center' as const, padding: '4px 6px', ...mono };

  function progressColor(count: number) {
    if (count >= target) return '#22c55e';
    if (count >= target * 0.5) return '#f59e0b';
    return '#ef4444';
  }

  function asymmetry(ab: LinkBaseline | null, ba: LinkBaseline | null): string {
    if (!ab || !ba || ab.mean === 0 || ba.mean === 0) return '—';
    const ratio = ab.mean / ba.mean;
    if (ratio > 1) return `${ratio.toFixed(2)}×`;
    return `${(1 / ratio).toFixed(2)}×`;
  }

  function asymmetryColor(ab: LinkBaseline | null, ba: LinkBaseline | null): string {
    if (!ab || !ba || ab.mean === 0 || ba.mean === 0) return '#888';
    const ratio = Math.max(ab.mean / ba.mean, ba.mean / ab.mean);
    if (ratio > 3) return '#ef4444';
    if (ratio > 1.5) return '#f59e0b';
    return '#22c55e';
  }

  function dirCell(bl: LinkBaseline | null) {
    if (!bl) return { count: 0, mean: '—', std: '—' };
    return {
      count: bl.count,
      mean: bl.mean.toFixed(2),
      std: bl.std.toFixed(2),
    };
  }

  return (
    <>
      <div className="section-title">Per-link calibration</div>
      <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 0, fontSize: 13, minWidth: 600 }}>
        <thead>
          <tr style={{ borderBottom: '2px solid #ddd' }}>
            <th style={{ ...thBase, textAlign: 'left' }}>Link</th>
            <th colSpan={3} style={thC}>A → B</th>
            <th colSpan={3} style={thC}>B → A</th>
            <th style={thR}>Asym</th>
          </tr>
          <tr style={{ borderBottom: '1px solid #eee' }}>
            <th style={{ ...thBase, textAlign: 'left' }}></th>
            <th style={thR}>n</th>
            <th style={thR}>μ</th>
            <th style={thR}>σ</th>
            <th style={thR}>n</th>
            <th style={thR}>μ</th>
            <th style={thR}>σ</th>
            <th style={thR}></th>
          </tr>
        </thead>
        <tbody>
          {pairs.map((p) => {
            const ab = dirCell(p.ab);
            const ba = dirCell(p.ba);
            const minCount = Math.min(p.ab?.count ?? 0, p.ba?.count ?? 0);
            const done = minCount >= target;
            return (
              <tr key={`${p.a}|${p.b}`} style={{ borderBottom: '1px solid #f0f0f0', opacity: done ? 0.6 : 1 }}>
                <td style={{ padding: '4px 6px', fontSize: 12, whiteSpace: 'nowrap' }}>
                  <span style={mono}>{macLabel(p.a)}</span>
                  <span style={{ color: '#bbb', margin: '0 2px' }}>⇄</span>
                  <span style={mono}>{macLabel(p.b)}</span>
                </td>
                <td style={{ ...tdR, color: progressColor(p.ab?.count ?? 0) }}>
                  {ab.count}{!done && <span style={{ color: '#ccc', fontSize: 10 }}>/{target}</span>}
                </td>
                <td style={tdR}>{ab.mean}</td>
                <td style={tdR}>{ab.std}</td>
                <td style={{ ...tdR, color: progressColor(p.ba?.count ?? 0) }}>
                  {ba.count}{!done && <span style={{ color: '#ccc', fontSize: 10 }}>/{target}</span>}
                </td>
                <td style={tdR}>{ba.mean}</td>
                <td style={tdR}>{ba.std}</td>
                <td style={{ ...tdC, color: asymmetryColor(p.ab, p.ba), fontWeight: 600 }}>
                  {asymmetry(p.ab, p.ba)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

export function StatsPanel() {
  const { stats, nodes } = useStore();
  const macLabel = (mac: string) => {
    const node = nodes.find((n) => n.mac === mac);
    const suffix = mac.slice(-5);
    return node?.label ? `${node.label} (${suffix})` : suffix;
  };
  const { tdma } = stats;

  const toggleFreeze = async () => {
    try {
      await fetch('/api/baseline/freeze', { method: 'POST' });
    } catch { /* WS will update state */ }
  };

  // Stable sort by label/MAC so the node rows do not reshuffle as live
  // avg_link_rate values tick up and down.
  const entries = Object.entries(stats.node_rates) as [string, NodeRate][];
  const sortedEntries = entries.sort(([macA], [macB]) => macLabel(macA).localeCompare(macLabel(macB)));

  const expected = stats.expected_link_rate;

  const thBase = { padding: '4px 8px', fontSize: 12, color: '#888' };
  const thR = { ...thBase, textAlign: 'right' as const };
  const td = { textAlign: 'right' as const, padding: '6px 8px', fontFamily: 'monospace', fontSize: 13 };
  const labelTd = { padding: '4px 12px 4px 0', color: '#666' };

  return (
    <div style={{ maxWidth: 900, display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card card-pad">
      <div className="section-title">Calibration</div>
      <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 0, fontSize: 14 }}>
        <tbody>
          <tr>
            <td style={labelTd}>Status</td>
            <td style={{ padding: 4, fontWeight: 500 }}>
              {stats.calibrating ? (
                <span style={{ color: '#f59e0b' }}>
                  Calibrating... {stats.calibration_elapsed_s != null && `(${stats.calibration_elapsed_s.toFixed(0)}s)`}
                </span>
              ) : (
                <span style={{ color: '#22c55e' }}>Complete</span>
              )}
            </td>
          </tr>
          {!stats.calibrating && (
            <tr>
              <td style={labelTd}>Baseline</td>
              <td style={{ padding: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ color: stats.baseline_frozen ? '#3b82f6' : '#888' }}>
                  {stats.baseline_frozen ? 'Frozen' : 'Updating'}
                </span>
                <button
                  onClick={toggleFreeze}
                  style={{
                    padding: '2px 10px',
                    fontSize: 12,
                    border: '1px solid #ccc',
                    borderRadius: 4,
                    background: stats.baseline_frozen ? '#dbeafe' : '#f5f5f5',
                    cursor: 'pointer',
                  }}
                >
                  {stats.baseline_frozen ? 'Unfreeze' : 'Freeze'}
                </button>
              </td>
            </tr>
          )}
          {stats.calibration_duration_s != null && (
            <tr>
              <td style={labelTd}>Duration</td>
              <td style={{ padding: 4 }}>{stats.calibration_duration_s.toFixed(1)}s</td>
            </tr>
          )}
          <tr>
            <td style={labelTd}>Links</td>
            <td style={{ padding: 4 }}>{stats.total_links}</td>
          </tr>
          <tr>
            <td style={labelTd}>Calibration samples</td>
            <td style={{ padding: 4 }}>{stats.calibration_samples.toLocaleString()}</td>
          </tr>
        </tbody>
      </table>

      </div>

      {Object.keys(stats.link_baselines).length > 0 && (
        <div className="card card-pad">
          <div style={{ overflowX: 'auto' }}>
            <LinkCalibrationTable baselines={stats.link_baselines} target={stats.baseline_target} macLabel={macLabel} />
          </div>
        </div>
      )}

      <div className="card card-pad">
      <div className="section-title">TDMA & sensing rates</div>
      <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 0, fontSize: 14 }}>
        <tbody>
          <tr>
            <td style={labelTd}>Total samples</td>
            <td style={{ padding: 4 }}>{stats.total_samples.toLocaleString()}</td>
          </tr>
          <tr>
            <td style={labelTd}>Cycle (config)</td>
            <td style={{ padding: 4 }}>
              {tdma.n_nodes} nodes × {tdma.slot_ms}ms + {tdma.cycle_delay_ms}ms = {tdma.cycle_ms}ms ({tdma.cycles_per_sec} Hz)
            </td>
          </tr>
          <tr>
            <td style={labelTd} title="Measured TDMA cycle rate from actual packet timing">Cycle (measured)</td>
            <td style={{ padding: 4, fontWeight: 500 }}>
              {tdma.measured_cycles_per_sec > 0 ? `${tdma.measured_cycles_per_sec} Hz` : '—'}
              {tdma.measured_cycles_per_sec > 0 && tdma.cycles_per_sec > 0 && (
                <span style={{ color: '#888', marginLeft: 8, fontSize: 12 }}>
                  ({(tdma.measured_cycles_per_sec / tdma.cycles_per_sec * 100).toFixed(0)}% of expected)
                </span>
              )}
            </td>
          </tr>
          <tr>
            <td style={labelTd}>Bursts / trigger</td>
            <td style={{ padding: 4 }}>{tdma.bursts_per_trigger}</td>
          </tr>
          <tr>
            <td style={labelTd} title="Maximum achievable CSI samples/s per link based on TDMA timing and burst count.">Expected link rate</td>
            <td style={{ padding: 4 }}>{expected} smp/s</td>
          </tr>
          <tr>
            <td style={labelTd} title="Average observed CSI samples/s per link across all links.">Avg link rate</td>
            <td style={{ padding: 4, color: rateColor(stats.avg_link_rate, expected) }}>
              {stats.avg_link_rate} smp/s
            </td>
          </tr>
          <tr>
            <td style={labelTd} title="Lowest observed CSI samples/s among all links. This link is the calibration and sensing bottleneck.">Min link rate</td>
            <td style={{ padding: 4, color: rateColor(stats.min_link_rate, expected) }}>
              {stats.min_link_rate} smp/s
            </td>
          </tr>
          <tr>
            <td style={labelTd} title="Fraction of transmitted bursts successfully captured as CSI across the system. 100% means every burst produced a CSI measurement.">System capture rate</td>
            <td style={{ padding: 4, color: captureColor(stats.system_capture_rate) }}>
              {(stats.system_capture_rate * 100).toFixed(0)}%
            </td>
          </tr>
        </tbody>
      </table>

      </div>

      <div className="card card-pad">
      <div className="section-title">Per-node rates</div>
      <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 14, minWidth: 700 }}>
        <thead>
          <tr style={{ borderBottom: '2px solid #ddd' }}>
            <th style={{ ...thBase, textAlign: 'left' }}
              title="Node MAC address">Node</th>
            <th style={thR}
              title="CSI yield: total CSI samples/s produced by this node's transmissions, summed across all receivers. Not the physical TX rate — one burst can produce up to (N−1) CSI samples.">TX yield</th>
            <th style={thR}
              title="CSI yield: total CSI samples/s this node captured as a receiver, summed across all transmitters.">RX yield</th>
            <th style={thR}
              title="Average CSI sampling rate across all links involving this node. This is the per-link sensing rate that determines temporal resolution for activity recognition.">Avg link (Hz)</th>
            <th style={thR}
              title="Slowest per-link CSI rate among all links involving this node. The calibration and sensing bottleneck for this node.">Min link (Hz)</th>
            <th style={thR}
              title="Fraction of transmitted bursts successfully captured as CSI. Expected is 100% — lower values indicate packet loss from interference, distance, or hardware issues.">Capture</th>
            <th style={{ ...thR, width: '14%' }}></th>
            <th style={thR}
              title="Number of directed links (TX→RX pairs) this node participates in. Fully meshed = 2×(N−1).">Links</th>
          </tr>
        </thead>
        <tbody>
          {sortedEntries.map(([mac, r]) => {
            const node = nodes.find((n) => n.mac === mac);
            const linkColor = rateColor(r.avg_link_rate, expected);
            const capColor = captureColor(r.capture_rate);
            return (
              <tr key={mac} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '6px 8px 6px 0', fontSize: 13 }}>
                  {node?.label && <span style={{ fontWeight: 600, marginRight: 6 }}>{node.label}</span>}
                  <span style={{ fontFamily: 'monospace', color: '#888' }}>{mac.slice(-5)}</span>
                </td>
                <td style={{ ...td, color: '#666' }}>{r.tx_yield.toFixed(0)}</td>
                <td style={{ ...td, color: '#666' }}>{r.rx_yield.toFixed(0)}</td>
                <td style={{ ...td, color: linkColor, fontWeight: 600 }}>{r.avg_link_rate.toFixed(1)}</td>
                <td style={{ ...td, color: rateColor(r.min_link_rate, expected) }}>{r.min_link_rate.toFixed(1)}</td>
                <td style={{ ...td, color: capColor }}>{(r.capture_rate * 100).toFixed(0)}%</td>
                <td style={{ padding: '6px 4px' }}>{pctBar(r.capture_rate, capColor)}</td>
                <td style={{ ...td, color: '#888' }}>{r.link_count}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      </div>
      {sortedEntries.length === 0 && (
        <p style={{ color: 'var(--text-faint)', fontStyle: 'italic', margin: '8px 0 0' }}>No sample data yet.</p>
      )}
      </div>
    </div>
  );
}
