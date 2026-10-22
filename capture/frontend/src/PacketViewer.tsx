import { useStore } from './store';
import { PacketInfo } from './types';
import { useEffect, useRef, useState, useMemo } from 'react';

export function PacketViewer() {
  const packets = useStore((state) => state.packets);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const nodeLabels = useMemo(() => {
    const uniqueMacs = new Set<string>();
    packets.forEach((pkt: PacketInfo) => {
      uniqueMacs.add(pkt.tx_mac);
      uniqueMacs.add(pkt.rx_mac);
    });

    const labels: Record<string, string> = {};
    const sortedMacs = Array.from(uniqueMacs).sort();
    sortedMacs.forEach((mac, idx) => {
      labels[mac] = `Node ${String.fromCharCode(65 + idx)}`; // A, B, C, D...
    });
    return labels;
  }, [packets]);

  const formatMac = (mac: string) => mac.slice(-8);

  const getNodeLabel = (mac: string) => nodeLabels[mac] || 'Unknown';
  const formatTimestamp = (ts: number) => {
    const ms = ts / 1000;
    return ms.toFixed(1);
  };

  useEffect(() => {
    if (autoScroll && scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [packets, autoScroll]);

  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;

    if (!isAtBottom && autoScroll) {
      setAutoScroll(false);
    } else if (isAtBottom && !autoScroll) {
      setAutoScroll(true);
    }
  };

  return (
    <div className="card card-pad">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
        <div>
          <div className="section-title" style={{ marginBottom: 2 }}>CSI Packet Stream</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Last {packets.length} packets · live calibration statistics
          </div>
        </div>
        <button
          onClick={() => setAutoScroll(!autoScroll)}
          className={autoScroll ? 'btn btn-success btn-sm' : 'btn btn-sm'}
        >
          {autoScroll ? '⟳ Auto-scroll on' : '⏸ Auto-scroll off'}
        </button>
      </div>

      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        style={{
          maxHeight: 600,
          overflow: 'auto',
          border: '1px solid var(--border)',
          borderRadius: 8,
        }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: 12,
          fontFamily: 'var(--font-mono)',
          minWidth: 900,
        }}>
          <thead style={{
            position: 'sticky',
            top: 0,
            backgroundColor: '#f5f5f5',
            borderBottom: '2px solid #ddd',
          }}>
            <tr>
              <th style={thStyle}>Link</th>
              <th style={thStyle}>Time (ms)</th>
              <th style={thStyle}>Seq</th>
              <th style={thStyle}>RSSI</th>
              <th style={thStyle}>Variance</th>
              <th style={thStyle}>Baseline μ</th>
              <th style={thStyle}>Baseline σ</th>
              <th style={thStyle}>Samples</th>
              <th style={{...thStyle, minWidth: 180}}>Z-Score</th>
              <th style={thStyle}>State</th>
            </tr>
          </thead>
          <tbody>
            {packets.length === 0 ? (
              <tr>
                <td colSpan={10} style={{
                  textAlign: 'center',
                  padding: 24,
                  color: '#999',
                  fontFamily: 'system-ui',
                }}>
                  Waiting for packets...
                </td>
              </tr>
            ) : (
              packets.map((pkt: PacketInfo, idx: number) => {
                const isActive = pkt.z_score !== null && Math.abs(pkt.z_score) > 2.5;
                const rowBg = isActive ? '#fff3cd' : idx % 2 === 0 ? '#fafafa' : '#fff';

                const txLabel = getNodeLabel(pkt.tx_mac);
                const rxLabel = getNodeLabel(pkt.rx_mac);

                const zScore = pkt.z_score ?? 0;
                const zScoreAbs = Math.abs(zScore);
                const barWidth = Math.min(100, (zScoreAbs / 5) * 100); // Scale: 0-5 maps to 0-100%
                const barColor = zScoreAbs > 2.5 ? '#d63031' : zScoreAbs > 1.5 ? '#fdcb6e' : '#74b9ff';

                return (
                  <tr key={`${pkt.tx_mac}-${pkt.rx_mac}-${pkt.seq}`} style={{ backgroundColor: rowBg }}>
                    <td style={tdStyle}>
                      <div style={{ fontSize: 11 }}>
                        <div style={{
                          fontWeight: 600,
                          color: '#2d3436',
                          marginBottom: 2
                        }}>
                          {txLabel} → {rxLabel}
                        </div>
                        <div style={{
                          color: '#999',
                          fontSize: 9,
                          fontFamily: 'monospace'
                        }}>
                          {formatMac(pkt.tx_mac)}→{formatMac(pkt.rx_mac)}
                        </div>
                      </div>
                    </td>
                    <td style={tdStyle}>{formatTimestamp(pkt.timestamp_us)}</td>
                    <td style={tdStyle}>{pkt.seq}</td>
                    <td style={tdStyle}>{pkt.rssi} dBm</td>
                    <td style={tdStyle}>{pkt.amplitude_variance.toFixed(2)}</td>
                    <td style={tdStyle}>{pkt.baseline_mean.toFixed(2)}</td>
                    <td style={tdStyle}>{pkt.baseline_std.toFixed(2)}</td>
                    <td style={tdStyle}>{pkt.baseline_samples}</td>
                    <td style={{
                      ...tdStyle,
                      fontWeight: isActive ? 'bold' : 'normal',
                      color: isActive ? '#d63031' : 'inherit',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ minWidth: 40 }}>
                          {pkt.z_score !== null ? pkt.z_score.toFixed(2) : '-'}
                        </span>
                        {pkt.z_score !== null && (
                          <div style={{
                            flex: 1,
                            height: 16,
                            backgroundColor: '#f0f0f0',
                            borderRadius: 8,
                            overflow: 'hidden',
                            position: 'relative',
                            minWidth: 80,
                          }}>
                            <div style={{
                              width: `${barWidth}%`,
                              height: '100%',
                              backgroundColor: barColor,
                              transition: 'width 0.2s ease-out',
                            }} />
                            <div style={{
                              position: 'absolute',
                              top: 0,
                              left: '50%',
                              width: 2,
                              height: '100%',
                              backgroundColor: '#999',
                              opacity: 0.3,
                            }} />
                          </div>
                        )}
                      </div>
                    </td>
                    <td style={tdStyle}>
                      <span style={{
                        padding: '2px 6px',
                        borderRadius: 3,
                        fontSize: 10,
                        backgroundColor: pkt.state === 'monitoring' ? (isActive ? '#ffe5e5' : '#d4edda') : '#d1ecf1',
                        color: pkt.state === 'monitoring' ? (isActive ? '#d63031' : '#155724') : '#0c5460',
                        fontWeight: isActive && pkt.state === 'monitoring' ? 600 : 'normal',
                      }}>
                        {pkt.state}
                        {isActive && pkt.state === 'monitoring' && ' ⚠️'}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <div style={{
        marginTop: 12,
        fontSize: 11,
        color: 'var(--text-muted)',
        display: 'flex',
        gap: 16,
        flexWrap: 'wrap',
        padding: '10px 12px',
        background: 'var(--bg-soft)',
        borderRadius: 8,
      }}>
        <div>
          <span style={{ fontWeight: 600 }}>Active Link:</span> |Z-score| &gt; 2.5
        </div>
        <div>
          <span style={{ fontWeight: 600 }}>Variance:</span> σ² of amplitude across 52 subcarriers
        </div>
        <div>
          <span style={{ fontWeight: 600 }}>Z-Score:</span> (variance - μ) / σ
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 600 }}>Z-Score Bar:</span>
          <div style={{ width: 16, height: 12, backgroundColor: '#74b9ff', borderRadius: 2 }} />
          <span style={{ fontSize: 11 }}>Low</span>
          <div style={{ width: 16, height: 12, backgroundColor: '#fdcb6e', borderRadius: 2 }} />
          <span style={{ fontSize: 11 }}>Medium</span>
          <div style={{ width: 16, height: 12, backgroundColor: '#d63031', borderRadius: 2 }} />
          <span style={{ fontSize: 11 }}>High (&gt;2.5)</span>
        </div>
      </div>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  padding: '8px 12px',
  textAlign: 'left',
  fontWeight: 600,
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: 0.04,
  color: 'var(--text-faint)',
  borderBottom: '1px solid var(--border)',
  whiteSpace: 'nowrap',
  background: 'var(--bg-soft)',
};

const tdStyle: React.CSSProperties = {
  padding: '6px 12px',
  borderBottom: '1px solid var(--border)',
  whiteSpace: 'nowrap',
};
