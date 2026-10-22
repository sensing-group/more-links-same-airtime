import { useState, useEffect } from 'react';
import { useStore } from './store';
import type { NodeInfo } from './types';

function formatHMS(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(sec)}`;
}

export function ConfigPanel() {
  const { room, nodes, detection, recording, setRoom } = useStore();
  const [width, setWidth] = useState('');
  const [height, setHeight] = useState('');
  const [edited, setEdited] = useState(false);

  useEffect(() => {
    if (!edited) {
      setWidth(String(room.width));
      setHeight(String(room.height));
    }
  }, [room.width, room.height, edited]);

  const [activities, setActivities] = useState<string[]>([]);
  const [environments, setEnvironments] = useState<{ environment_id: string }[]>([]);
  const [subjects, setSubjects] = useState<{ subject_id: string }[]>([]);
  const [topologies, setTopologies] = useState<{ topology_id: string }[]>([]);

  const [activity, setActivity] = useState('');
  const [subjectId, setSubjectId] = useState('');
  const [environmentId, setEnvironmentId] = useState('');
  const [topologyId, setTopologyId] = useState('');
  const [durationHours, setDurationHours] = useState('');
  const [recError, setRecError] = useState('');

  useEffect(() => {
    const okJson = async (url: string) => {
      try {
        const r = await fetch(url);
        if (!r.ok) return null;
        const data = await r.json();
        return Array.isArray(data) ? data : null;
      } catch { return null; }
    };
    okJson('/api/annotations/activities').then(a => { if (a) { setActivities(a); if (a.length) setActivity(v => v || a[0]); } });
    okJson('/api/environments').then(e => { if (e) { setEnvironments(e); if (e.length) setEnvironmentId(v => v || e[0].environment_id); } });
    okJson('/api/subjects').then(s => { if (s) { setSubjects(s); if (s.length) setSubjectId(v => v || s[0].subject_id); } });
    okJson('/api/topologies').then(t => { if (t) { setTopologies(t); if (t.length) setTopologyId(v => v || t[0].topology_id); } });
  }, []);

  const handleSave = async () => {
    const w = parseFloat(width);
    const h = parseFloat(height);
    if (isNaN(w) || isNaN(h) || w <= 0 || h <= 0) return;
    try {
      await fetch('/api/room', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ width: w, height: h }),
      });
      setRoom({ width: w, height: h });
      setEdited(false);
    } catch { /* ignore */ }
  };

  const handleReset = async () => {
    try {
      await fetch('/api/calibration/reset', { method: 'POST' });
    } catch { /* ignore */ }
  };

  const onlineCount = nodes.filter(n => n.status === 'online').length;
  const positionedCount = nodes.filter(n => n.position).length;

  return (
    <div className="card card-pad">
      <div className="section-title">Room</div>

      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8,
        marginBottom: 14,
      }}>
        <Stat label="Online" value={`${onlineCount}/${nodes.length}`} />
        <Stat label="Placed" value={`${positionedCount}/${nodes.length}`} />
        <Stat label="Links"  value={detection.total_links} />
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end', marginBottom: 10 }}>
        <div className="field" style={{ flex: 1, minWidth: 0 }}>
          <label className="field-label">Width (m)</label>
          <input className="field-input" inputMode="decimal" value={width}
                 onChange={e => { setWidth(e.target.value); setEdited(true); }}
                 placeholder="4.43" />
        </div>
        <div style={{ paddingBottom: 10, color: 'var(--text-faint)' }}>×</div>
        <div className="field" style={{ flex: 1, minWidth: 0 }}>
          <label className="field-label">Depth (m)</label>
          <input className="field-input" inputMode="decimal" value={height}
                 onChange={e => { setHeight(e.target.value); setEdited(true); }}
                 placeholder="2.91" />
        </div>
        <button className="btn btn-primary" onClick={handleSave} disabled={!edited}>
          Save
        </button>
      </div>

      <button className="btn btn-block" onClick={handleReset}
              style={{ borderColor: 'var(--warning)', color: 'var(--warning)' }}>
        Reset calibration
      </button>

      <p style={{
        fontSize: 11, color: 'var(--text-faint)', marginTop: 10, marginBottom: 0, lineHeight: 1.5,
      }}>
        Drag nodes on the map to place them. Keep the room empty during calibration.
      </p>

      {/* Recording */}
      <div style={{
        marginTop: 18, paddingTop: 16, borderTop: '1px solid var(--border)',
      }}>
        <div className="section-title" style={{ marginBottom: 10 }}>Recording</div>

        {!recording.recording && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
            <Select label="Activity" value={activity} onChange={setActivity}
                    options={activities.map(a => ({ value: a, label: a.replace(/_/g, ' ') }))} />
            <Select label="Subject" value={subjectId} onChange={setSubjectId}
                    options={subjects.map(s => ({ value: s.subject_id, label: s.subject_id }))} />
            <Select label="Environment" value={environmentId} onChange={setEnvironmentId}
                    options={environments.map(e => ({ value: e.environment_id, label: e.environment_id }))} />
            <Select label="Topology" value={topologyId} onChange={setTopologyId}
                    options={topologies.map(t => ({ value: t.topology_id, label: t.topology_id }))} />

            <div className="field">
              <label className="field-label">Auto-stop (hours, optional)</label>
              <input className="field-input" type="number" min="0" step="0.25"
                     value={durationHours}
                     onChange={e => setDurationHours(e.target.value)}
                     placeholder="Leave empty = until manually stopped" />
            </div>
          </div>
        )}

        {recError && (
          <div style={{
            color: 'var(--danger)', fontSize: 12, marginBottom: 8,
            padding: '6px 10px', background: 'var(--danger-soft)', borderRadius: 6,
          }}>{recError}</div>
        )}

        <button
          className={recording.recording ? 'btn btn-danger btn-block btn-lg' : 'btn btn-success btn-block btn-lg'}
          onClick={async () => {
            setRecError('');
            try {
              if (recording.recording) {
                await fetch('/api/recording/stop', { method: 'POST' });
              } else {
                let durationS: number | null = null;
                const trimmed = durationHours.trim();
                if (trimmed !== '') {
                  const h = parseFloat(trimmed);
                  if (isNaN(h) || h <= 0) {
                    setRecError('Duration must be a positive number of hours');
                    return;
                  }
                  durationS = h * 3600;
                }
                const r = await fetch('/api/recording/start', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    activity, subject_id: subjectId,
                    environment_id: environmentId, topology_id: topologyId,
                    duration_s: durationS,
                  }),
                });
                if (!r.ok) {
                  const data = await r.json();
                  setRecError(data.detail || 'Failed to start');
                }
              }
            } catch { setRecError('Connection error'); }
          }}
        >
          {recording.recording ? '■ Stop recording' : '● Start recording'}
        </button>

        {recording.recording && (
          <div style={{
            marginTop: 12, padding: 12, background: 'var(--bg-soft)',
            borderRadius: 8, fontSize: 12, color: 'var(--text-muted)',
            lineHeight: 1.7,
          }}>
            <KV k="Elapsed" v={formatHMS(recording.elapsed_s ?? 0)} />
            {recording.remaining_s != null && (
              <KV k="Auto-stop in" v={formatHMS(recording.remaining_s)} />
            )}
            <KV k="Raw packets" v={recording.raw_written.toLocaleString()} />
            <KV k="Calibrated"  v={recording.cal_written.toLocaleString()} />
            <KV k="Dropped"     v={String(recording.dropped)} />
            {recording.current_raw_file && (
              <div style={{
                fontSize: 10, marginTop: 6, paddingTop: 6,
                borderTop: '1px solid var(--border)',
                wordBreak: 'break-all', color: 'var(--text-faint)',
              }}>
                {recording.current_raw_file}
              </div>
            )}
          </div>
        )}

        <label style={{
          display: 'flex', alignItems: 'center', gap: 8, fontSize: 12,
          color: 'var(--text-muted)', cursor: 'pointer', marginTop: 12,
        }}>
          <input
            type="checkbox"
            checked={recording.compress}
            onChange={async (e) => {
              try {
                await fetch('/api/recording/compress', {
                  method: 'PUT',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ enabled: e.target.checked }),
                });
              } catch { /* ignore */ }
            }}
          />
          Compression {recording.compress ? 'enabled' : 'disabled'}
          <span style={{ fontSize: 10, color: 'var(--text-faint)' }}>(applies on next file)</span>
        </label>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{
      padding: '8px 10px', background: 'var(--bg-soft)', borderRadius: 8,
      textAlign: 'center', minWidth: 0,
    }}>
      <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>{value}</div>
      <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: 0.04 }}>{label}</div>
    </div>
  );
}

function Select({ label, value, onChange, options }: {
  label: string; value: string; onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="field">
      <label className="field-label">{label}</label>
      <select className="field-input" value={value} onChange={e => onChange(e.target.value)}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
      <span>{k}</span>
      <strong style={{ color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>{v}</strong>
    </div>
  );
}

function NodeRow({ node }: { node: NodeInfo }) {
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(node.label ?? '');
  const [x, setX] = useState(node.position ? String(node.position[0]) : '');
  const [y, setY] = useState(node.position ? String(node.position[1]) : '');

  useEffect(() => {
    if (!editing) {
      setLabel(node.label ?? '');
      setX(node.position ? String(node.position[0]) : '');
      setY(node.position ? String(node.position[1]) : '');
    }
  }, [node, editing]);

  const save = async () => {
    const px = parseFloat(x);
    const py = parseFloat(y);
    const promises: Promise<unknown>[] = [];
    if (!isNaN(px) && !isNaN(py)) {
      promises.push(fetch(`/api/nodes/${encodeURIComponent(node.mac)}/position`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: px, y: py }),
      }));
    }
    if (label.trim()) {
      promises.push(fetch(`/api/nodes/${encodeURIComponent(node.mac)}/label`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: label.trim() }),
      }));
    }
    await Promise.all(promises);
    setEditing(false);
  };

  const cellStyle: React.CSSProperties = { padding: '6px 6px', fontSize: 12 };
  const smallInput: React.CSSProperties = {
    width: '100%', padding: '4px 6px', border: '1px solid var(--border-strong)',
    borderRadius: 4, fontSize: 12,
  };

  return (
    <tr style={{ borderBottom: '1px solid var(--border)' }}>
      <td style={cellStyle}>
        <span style={{
          display: 'inline-block', width: 7, height: 7, borderRadius: '50%', marginRight: 6,
          background: node.status === 'online' ? 'var(--success)' : 'var(--text-faint)',
        }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
          {node.mac.slice(-5)}
        </span>
      </td>
      <td style={cellStyle}>
        {editing
          ? <input value={label} onChange={e => setLabel(e.target.value)} style={smallInput} />
          : <span style={{ fontWeight: 600 }}>{node.label || '—'}</span>}
      </td>
      <td style={{ ...cellStyle, width: 56 }}>
        {editing
          ? <input type="number" step="0.01" value={x} onChange={e => setX(e.target.value)} style={smallInput} />
          : <span style={{ fontFamily: 'var(--font-mono)' }}>{node.position ? node.position[0].toFixed(2) : '—'}</span>}
      </td>
      <td style={{ ...cellStyle, width: 56 }}>
        {editing
          ? <input type="number" step="0.01" value={y} onChange={e => setY(e.target.value)} style={smallInput} />
          : <span style={{ fontFamily: 'var(--font-mono)' }}>{node.position ? node.position[1].toFixed(2) : '—'}</span>}
      </td>
      <td style={{ ...cellStyle, width: 60, textAlign: 'right' }}>
        {editing ? (
          <div style={{ display: 'flex', gap: 3, justifyContent: 'flex-end' }}>
            <button className="btn btn-success btn-sm" onClick={save}
                    style={{ padding: '2px 6px', fontSize: 10 }}>OK</button>
            <button className="btn btn-sm" onClick={() => setEditing(false)}
                    style={{ padding: '2px 6px', fontSize: 10 }}>×</button>
          </div>
        ) : (
          <button className="btn btn-sm" onClick={() => setEditing(true)}
                  style={{ padding: '2px 8px', fontSize: 10 }}>Edit</button>
        )}
      </td>
    </tr>
  );
}

export function NodeListPanel() {
  const { nodes } = useStore();

  const sorted = [...nodes].sort((a, b) => {
    const la = a.label ?? '';
    const lb = b.label ?? '';
    const na = parseInt(la.replace(/\D/g, '')) || 999;
    const nb = parseInt(lb.replace(/\D/g, '')) || 999;
    return na - nb;
  });

  return (
    <div className="card card-pad">
      <div className="section-title">Nodes ({nodes.length})</div>
      {nodes.length === 0 ? (
        <div style={{ fontSize: 13, color: 'var(--text-faint)', padding: '8px 0' }}>
          No nodes detected yet.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>MAC</th>
                <th style={thStyle}>Label</th>
                <th style={thStyle}>X</th>
                <th style={thStyle}>Y</th>
                <th style={thStyle}></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(n => <NodeRow key={n.mac} node={n} />)}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const thStyle: React.CSSProperties = {
  padding: '4px 6px', fontSize: 10, textAlign: 'left',
  color: 'var(--text-faint)', textTransform: 'uppercase',
  letterSpacing: 0.04, fontWeight: 600,
  borderBottom: '1px solid var(--border)',
};
