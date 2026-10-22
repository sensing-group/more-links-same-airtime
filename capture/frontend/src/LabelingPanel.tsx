import { useState, useEffect, useCallback } from 'react';
import { useStore } from './store';
import { Annotation } from './types';

const EMOJI_MAP: Record<string, string> = {
  walking: '🚶', running: '🏃', climbing_stairs: '🧗',
  descending_stairs: '⬇️', cycling: '🚲',
  sitting: '🪑', standing: '🧍', lying_down: '🛏️',
  sleeping: '😴', working_at_desk: '💻', eating: '🍽️',
  phone_call: '📱', cooking: '🍳', reading: '📚',
  sitting_down: '⬇️', standing_up: '⬆️',
  entering_room: '🚪', leaving_room: '🚪',
  lying_down_transition: '⬇️', getting_up: '⬆️',
  push_ups: '💪', jumping_jacks: '⭐',
  empty_room: '🏠', multiple_people: '👥',
  other: '❓', unknown: '❔',
};

function formatDuration(s: number): string {
  if (s === 0) return '';
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return sec > 0 ? `${m}m ${sec}s` : `${m}m`;
}

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString('en-GB', { timeZone: 'UTC', hour12: false });
}

function useAnnotations() {
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [taxonomy, setTaxonomy] = useState<Record<string, string[]>>({});

  const refresh = useCallback(async () => {
    try {
      const r = await fetch('/api/annotations');
      if (r.ok) setAnnotations(await r.json());
    } catch { /* ignore */ }
  }, []);

  const loadTaxonomy = useCallback(async () => {
    try {
      const r = await fetch('/api/annotations/taxonomy');
      if (r.ok) setTaxonomy(await r.json());
    } catch { /* ignore */ }
  }, []);

  const submit = useCallback(async (fineLabel: string, startTs?: number, endTs?: number) => {
    const body: Record<string, unknown> = { fine_label: fineLabel, start_ts: startTs ?? Date.now() / 1000 };
    if (endTs) body.end_ts = endTs;
    const r = await fetch('/api/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    await refresh();
    return r.json();
  }, [refresh]);

  const remove = useCallback(async (annotationId: string) => {
    await fetch(`/api/annotations/${annotationId}`, { method: 'DELETE' });
    await refresh();
  }, [refresh]);

  return { annotations, taxonomy, refresh, loadTaxonomy, submit, remove };
}

export function LabelingPanel() {
  const { recording } = useStore();
  const { annotations, taxonomy, refresh, loadTaxonomy, submit, remove } = useAnnotations();

  const [clock, setClock] = useState('');
  const [flashId, setFlashId] = useState<string | null>(null);
  const [flashColor, setFlashColor] = useState('');
  const [disabled, setDisabled] = useState<string | null>(null);
  const [durationMode, setDurationMode] = useState(false);
  const [activeInterval, setActiveInterval] = useState<{ label: string; startTs: number } | null>(null);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const tick = () => {
      const d = new Date();
      const parts = d.toUTCString().split(' ');
      // "Thu, 26 Mar 2026 14:23:51 GMT"
      const time = d.toLocaleTimeString('en-GB', { timeZone: 'UTC', hour12: false });
      setClock(`${parts[0]} ${parts[1]} ${parts[2]} ${parts[3]} \u00b7 ${time} UTC`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => { loadTaxonomy(); refresh(); }, [loadTaxonomy, refresh]);

  useEffect(() => {
    const id = setInterval(refresh, 15000);
    return () => clearInterval(id);
  }, [refresh]);

  const handleTap = async (fineLabel: string) => {
    if (disabled === fineLabel) return;

    if (durationMode) {
      if (activeInterval) {
        const endTs = Date.now() / 1000;
        try {
          await submit(activeInterval.label, activeInterval.startTs, endTs);
          setFlashId(activeInterval.label);
          setFlashColor('#22c55e');
        } catch {
          setFlashId(activeInterval.label);
          setFlashColor('#ef4444');
        }
        setActiveInterval(null);
        setTimeout(() => setFlashId(null), 600);
      } else {
        setActiveInterval({ label: fineLabel, startTs: Date.now() / 1000 });
      }
      return;
    }

    setDisabled(fineLabel);
    try {
      await submit(fineLabel);
      setFlashId(fineLabel);
      setFlashColor('#22c55e');
      setTimeout(() => setFlashId(null), 600);
    } catch {
      setFlashId(fineLabel);
      setFlashColor('#ef4444');
      setTimeout(() => setFlashId(null), 800);
    }
    setTimeout(() => setDisabled(null), 500);
  };

  const prettyLabel = (s: string) => s.replace(/_/g, ' ');

  return (
    <div style={{ maxWidth: 640, margin: '0 auto' }}>
      <div className="card card-pad" style={{
        marginBottom: 16, display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
      }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: 0.04 }}>
            Server clock
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-muted)' }}>{clock}</div>
        </div>
        <span className={recording.recording ? 'badge badge-danger' : 'badge'}>
          {recording.recording ? '● Recording' : 'Not recording'}
        </span>
        <label style={{
          display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
          cursor: 'pointer', color: 'var(--text-muted)',
        }}>
          <input type="checkbox" checked={durationMode}
                 onChange={(e) => { setDurationMode(e.target.checked); setActiveInterval(null); }} />
          Duration mode
          {activeInterval && <span style={{ color: 'var(--warning)' }}> — tap again to end</span>}
        </label>
      </div>

      {Object.entries(taxonomy).map(([category, activities]) => {
        const isCollapsed = collapsed[category] ?? false;
        return (
          <div key={category} className="card card-pad" style={{ marginBottom: 12 }}>
            <div
              onClick={() => setCollapsed(prev => ({ ...prev, [category]: !isCollapsed }))}
              style={{
                fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase',
                letterSpacing: 0.06, marginBottom: isCollapsed ? 0 : 10,
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
                userSelect: 'none',
              }}
            >
              <span style={{
                display: 'inline-block', transition: 'transform 0.2s',
                transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
                fontSize: 10,
              }}>▼</span>
              {category}
              <span style={{ fontWeight: 400, fontSize: 10, color: '#bbb' }}>
                ({activities.length})
              </span>
            </div>
            {!isCollapsed && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {activities.map((act) => {
                  const isActive = activeInterval?.label === act;
                  const isFlashing = flashId === act;
                  return (
                    <button
                      key={act}
                      onClick={() => handleTap(act)}
                      disabled={disabled === act}
                      style={{
                        width: '100%', minHeight: 48, padding: '10px 16px',
                        border: isActive ? '2px solid var(--warning)' : '1px solid var(--border)',
                        borderRadius: 10, cursor: 'pointer', fontSize: 15,
                        textAlign: 'left', display: 'flex', alignItems: 'center', gap: 12,
                        background: isFlashing ? flashColor : isActive ? 'var(--warning-soft)' : '#fff',
                        color: isFlashing ? '#fff' : 'var(--text)',
                        transition: 'background 0.2s, border-color 0.2s',
                        opacity: disabled === act ? 0.5 : 1,
                        fontWeight: 500,
                      }}
                    >
                      <span style={{ fontSize: 22 }}>{EMOJI_MAP[act] || '⭕'}</span>
                      <span>{prettyLabel(act)}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      <div className="card card-pad" style={{ marginTop: 16 }}>
        <div className="section-title">Recent (today)</div>
        <div style={{ maxHeight: 280, overflowY: 'auto' }}>
          {annotations.length === 0 && (
            <div style={{ fontSize: 13, color: '#999' }}>No annotations yet</div>
          )}
          {annotations.map((a) => (
            <div key={a.annotation_id} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '8px 0', borderBottom: '1px solid var(--border)', fontSize: 13,
            }}>
              <span style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                {formatTime(a.start_ts)}
              </span>
              <span style={{ fontSize: 16 }}>{EMOJI_MAP[a.fine_label] || '⭕'}</span>
              <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{prettyLabel(a.fine_label)}</span>
              {a.duration_s > 0 && (
                <span style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>{formatDuration(a.duration_s)}</span>
              )}
              <button
                onClick={() => remove(a.annotation_id)}
                style={{
                  border: 'none', background: 'none', cursor: 'pointer',
                  color: 'var(--text-faint)', fontSize: 16, padding: '0 4px',
                  lineHeight: 1,
                }}
                title="Delete"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
