import { useState, useEffect, useCallback, useRef } from 'react';
import { useStore } from './store';

let _audioCtx: AudioContext | null = null;
function getAudioCtx(): AudioContext {
  if (!_audioCtx) _audioCtx = new AudioContext();
  return _audioCtx;
}

function playBeep(type: 'position' | 'start' | 'done') {
  const ctx = getAudioCtx();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain);
  gain.connect(ctx.destination);

  if (type === 'position') {
    // Single medium tone
    osc.frequency.value = 660;
    gain.gain.value = 0.4;
    osc.start();
    osc.stop(ctx.currentTime + 0.4);
  } else if (type === 'start') {
    // Two quick high tones
    osc.frequency.value = 880;
    gain.gain.value = 0.5;
    gain.gain.setValueAtTime(0.5, ctx.currentTime);
    gain.gain.setValueAtTime(0, ctx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.5, ctx.currentTime + 0.25);
    gain.gain.setValueAtTime(0, ctx.currentTime + 0.4);
    osc.start();
    osc.stop(ctx.currentTime + 0.5);
  } else {
    // Three descending tones = done
    osc.frequency.value = 880;
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.setValueAtTime(660, ctx.currentTime + 0.2);
    osc.frequency.setValueAtTime(440, ctx.currentTime + 0.4);
    gain.gain.value = 0.5;
    osc.start();
    osc.stop(ctx.currentTime + 0.6);
  }
}

function formatCountdown(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}:${sec.toString().padStart(2, '0')}` : `${sec}s`;
}

const PHASE_LABELS: Record<string, string> = {
  idle: 'Ready',
  baseline: 'Empty Room (Baseline)',
  position: 'Get in Position',
  active: 'Activity in Progress',
  done: 'Experiment Complete',
};

const PHASE_COLORS: Record<string, string> = {
  idle: '#6b7280',
  baseline: '#3b82f6',
  position: '#f59e0b',
  active: '#22c55e',
  done: '#8b5cf6',
};

interface ActivityEntry {
  activity: string;
  activity_s: string; // string for input binding
  note: string;
}

export function ExperimentPanel() {
  const { experiment } = useStore();
  const [allActivities, setAllActivities] = useState<string[]>([]);
  const [steps, setSteps] = useState<ActivityEntry[]>([
    { activity: 'walking', activity_s: '30', note: '' },
  ]);
  const [baselineS, setBaselineS] = useState('30');
  const [positionS, setPositionS] = useState('10');
  const [transitionS, setTransitionS] = useState('10');
  const [globalNote, setGlobalNote] = useState('');
  const [compress, setCompress] = useState(false);
  const [error, setError] = useState('');

  const [environments, setEnvironments] = useState<{ environment_id: string }[]>([]);
  const [subjects, setSubjects] = useState<{ subject_id: string }[]>([]);
  const [topologies, setTopologies] = useState<{ topology_id: string }[]>([]);
  const [subjectId, setSubjectId] = useState('');
  const [environmentId, setEnvironmentId] = useState('');
  const [topologyId, setTopologyId] = useState('');

  const lastBeepRef = useRef<string | null>(null);

  useEffect(() => {
    fetch('/api/annotations/activities').then(r => r.json()).then(setAllActivities).catch(() => {});
    fetch('/api/environments').then(r => r.json()).then(e => { setEnvironments(e); if (e.length) setEnvironmentId(v => v || e[0].environment_id); }).catch(() => {});
    fetch('/api/subjects').then(r => r.json()).then(s => { setSubjects(s); if (s.length) setSubjectId(v => v || s[0].subject_id); }).catch(() => {});
    fetch('/api/topologies').then(r => r.json()).then(t => { setTopologies(t); if (t.length) setTopologyId(v => v || t[0].topology_id); }).catch(() => {});
  }, []);

  useEffect(() => {
    if (experiment.beep && experiment.beep !== lastBeepRef.current) {
      lastBeepRef.current = experiment.beep;
      playBeep(experiment.beep as 'position' | 'start' | 'done');
    }
    if (experiment.phase === 'idle') {
      lastBeepRef.current = null;
    }
  }, [experiment.beep, experiment.phase]);

  const addStep = useCallback(() => {
    setSteps(prev => [...prev, { activity: allActivities[0] || 'walking', activity_s: '30', note: '' }]);
  }, [allActivities]);

  const removeStep = useCallback((index: number) => {
    setSteps(prev => prev.length > 1 ? prev.filter((_, i) => i !== index) : prev);
  }, []);

  const updateStep = useCallback((index: number, field: keyof ActivityEntry, value: string) => {
    setSteps(prev => prev.map((s, i) => i === index ? { ...s, [field]: value } : s));
  }, []);

  const moveStep = useCallback((index: number, dir: -1 | 1) => {
    setSteps(prev => {
      const next = [...prev];
      const target = index + dir;
      if (target < 0 || target >= next.length) return prev;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }, []);

  const handleStart = useCallback(async () => {
    setError('');
    getAudioCtx().resume();
    try {
      const r = await fetch('/api/experiment/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          activities: steps.map(s => ({
            activity: s.activity,
            activity_s: parseFloat(s.activity_s) || 30,
            note: s.note,
          })),
          baseline_s: parseFloat(baselineS) || 30,
          position_s: parseFloat(positionS) || 10,
          transition_s: parseFloat(transitionS) || 10,
          compress,
          note: globalNote,
          subject_id: subjectId,
          environment_id: environmentId,
          topology_id: topologyId,
        }),
      });
      if (!r.ok) {
        const data = await r.json();
        setError(data.detail || 'Failed to start');
      }
    } catch {
      setError('Connection error');
    }
  }, [steps, baselineS, positionS, transitionS, compress, globalNote, subjectId, environmentId, topologyId]);

  const handleStop = useCallback(async () => {
    await fetch('/api/experiment/stop', { method: 'POST' });
  }, []);

  const isRunning = experiment.phase !== 'idle';
  const progress = experiment.phase_duration_s > 0
    ? ((experiment.phase_duration_s - experiment.remaining_s) / experiment.phase_duration_s) * 100
    : 0;

  const prettyLabel = (s: string) => s.replace(/_/g, ' ');

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '8px 12px', border: '1px solid var(--border-strong)',
    borderRadius: 8, fontSize: 14, boxSizing: 'border-box', background: '#fff',
  };

  const smallBtnStyle: React.CSSProperties = {
    width: 26, height: 26, padding: 0, border: '1px solid var(--border-strong)',
    borderRadius: 6, background: '#fff', cursor: 'pointer', fontSize: 11,
    lineHeight: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    flexShrink: 0,
  };

  const buildTimeline = () => {
    const lines: string[] = [];
    let t = 0;
    lines.push(`${formatCountdown(t)} — Recording starts, room empty`);
    t += parseFloat(baselineS) || 0;
    for (let i = 0; i < steps.length; i++) {
      const wait = i === 0 ? (parseFloat(positionS) || 0) : (parseFloat(transitionS) || 0);
      lines.push(`${formatCountdown(t)} — Beep: get ready for ${prettyLabel(steps[i].activity)}`);
      t += wait;
      lines.push(`${formatCountdown(t)} — Beep: start ${prettyLabel(steps[i].activity)}`);
      t += parseFloat(steps[i].activity_s) || 0;
    }
    lines.push(`${formatCountdown(t)} — Beep: done, recording stops`);
    return lines;
  };

  return (
    <div style={{ maxWidth: 640, margin: '0 auto' }}>
      {isRunning && (
        <div style={{
          padding: 24, borderRadius: 12, marginBottom: 20, textAlign: 'center',
          background: `linear-gradient(135deg, ${PHASE_COLORS[experiment.phase]}15, ${PHASE_COLORS[experiment.phase]}08)`,
          border: `2px solid ${PHASE_COLORS[experiment.phase]}40`,
        }}>
          <div style={{
            fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1.5,
            color: PHASE_COLORS[experiment.phase], marginBottom: 8,
          }}>
            {PHASE_LABELS[experiment.phase]}
          </div>

          <div style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
            {prettyLabel(experiment.activity)}
          </div>
          {experiment.activity_count > 1 && (
            <div style={{ fontSize: 13, color: '#888', marginBottom: 8 }}>
              Activity {experiment.activity_index + 1} of {experiment.activity_count}
            </div>
          )}

          {experiment.phase !== 'done' && (
            <div style={{ fontSize: 56, fontWeight: 700, fontFamily: 'monospace', margin: '12px 0',
                          color: PHASE_COLORS[experiment.phase] }}>
              {formatCountdown(experiment.remaining_s)}
            </div>
          )}

          {experiment.phase !== 'done' && experiment.phase_duration_s > 0 && (
            <div style={{ width: '100%', height: 8, background: '#e5e7eb', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{
                width: `${progress}%`, height: '100%', borderRadius: 4,
                background: PHASE_COLORS[experiment.phase],
                transition: 'width 0.5s linear',
              }} />
            </div>
          )}

          {experiment.activity_count > 1 && (
            <div style={{ display: 'flex', justifyContent: 'center', gap: 6, marginTop: 12 }}>
              {Array.from({ length: experiment.activity_count }, (_, i) => (
                <div key={i} style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: i < experiment.activity_index ? PHASE_COLORS.done
                    : i === experiment.activity_index ? PHASE_COLORS[experiment.phase]
                    : '#d1d5db',
                  transition: 'background 0.3s',
                }} />
              ))}
            </div>
          )}

          <div style={{ fontSize: 13, color: '#888', marginTop: 12 }}>
            Total: {formatCountdown(experiment.total_elapsed_s)}
          </div>

          <button onClick={handleStop} style={{
            marginTop: 16, padding: '10px 32px', border: 'none', borderRadius: 8,
            background: '#ef4444', color: '#fff', fontSize: 15, fontWeight: 600,
            cursor: 'pointer',
          }}>
            Abort Experiment
          </button>
        </div>
      )}

      {!isRunning && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div>
            <label style={{ fontSize: 13, fontWeight: 600, display: 'block', marginBottom: 6 }}>
              Activities ({steps.length})
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {steps.map((step, i) => (
                <div key={i} style={{
                  display: 'grid',
                  gridTemplateColumns: '20px minmax(0, 1.6fr) 64px minmax(0, 1fr) auto',
                  alignItems: 'center', gap: 8,
                  padding: '10px 12px', background: '#fff', borderRadius: 10,
                  border: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-faint)' }}>
                    {i + 1}
                  </span>
                  <select value={step.activity} onChange={e => updateStep(i, 'activity', e.target.value)}
                          style={{ minWidth: 0, padding: '6px 8px', border: '1px solid var(--border-strong)', borderRadius: 6, fontSize: 13 }}>
                    {allActivities.map(a => (
                      <option key={a} value={a}>{prettyLabel(a)}</option>
                    ))}
                  </select>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <input type="number" min="1" step="5" value={step.activity_s}
                           onChange={e => updateStep(i, 'activity_s', e.target.value)}
                           style={{ width: 44, padding: '6px 6px', border: '1px solid var(--border-strong)', borderRadius: 6, fontSize: 13, textAlign: 'center' }}
                           title="Duration (s)" />
                    <span style={{ fontSize: 11, color: 'var(--text-faint)' }}>s</span>
                  </div>
                  <input type="text" value={step.note} onChange={e => updateStep(i, 'note', e.target.value)}
                         placeholder="note" style={{ minWidth: 0, padding: '6px 8px', border: '1px solid var(--border-strong)', borderRadius: 6, fontSize: 12 }} />
                  <div style={{ display: 'flex', gap: 4 }}>
                    <button onClick={() => moveStep(i, -1)} disabled={i === 0}
                            style={{ ...smallBtnStyle, opacity: i === 0 ? 0.3 : 1 }} title="Move up">▲</button>
                    <button onClick={() => moveStep(i, 1)} disabled={i === steps.length - 1}
                            style={{ ...smallBtnStyle, opacity: i === steps.length - 1 ? 0.3 : 1 }} title="Move down">▼</button>
                    <button onClick={() => removeStep(i)} disabled={steps.length <= 1}
                            style={{ ...smallBtnStyle, color: 'var(--danger)', opacity: steps.length <= 1 ? 0.3 : 1 }}
                            title="Remove">✕</button>
                  </div>
                </div>
              ))}
            </div>
            <button onClick={addStep} style={{
              marginTop: 8, padding: '8px 14px', border: '1px dashed var(--border-strong)', borderRadius: 8,
              background: '#fff', cursor: 'pointer', fontSize: 13, color: 'var(--text-muted)', width: '100%',
            }}>
              + Add activity
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 4 }}>
                Empty room (s)
              </label>
              <input type="number" min="0" step="5" value={baselineS}
                     onChange={e => setBaselineS(e.target.value)} style={inputStyle} />
            </div>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 4 }}>
                Get ready (s)
              </label>
              <input type="number" min="0" step="1" value={positionS}
                     onChange={e => setPositionS(e.target.value)} style={inputStyle} />
            </div>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 4 }}>
                Transition (s)
              </label>
              <input type="number" min="0" step="1" value={transitionS}
                     onChange={e => setTransitionS(e.target.value)} style={inputStyle}
                     title="Time between activities to get into next position" />
            </div>
          </div>

          <div style={{ fontSize: 12, color: '#888', lineHeight: 1.8, padding: '8px 12px',
                        background: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
            <strong>Timeline:</strong><br />
            {buildTimeline().map((line, i) => (
              <span key={i}>{line}<br /></span>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 4 }}>Subject</label>
              <select value={subjectId} onChange={e => setSubjectId(e.target.value)}
                      style={{ ...inputStyle, padding: '8px 8px' }}>
                {subjects.map(s => <option key={s.subject_id} value={s.subject_id}>{s.subject_id}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 4 }}>Environment</label>
              <select value={environmentId} onChange={e => setEnvironmentId(e.target.value)}
                      style={{ ...inputStyle, padding: '8px 8px' }}>
                {environments.map(e => <option key={e.environment_id} value={e.environment_id}>{e.environment_id}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 4 }}>Topology</label>
              <select value={topologyId} onChange={e => setTopologyId(e.target.value)}
                      style={{ ...inputStyle, padding: '8px 8px' }}>
                {topologies.map(t => <option key={t.topology_id} value={t.topology_id}>{t.topology_id}</option>)}
              </select>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 13, fontWeight: 600, display: 'block', marginBottom: 4 }}>
                Note (optional)
              </label>
              <input type="text" value={globalNote} onChange={e => setGlobalNote(e.target.value)}
                     placeholder="e.g. trial 3, fast pace" style={inputStyle} />
            </div>
            <label style={{
              display: 'flex', alignItems: 'center', gap: 6, fontSize: 13,
              cursor: 'pointer', color: '#555', whiteSpace: 'nowrap', paddingBottom: 8,
            }}>
              <input type="checkbox" checked={compress} onChange={e => setCompress(e.target.checked)} />
              Compress
            </label>
          </div>

          {error && <div style={{ color: '#ef4444', fontSize: 13 }}>{error}</div>}

          <button onClick={handleStart} className="btn btn-success btn-block btn-lg"
                  style={{ padding: '14px 0', fontSize: 16, fontWeight: 700 }}>
            ▶  Start experiment · {steps.length} activit{steps.length === 1 ? 'y' : 'ies'}
          </button>
        </div>
      )}
    </div>
  );
}
