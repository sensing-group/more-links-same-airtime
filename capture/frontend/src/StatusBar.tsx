import { useStore } from './store';

export function StatusBar() {
  const { connected, detection, recording } = useStore();
  const isCalibrating = detection.state === 'calibrating';

  const toggleRecording = async () => {
    try {
      const endpoint = recording.recording ? '/api/recording/stop' : '/api/recording/start';
      await fetch(endpoint, { method: 'POST' });
    } catch { /* ignore */ }
  };

  return (
    <div className="status-bar" style={{
      display: 'flex', alignItems: 'center', gap: 14, rowGap: 6,
      padding: '6px 10px', background: 'var(--bg-soft)',
      border: '1px solid var(--border)', borderRadius: 999,
      flexWrap: 'nowrap', minWidth: 0,
    }}>
      {/* Connection */}
      <div title={connected ? 'WebSocket connected' : 'WebSocket disconnected'}
           style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: connected ? 'var(--success)' : 'var(--danger)',
          boxShadow: connected ? '0 0 0 3px rgba(34,197,94,0.18)' : '0 0 0 3px rgba(239,68,68,0.18)',
        }} />
        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 500 }}>
          {connected ? 'Live' : 'Offline'}
        </span>
      </div>

      <span className="status-bar-sep" style={{ width: 1, height: 18, background: 'var(--border)' }} />

      {/* Occupancy */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}
           title={isCalibrating ? 'Calibration in progress' : detection.occupied ? 'Room is occupied' : 'Room is empty'}>
        <span style={{
          fontSize: 22, fontWeight: 800, lineHeight: 1,
          color: detection.occupied ? 'var(--danger)' : 'var(--success)',
        }}>
          {detection.occupancy}
        </span>
        <span style={{
          fontSize: 11, color: 'var(--text-muted)', fontWeight: 600,
          textTransform: 'uppercase', letterSpacing: 0.04,
        }}>
          {isCalibrating ? 'Calibrating' : detection.occupied ? 'Occupied' : 'Empty'}
        </span>
      </div>

      <span style={{
        fontSize: 11, color: 'var(--text-faint)',
        fontVariantNumeric: 'tabular-nums',
        display: 'inline-block', minWidth: 86, textAlign: 'right',
      }}>
        {detection.active_links}/{detection.total_links} links
      </span>

      {isCalibrating && (
        <div style={{
          width: 70, height: 5, background: 'var(--bg-sunken)',
          borderRadius: 999, overflow: 'hidden',
        }}>
          <div style={{
            width: `${detection.calibration_progress * 100}%`, height: '100%',
            background: 'var(--brand)', transition: 'width 0.3s',
          }} />
        </div>
      )}

      <span className="status-bar-sep" style={{ width: 1, height: 18, background: 'var(--border)' }} />

      {/* Recording toggle */}
      <button
        onClick={toggleRecording}
        title={recording.recording ? 'Click to stop recording' : 'Click to start recording'}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '4px 10px', borderRadius: 999, cursor: 'pointer',
          border: '1px solid',
          borderColor: recording.recording ? 'var(--danger)' : 'var(--border-strong)',
          background: recording.recording ? 'rgba(220,38,38,0.08)' : '#fff',
          color: recording.recording ? 'var(--danger)' : 'var(--text-muted)',
          fontSize: 12, fontWeight: 600,
        }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: recording.recording ? 'var(--danger)' : 'var(--text-faint)',
          animation: recording.recording ? 'rec-pulse 1.6s infinite' : 'none',
        }} />
        {recording.recording
          ? `REC · ${recording.raw_written.toLocaleString()}`
          : 'Not recording'}
      </button>
    </div>
  );
}
