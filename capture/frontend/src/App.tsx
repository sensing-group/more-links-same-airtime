import { useState } from 'react';
import { useWebSocket } from './useWebSocket';
import { RoomMap } from './RoomMap';
import { StatusBar } from './StatusBar';
import { ConfigPanel, NodeListPanel } from './ConfigPanel';
import { PacketViewer } from './PacketViewer';
import { LabelingPanel } from './LabelingPanel';
import { ExperimentPanel } from './ExperimentPanel';
import { StatsPanel } from './StatsPanel';
import { SetupPanel } from './SetupPanel';

type Tab = 'room' | 'experiment' | 'labels' | 'stats' | 'packets' | 'setup';

interface TabDef {
  id: Tab;
  label: string;
  hint: string;
}

const TABS: TabDef[] = [
  { id: 'room',       label: 'Room',       hint: 'Live map & calibration' },
  { id: 'experiment', label: 'Experiment', hint: 'Run a structured trial' },
  { id: 'labels',     label: 'Label',      hint: 'Tag activities live' },
  { id: 'stats',      label: 'Stats',      hint: 'Sensing rates & quality' },
  { id: 'packets',    label: 'Packets',    hint: 'Raw CSI stream' },
  { id: 'setup',      label: 'Setup',      hint: 'Subjects, environments, topologies' },
];

export default function App() {
  useWebSocket();
  const [activeTab, setActiveTab] = useState<Tab>('room');
  const current = TABS.find(t => t.id === activeTab)!;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <div className="app-brand">
            <span className="app-brand-mark">CS</span>
            <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
              <span>CSI-Sense</span>
              <span className="app-brand-sub">Wi-Fi sensing console</span>
            </div>
          </div>
          <div className="app-header-spacer" style={{ flex: 1 }} />
          <div className="app-status-wrap" style={{ minWidth: 0, flex: '0 1 auto' }}>
            <StatusBar />
          </div>
        </div>
        <nav className="tabs" aria-label="Sections">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab ${activeTab === t.id ? 'active' : ''}`}
              onClick={() => setActiveTab(t.id)}
              title={t.hint}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main">
        <div style={{ marginBottom: 18 }}>
          <h1 style={{
            margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: '-0.01em',
          }}>
            {current.label}
          </h1>
          <p style={{
            margin: '2px 0 0', fontSize: 13, color: 'var(--text-muted)',
          }}>
            {current.hint}
          </p>
        </div>

        {activeTab === 'room' && (
          <div className="room-layout">
            <div className="room-map-wrap">
              <RoomMap />
            </div>
            <aside className="room-side">
              <ConfigPanel />
              <NodeListPanel />
            </aside>
          </div>
        )}

        {activeTab === 'experiment' && <ExperimentPanel />}
        {activeTab === 'labels'     && <LabelingPanel />}
        {activeTab === 'stats'      && <StatsPanel />}
        {activeTab === 'packets'    && <PacketViewer />}
        {activeTab === 'setup'      && <SetupPanel />}
      </main>
    </div>
  );
}
