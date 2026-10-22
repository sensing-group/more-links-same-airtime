import { create } from 'zustand';
import { LiveState, NodeInfo, DetectionState, RoomConfig, ZoneInfo, PacketInfo, RecordingState, ExperimentState, StatsState, RoomFeature, LocalizationEstimate, SimulationState } from './types';

interface AppState {
  connected: boolean;
  nodes: NodeInfo[];
  detection: DetectionState;
  room: RoomConfig;
  zones: ZoneInfo[];
  packets: PacketInfo[];
  recording: RecordingState;
  experiment: ExperimentState;
  stats: StatsState;
  features: RoomFeature[];
  localization: LocalizationEstimate | null;
  simulation: SimulationState | null;
  setConnected: (v: boolean) => void;
  updateFromWS: (data: LiveState) => void;
  setRoom: (room: RoomConfig) => void;
}

const defaultDetection: DetectionState = {
  state: "calibrating",
  occupied: false,
  occupancy: 0,
  active_links: 0,
  active_link_details: [],
  total_links: 0,
  calibration_progress: 0,
  total_samples: 0,
};

const defaultRecording: RecordingState = {
  recording: false,
  trial_id: '',
  raw_written: 0,
  cal_written: 0,
  dropped: 0,
  current_raw_file: '',
  compress: true,
};

const defaultExperiment: ExperimentState = {
  phase: 'idle',
  activity: '',
  elapsed_s: 0,
  remaining_s: 0,
  phase_duration_s: 0,
  total_elapsed_s: 0,
  activity_index: 0,
  activity_count: 0,
};

const defaultStats: StatsState = {
  calibration_duration_s: null,
  calibrating: true,
  baseline_frozen: false,
  calibration_elapsed_s: null,
  total_links: 0,
  total_samples: 0,
  node_rates: {},
  link_rates: {},
  link_baselines: {},
  baseline_target: 200,
  avg_link_rate: 0,
  min_link_rate: 0,
  expected_link_rate: 0,
  system_capture_rate: 0,
  calibration_samples: 0,
  tdma: { n_nodes: 0, slot_ms: 20, cycle_delay_ms: 5, bursts_per_trigger: 10, cycle_ms: 0, cycles_per_sec: 0, measured_cycles_per_sec: 0 },
};

export const useStore = create<AppState>((set) => ({
  connected: false,
  nodes: [],
  detection: defaultDetection,
  room: { width: 5, height: 4 },
  zones: [],
  packets: [],
  recording: defaultRecording,
  experiment: defaultExperiment,
  stats: defaultStats,
  features: [],
  localization: null,
  simulation: null,
  setConnected: (connected) => set({ connected }),
  updateFromWS: (data) => set((state) => ({
    nodes: data.nodes ?? state.nodes,
    detection: data.detection ?? state.detection,
    room: data.room ?? state.room,
    zones: data.zones ?? state.zones,
    packets: data.recent_packets ?? state.packets,
    recording: data.recording ?? state.recording,
    experiment: data.experiment ?? state.experiment,
    stats: data.stats ?? state.stats,
    features: data.features ?? state.features,
    localization: data.localization ?? null,
    simulation: data.simulation ?? null,
  })),
  setRoom: (room) => set({ room }),
}));
