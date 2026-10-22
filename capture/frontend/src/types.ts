export interface NodeInfo {
  mac: string;
  ip: string;
  status: "online" | "offline";
  position: [number, number] | null;
  chip: string;
  label: string | null;
}

export interface DetectionState {
  state: "calibrating" | "monitoring";
  occupied: boolean;
  occupancy: number;
  active_links: number;
  active_link_details: ActiveLink[];
  total_links: number;
  calibration_progress: number;
  total_samples: number;
}

export interface ActiveLink {
  tx: string;
  rx: string;
  z_score: number;
}

export interface ZoneInfo {
  zone_id: number;
  nodes: string[];
  centroid: [number, number];
}

export interface RoomConfig {
  width: number;
  height: number;
}

export interface PacketInfo {
  tx_mac: string;
  rx_mac: string;
  timestamp_us: number;
  seq: number;
  rssi: number;
  noise_floor: number;
  amplitude_variance: number;
  baseline_mean: number;
  baseline_std: number;
  baseline_samples: number;
  z_score: number | null;
  state: "calibrating" | "monitoring";
}

export interface Annotation {
  annotation_id: string;
  trial_id: string;
  start_ts: number;
  end_ts: number;
  duration_s: number;
  server_ts: number;
  coarse_label: string;
  fine_label: string;
  annotator_id: string;
  confidence: number;
  note: string;
  source: string;
}

export interface RecordingState {
  recording: boolean;
  trial_id: string;
  raw_written: number;
  cal_written: number;
  dropped: number;
  current_raw_file: string;
  compress: boolean;
  scheduled_stop_ts?: number | null;
  elapsed_s?: number;
  remaining_s?: number | null;
}

export interface ExperimentState {
  phase: 'idle' | 'baseline' | 'position' | 'active' | 'done';
  activity: string;
  elapsed_s: number;
  remaining_s: number;
  phase_duration_s: number;
  total_elapsed_s: number;
  activity_index: number;
  activity_count: number;
  beep?: 'position' | 'start' | 'done';
  config?: {
    activities: { activity: string; activity_s: number; note: string }[];
    activity: string;
    baseline_s: number;
    position_s: number;
    transition_s: number;
    activity_s: number;
    note: string;
  };
}

export interface NodeRate {
  tx_yield: number;
  rx_yield: number;
  avg_link_rate: number;
  min_link_rate: number;
  link_count: number;
  capture_rate: number;
}

export interface TdmaInfo {
  n_nodes: number;
  slot_ms: number;
  cycle_delay_ms: number;
  bursts_per_trigger: number;
  cycle_ms: number;
  cycles_per_sec: number;
  measured_cycles_per_sec: number;
}

export interface LinkBaseline {
  tx: string;
  rx: string;
  count: number;
  mean: number;
  std: number;
}

export interface StatsState {
  calibration_duration_s: number | null;
  calibrating: boolean;
  baseline_frozen: boolean;
  calibration_elapsed_s: number | null;
  total_links: number;
  calibration_samples: number;
  total_samples: number;
  node_rates: Record<string, NodeRate>;
  link_rates: Record<string, number>;
  link_baselines: Record<string, LinkBaseline>;
  baseline_target: number;
  avg_link_rate: number;
  min_link_rate: number;
  expected_link_rate: number;
  system_capture_rate: number;
  tdma: TdmaInfo;
}

export interface RoomFeature {
  id: string;
  type: "door" | "window" | "table" | "wall";
  x: number;
  y: number;
  width: number;
  height?: number;
  rotation: number;
  flip?: boolean;
}

export interface LocalizationEstimate {
  x: number;
  y: number;
  radius: number;
  n_intersections: number;
}

export interface SimulationState {
  x: number;
  y: number;
}

export interface LiveState {
  nodes: NodeInfo[];
  detection: DetectionState;
  room: RoomConfig;
  zones: ZoneInfo[];
  recent_packets?: PacketInfo[];
  recording?: RecordingState;
  experiment?: ExperimentState;
  stats?: StatsState;
  features?: RoomFeature[];
  localization?: LocalizationEstimate | null;
  simulation?: SimulationState | null;
}
