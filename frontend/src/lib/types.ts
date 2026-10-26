export interface InspectionResponse {
  id?: number;
  created_at?: string;
  filename: string;
  model_name: string;
  threshold: number;
  prob_defect: number;
  prob_ok: number;
  is_defective: boolean;
  verdict: 'DEFECT' | 'OK';
  recommendation: string;
  latency_ms: number;
  target_met_le_100ms: boolean;
  notes?: string;
}

export interface InspectionHistoryItem {
  id: number;
  created_at: string;
  filename: string;
  model_name: string;
  threshold: number;
  prob_defect: number;
  prob_ok: number;
  is_defective: boolean;
  verdict: 'DEFECT' | 'OK';
  recommendation: string;
  latency_ms: number;
}

export interface InspectionStats {
  total_inspections: number;
  total_defects: number;
  total_approved: number;
  defect_rate_percent: number;
  average_latency_ms: number;
}

export interface ModelInfo {
  id: string;
  display_name: string;
  role: string;
  description: string;
  optimal_threshold: number;
  has_weights: boolean;
  weights_file: string;
  parameters: number;
  size_mb: number;
  device: string;
}

export interface BenchmarkResponse {
  model_name: string;
  device: string;
  iterations: number;
  mean_latency_ms: number;
  median_latency_ms: number;
  p95_latency_ms: number;
  target_met_le_100ms: boolean;
}

export interface SampleItem {
  category: 'def' | 'ok';
  filename: string;
  label: string;
}

export interface SampleListResponse {
  def: SampleItem[];
  ok: SampleItem[];
}

export interface SimulationStepResponse {
  item_id: number;
  filename: string;
  filepath: string;
  model_name: string;
  threshold: number;
  prob_defect: number;
  prob_ok: number;
  is_defective: boolean;
  verdict: 'DEFECT' | 'OK';
  recommendation: string;
  latency_ms: number;
  ground_truth: 'def' | 'ok';
  ground_truth_label: number;
  ground_truth_name: string;
  is_correct: boolean;
  classification_type: 'TP' | 'TN' | 'FP' | 'FN';
  classification_badge: string;
  image_url: string;
  image_base64?: string;
  raw_latency_ms?: number;
  cpu_config?: {
    profile: string;
    profile_name: string;
    threads: number;
    throttle_percent: number;
    total_system_cores: number;
    is_throttled: boolean;
  };
}

export interface SimulationItem {
  id: number;
  filename: string;
  ground_truth: 'def' | 'ok';
  ground_truth_label: number;
  ground_truth_name: string;
}

export interface ModelMemoryStatus {
  strategy: string;
  max_loaded_models: number;
  loaded_models: string[];
  active_model: string | null;
  rss_memory_mb: number;
  device: string;
}

export interface CpuHardwareProfile {
  id: string;
  name: string;
  threads: number;
  throttle_percent: number;
  multiplier: number;
  description: string;
}

export interface CpuConfig {
  profile: string;
  profile_name: string;
  threads: number;
  throttle_percent: number;
  total_system_cores: number;
  is_throttled: boolean;
  profiles?: Record<string, CpuHardwareProfile>;
}


