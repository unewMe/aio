import {
  InspectionResponse,
  InspectionHistoryItem,
  InspectionStats,
  ModelInfo,
  BenchmarkResponse,
  SampleListResponse,
} from './types';

// Browser calls use the Next.js rewrite or direct URL
const BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000/api/v1';

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE_URL}/models`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Nie udało się pobrać listy modeli.');
  const data = await res.json();
  return data.models;
}

export async function runBenchmark(
  modelName: string,
  iterations = 20
): Promise<BenchmarkResponse> {
  const res = await fetch(
    `${BASE_URL}/models/benchmark?model_name=${modelName}&iterations=${iterations}`,
    { method: 'POST' }
  );
  if (!res.ok) throw new Error('Błąd podczas pomiaru benchmarku.');
  return res.json();
}

export async function fetchSamples(): Promise<SampleListResponse> {
  const res = await fetch(`${BASE_URL}/samples`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Nie udało się pobrać próbek referencyjnych.');
  return res.json();
}

export function getSampleImageUrl(category: string, filename: string): string {
  return `${BASE_URL}/samples/${category}/${filename}`;
}

export function getFigureImageUrl(figureName: string): string {
  return `${BASE_URL}/analytics/figures/${figureName}`;
}

export async function inspectUploadedFile(
  file: File,
  modelName: string,
  threshold: number,
  notes?: string
): Promise<InspectionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);
  formData.append('threshold', threshold.toString());
  if (notes) formData.append('notes', notes);

  const res = await fetch(`${BASE_URL}/inspections/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Błąd podczas wykonywania inspekcji.');
  }
  return res.json();
}

export async function inspectSampleFile(
  category: string,
  filename: string,
  modelName: string,
  threshold: number,
  notes?: string
): Promise<InspectionResponse> {
  const formData = new FormData();
  formData.append('category', category);
  formData.append('filename', filename);
  formData.append('model_name', modelName);
  formData.append('threshold', threshold.toString());
  if (notes) formData.append('notes', notes);

  const res = await fetch(`${BASE_URL}/inspections/sample`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Błąd podczas inspekcji próbki.');
  }
  return res.json();
}

export async function fetchInspectionHistory(
  skip = 0,
  limit = 50
): Promise<InspectionHistoryItem[]> {
  const res = await fetch(
    `${BASE_URL}/inspections/history?skip=${skip}&limit=${limit}`,
    { cache: 'no-store' }
  );
  if (!res.ok) throw new Error('Nie udało się pobrać historii inspekcji.');
  return res.json();
}

export async function fetchInspectionStats(): Promise<InspectionStats> {
  const res = await fetch(`${BASE_URL}/inspections/stats`, {
    cache: 'no-store',
  });
  if (!res.ok) throw new Error('Nie udało się pobrać statystyk.');
  return res.json();
}

export async function clearInspectionHistory(): Promise<void> {
  const res = await fetch(`${BASE_URL}/inspections/clear`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Nie udało się wyczyścić historii.');
}

export async function fetchEvaluationReport(): Promise<any> {
  const res = await fetch(`${BASE_URL}/analytics/evaluation`, {
    cache: 'no-store',
  });
  if (!res.ok) throw new Error('Nie udało się pobrać raportu ewaluacyjnego.');
  return res.json();
}

export async function fetchTrainingHistory(): Promise<any> {
  const res = await fetch(`${BASE_URL}/analytics/training-history`, {
    cache: 'no-store',
  });
  if (!res.ok) throw new Error('Nie udało się pobrać historii treningu.');
  return res.json();
}

export async function fetchFiguresList(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/analytics/figures`, {
    cache: 'no-store',
  });
  if (!res.ok) return [];
  const data = await res.json();
  return data.figures || [];
}

export async function runSimulationStep(
  modelName = 'resnet18',
  threshold = 0.197,
  itemId?: number,
  categoryFilter = 'all'
): Promise<import('./types').SimulationStepResponse> {
  const itemParam = itemId !== undefined && itemId !== null ? `&item_id=${itemId}` : '';
  const filterParam = `&category_filter=${categoryFilter}`;
  const res = await fetch(
    `${BASE_URL}/simulation/step?model_name=${modelName}&threshold=${threshold}${itemParam}${filterParam}`,
    { method: 'POST' }
  );
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Błąd kroku symulacji.');
  }
  return res.json();
}

export async function fetchSimulationItems(): Promise<import('./types').SimulationItem[]> {
  const res = await fetch(`${BASE_URL}/simulation/items`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Nie udało się pobrać próbek testowych.');
  const data = await res.json();
  return data.items || [];
}

export function getSimulationImageUrl(itemId: number): string {
  return `${BASE_URL}/simulation/image/${itemId}`;
}

export async function fetchModelMemoryStatus(): Promise<import('./types').ModelMemoryStatus> {
  const res = await fetch(`${BASE_URL}/models/memory`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Nie udało się pobrać stanu pamięci RAM modeli.');
  return res.json();
}

export async function unloadModels(): Promise<{ status: string; loaded_models: string[]; rss_memory_mb: number }> {
  const res = await fetch(`${BASE_URL}/models/unload`, { method: 'POST' });
  if (!res.ok) throw new Error('Nie udało się zwolnić modeli z pamięci.');
  return res.json();
}

export async function fetchCpuConfig(): Promise<import('./types').CpuConfig> {
  const res = await fetch(`${BASE_URL}/models/cpu-config`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Nie udało się pobrać konfiguracji CPU.');
  return res.json();
}

export async function updateCpuConfig(payload: {
  profile?: string;
  threads?: number;
  throttle_percent?: number;
}): Promise<import('./types').CpuConfig> {
  const res = await fetch(`${BASE_URL}/models/cpu-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Nie udało się zapisać konfiguracji CPU.');
  return res.json();
}



