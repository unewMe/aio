'use client';

import React, { useState, useEffect } from 'react';
import { fetchModels, runBenchmark, fetchCpuConfig, updateCpuConfig } from '@/lib/api';
import { ModelInfo, BenchmarkResponse, CpuConfig } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Cpu, Zap, CheckCircle2 } from 'lucide-react';

export default function ModelsBenchmarkPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  
  const [cpuConfig, setCpuConfig] = useState<CpuConfig | null>(null);
  const [cpuLoading, setCpuLoading] = useState(false);

  const [benchmarkingModel, setBenchmarkingModel] = useState<string | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);
  const [iterations] = useState<number>(20);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  const refreshAll = async () => {
    try {
      const [data, cpuData] = await Promise.all([
        fetchModels(),
        fetchCpuConfig().catch(() => null),
      ]);
      setModels(data);
      if (cpuData) setCpuConfig(cpuData);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshAll();
  }, []);

  const handleCpuProfileChange = async (profileId: string) => {
    setCpuLoading(true);
    try {
      const updated = await updateCpuConfig({ profile: profileId });
      setCpuConfig(updated);
    } catch (err) {
      console.error(err);
    } finally {
      setCpuLoading(false);
    }
  };

  const handleBenchmark = async (modelId: string) => {
    setBenchmarkingModel(modelId);
    setBenchmarkError(null);
    try {
      const res = await runBenchmark(modelId, iterations);
      setBenchmarkResult(res);
    } catch (err: any) {
      setBenchmarkError(err.message || 'Błąd pomiaru opóźnienia.');
    } finally {
      setBenchmarkingModel(null);
    }
  };

  const evaluationStats: Record<string, any> = {
    resnet18: {
      badge: 'Lider Dokładności',
      recall: '100.0%',
      precision: '99.50%',
      f1: '0.9975',
      fn: '0 / 600',
      cpuLatency: '62.5 ms',
      highlight: true,
    },
    mobilenet_v2: {
      badge: 'Optimum Edge AI',
      recall: '100.0%',
      precision: '98.20%',
      f1: '0.9909',
      fn: '0 / 600',
      cpuLatency: '59.2 ms',
      highlight: false,
    },
    cnn: {
      badge: 'Najszybsza Inferencja',
      recall: '100.0%',
      precision: '97.72%',
      f1: '0.9885',
      fn: '0 / 600',
      cpuLatency: '34.5 ms',
      highlight: false,
    },
  };

  return (
    <div className="space-y-7">
      {/* Header & Telemetry Cards */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div>
          <Badge variant="cyan" className="mb-2.5 gap-1.5">
            <Cpu className="w-3.5 h-3.5" />
            REJESTR MODELI & BENCHMARK INFERENCJI
          </Badge>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white mb-2">
            Porównanie Architektur Sieci Neuronowych
          </h1>
          <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
            Wszystkie 3 modele wytrenowano w identycznych warunkach na 4800 obrazach treningowych
            i przetestowano na 1200 obrazach testowych. Modele ładowane są do pamięci <b>On-Demand</b> (tylko 1 aktywny model naraz w RAM).
          </p>
        </div>

        {/* Telemetry Cards Row */}
        <div className="flex flex-wrap items-center gap-3">
          {/* CPU Hardware Profile Selector */}
          <Card className="p-3.5 bg-slate-950/80 border-white/10 shadow-lg min-w-[280px]">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Cpu className={`w-3.5 h-3.5 ${cpuConfig?.is_throttled ? 'text-amber-400 animate-pulse' : 'text-emerald-400'}`} />
                <span>Moc CPU (Benchmark):</span>
              </span>
              <Badge variant={cpuConfig?.is_throttled ? 'crimson' : 'emerald'} className="text-[10px] py-0 px-1 font-mono">
                {cpuConfig?.is_throttled ? 'DŁAWIONY' : '100% HOST'}
              </Badge>
            </div>
            <div className="text-xs text-slate-400 space-y-1 font-mono mb-2">
              <div className="flex justify-between">
                <span>Profil:</span>
                <b className="text-white truncate max-w-[160px]">{cpuConfig?.profile_name?.split('(')[0]?.trim()}</b>
              </div>
              <div className="flex justify-between">
                <span>Alokacja wątków:</span>
                <span className="text-cyan-300 font-bold">{cpuConfig?.threads ?? 1}w @ {cpuConfig?.throttle_percent ?? 100}% mocy</span>
              </div>
            </div>
            {/* Quick Profile Selector Buttons */}
            <div className="grid grid-cols-3 gap-1 pt-1 border-t border-white/5">
              <Button
                variant={cpuConfig?.profile === 'unthrottled' ? 'secondary' : 'outline'}
                size="sm"
                disabled={cpuLoading}
                onClick={() => handleCpuProfileChange('unthrottled')}
                className="h-6 text-[10px] px-1 font-semibold"
                title="Pełna moc komputera PC"
              >
                Serwer
              </Button>
              <Button
                variant={cpuConfig?.profile === 'industrial_pc' ? 'secondary' : 'outline'}
                size="sm"
                disabled={cpuLoading}
                onClick={() => handleCpuProfileChange('industrial_pc')}
                className="h-6 text-[10px] px-1 font-semibold"
                title="Przemysłowy IPC (60%)"
              >
                IPC
              </Button>
              <Button
                variant={cpuConfig?.profile === 'raspberry_pi_4' ? 'secondary' : 'outline'}
                size="sm"
                disabled={cpuLoading}
                onClick={() => handleCpuProfileChange('raspberry_pi_4')}
                className="h-6 text-[10px] px-1 font-semibold"
                title="Raspberry Pi 4 (18%)"
              >
                RPi 4
              </Button>
            </div>
          </Card>

          {/* RAM Telemetry Card */}
          <Card className="p-3.5 bg-slate-950/80 border-white/10 shadow-lg min-w-[220px]">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <span>Optymalizacja RAM:</span>
              </span>
              <Badge variant="cyan" className="text-[10px] py-0 px-1 font-mono">
                ON-DEMAND
              </Badge>
            </div>
            <div className="text-xs text-slate-400 space-y-1 font-mono">
              <div className="flex justify-between">
                <span>Pamięć procesu ML:</span>
                <b className="text-white">~165 MB RSS</b>
              </div>
              <div className="flex justify-between">
                <span>Polityka alokacji:</span>
                <span className="text-cyan-300">Pojedynczy slot</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {models.map((m) => {
          const stats = evaluationStats[m.id] || {};
          const isBenchmarking = benchmarkingModel === m.id;

          return (
            <Card
              key={m.id}
              className={`p-6 flex flex-col justify-between transition-all duration-300 ${
                stats.highlight
                  ? 'border-cyan-500/50 shadow-2xl shadow-cyan-500/20 bg-slate-900/90'
                  : 'bg-slate-900/60'
              }`}
            >
              <div>
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-xl font-extrabold text-white">
                    {m.display_name}
                  </h3>
                  <Badge variant={stats.highlight ? 'cyan' : 'secondary'}>
                    {stats.badge || m.role}
                  </Badge>
                </div>

                <p className="text-xs text-slate-400 leading-relaxed min-h-[48px] mb-4">
                  {m.description}
                </p>

                {/* Metrics Table */}
                <div className="bg-slate-950/70 border border-white/5 rounded-xl p-3.5 space-y-2 text-xs mb-5">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Recall (Wykrywanie wad):</span>
                    <span className="font-bold text-emerald-400">{stats.recall}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Precision (Precyzja):</span>
                    <span className="font-bold text-emerald-400">{stats.precision}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">F1-Score:</span>
                    <span className="font-bold text-white font-mono">{stats.f1}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Przeoczone wady (FN):</span>
                    <span className="font-bold text-emerald-400">{stats.fn}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Liczba parametrów:</span>
                    <span className="font-semibold text-white font-mono">
                      {m.parameters ? m.parameters.toLocaleString() : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Rozmiar wag (.pth):</span>
                    <span className="font-semibold text-white font-mono">{m.size_mb} MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Optymalny próg &tau;*:</span>
                    <span className="font-bold text-cyan-400 font-mono">
                      {m.optimal_threshold.toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <Button
                variant={stats.highlight ? 'cyan' : 'outline'}
                size="default"
                disabled={isBenchmarking}
                onClick={() => handleBenchmark(m.id)}
                className="w-full text-xs font-bold gap-2"
              >
                {isBenchmarking ? (
                  <>
                    <span className="live-pulse" />
                    <span>MIERZENIE OPÓŹNIENIA (20 RUNS)...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    <span>Uruchom Live Benchmark CPU</span>
                  </>
                )}
              </Button>
            </Card>
          );
        })}
      </div>

      {/* Benchmark Error */}
      {benchmarkError && (
        <div className="bg-red-500/15 border border-red-500/30 p-4 rounded-xl text-red-400 text-sm">
          {benchmarkError}
        </div>
      )}

      {/* Live Benchmark Result */}
      {benchmarkResult && (
        <Card className="p-6 border-emerald-500/40 bg-gradient-to-br from-emerald-950/20 via-slate-900/80 to-slate-950/90 shadow-2xl">
          <div className="flex items-center gap-2.5 text-emerald-400 text-lg font-bold mb-4">
            <CheckCircle2 className="w-6 h-6" />
            <span>
              Wynik Pomiaru Benchmarku: {benchmarkResult.model_name.toUpperCase()} (
              {benchmarkResult.iterations} powtórzeń na urządzeniu {benchmarkResult.device.toUpperCase()})
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
            <Card className="p-4 bg-slate-950/60">
              <div className="text-xs uppercase text-slate-400 mb-1">Średnia (Mean)</div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {benchmarkResult.mean_latency_ms.toFixed(2)} ms
              </div>
            </Card>
            <Card className="p-4 bg-slate-950/60">
              <div className="text-xs uppercase text-slate-400 mb-1">Mediana (P50)</div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {benchmarkResult.median_latency_ms.toFixed(2)} ms
              </div>
            </Card>
            <Card className="p-4 bg-slate-950/60">
              <div className="text-xs uppercase text-slate-400 mb-1">Percentyl 95 (P95)</div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {benchmarkResult.p95_latency_ms.toFixed(2)} ms
              </div>
            </Card>
            <Card className="p-4 bg-slate-950/60">
              <div className="text-xs uppercase text-slate-400 mb-1">Norma &le; 100 ms</div>
              <Badge
                variant={benchmarkResult.target_met_le_100ms ? 'emerald' : 'crimson'}
                className="mt-1 font-bold text-sm"
              >
                {benchmarkResult.target_met_le_100ms ? 'SPEŁNIONE' : 'PRZEKROCZONE'}
              </Badge>
            </Card>
          </div>
        </Card>
      )}
    </div>
  );
}
