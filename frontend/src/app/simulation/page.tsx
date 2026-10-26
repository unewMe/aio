'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  runSimulationStep,
  getSimulationImageUrl,
  fetchModelMemoryStatus,
  unloadModels,
  fetchCpuConfig,
  updateCpuConfig,
} from '@/lib/api';
import { SimulationStepResponse, ModelMemoryStatus, CpuConfig } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import {
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  CheckCircle2,
  AlertOctagon,
  Clock,
  Cpu,
  Layers,
  Radio,
  Sliders,
  Sparkles,
  Zap,
  Volume2,
  VolumeX,
  ShieldAlert,
  Flame,
  Settings2,
  Gauge,
  Activity,
  ArrowRight,
  ShieldCheck,
  Percent,
  Database,
  Trash2,
  RefreshCw,
  HardDrive,
  Flag,
  Shuffle,
  Repeat,
} from 'lucide-react';

const OPTIMAL_THRESHOLDS: Record<string, number> = {
  resnet18: 0.197,
  mobilenet_v2: 0.099,
  cnn: 0.158,
};

const TOTAL_TEST_SET_ITEMS = 1200;

// Utility to create a shuffled array of 0..1199
function createShuffledIndices(count = TOTAL_TEST_SET_ITEMS): number[] {
  const arr = Array.from({ length: count }, (_, i) => i);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export default function ConveyorSimulationPage() {
  // Model & Threshold
  const [modelName, setModelName] = useState<string>('resnet18');
  const [threshold, setThreshold] = useState<number>(0.197);

  // Speed & Cadence Configuration (DOWOLNY CZAS NAPŁYWU)
  const [intervalMs, setIntervalMs] = useState<number>(500); // Domyślnie 500 ms
  const [customIntervalInput, setCustomIntervalInput] = useState<string>('500');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);

  // 1200 Items Dataset Traversal State
  const [testQueue, setTestQueue] = useState<number[]>(() => createShuffledIndices(TOTAL_TEST_SET_ITEMS));
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [stopAt1200, setStopAt1200] = useState<boolean>(true); // Domyślnie zatrzymuj po 1200
  const [shuffleOrder, setShuffleOrder] = useState<boolean>(true);
  const [isBatchFinished, setIsBatchFinished] = useState<boolean>(false);

  // Stream & Defect Rate Configuration (Profil napływu)
  const [categoryFilter, setCategoryFilter] = useState<string>('all'); // 'all' | 'realistic_5' | 'realistic_10' | 'realistic_20' | 'def' | 'ok'
  const [stopOnFN, setStopOnFN] = useState<boolean>(true); // Auto-stop on critical False Negative
  const [stopOnFP, setStopOnFP] = useState<boolean>(false); // Auto-stop on False Positive
  const [soundEnabled, setSoundEnabled] = useState<boolean>(false);
  const [showConfig, setShowConfig] = useState<boolean>(true);

  // RAM Telemetry & On-Demand Model Management
  const [memoryStatus, setMemoryStatus] = useState<ModelMemoryStatus | null>(null);
  const [unloadingMemory, setUnloadingMemory] = useState<boolean>(false);

  // Current item and history
  const [currentStep, setCurrentStep] = useState<SimulationStepResponse | null>(null);
  const [recentFeed, setRecentFeed] = useState<SimulationStepResponse[]>([]);

  // Session KPIs
  const [stats, setStats] = useState({
    total: 0,
    tp: 0,
    tn: 0,
    fp: 0,
    fn: 0,
    totalLatency: 0,
  });

  // REFS for bulletproof, drift-free loop execution and INSTANT PAUSE
  const isRunningRef = useRef<boolean>(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const inFlightRef = useRef<boolean>(false);
  const intervalMsRef = useRef<number>(intervalMs);
  intervalMsRef.current = intervalMs;
  const modelNameRef = useRef<string>(modelName);
  modelNameRef.current = modelName;
  const thresholdRef = useRef<number>(threshold);
  thresholdRef.current = threshold;
  const categoryFilterRef = useRef<string>(categoryFilter);
  categoryFilterRef.current = categoryFilter;
  const currentIndexRef = useRef<number>(currentIndex);
  currentIndexRef.current = currentIndex;
  const testQueueRef = useRef<number[]>(testQueue);
  testQueueRef.current = testQueue;
  const stopAt1200Ref = useRef<boolean>(stopAt1200);
  stopAt1200Ref.current = stopAt1200;
  const stopOnFNRef = useRef<boolean>(stopOnFN);
  stopOnFNRef.current = stopOnFN;
  const stopOnFPRef = useRef<boolean>(stopOnFP);
  stopOnFPRef.current = stopOnFP;

  // Sound cue using Web Audio API
  const playBeep = (type: 'defect' | 'ok' | 'fn') => {
    if (!soundEnabled || typeof window === 'undefined') return;
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      if (type === 'fn') {
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(220, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.35);
      } else if (type === 'defect') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.15);
      } else {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.08);
      }
    } catch {
      // AudioContext restricted
    }
  };

  // Fetch RAM telemetry
  const refreshMemory = async () => {
    try {
      const data = await fetchModelMemoryStatus();
      setMemoryStatus(data);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    refreshMemory();
  }, [modelName]);

  const handleUnloadMemory = async () => {
    setUnloadingMemory(true);
    try {
      await unloadModels();
      await refreshMemory();
    } catch (err) {
      console.error('Error unloading models:', err);
    } finally {
      setUnloadingMemory(false);
    }
  };

  // CPU Hardware Throttling State & Handlers
  const [cpuConfig, setCpuConfig] = useState<CpuConfig | null>(null);
  const [cpuLoading, setCpuLoading] = useState<boolean>(false);
  const [cpuThreads, setCpuThreads] = useState<number>(1);
  const [cpuThrottlePercent, setCpuThrottlePercent] = useState<number>(100);

  const refreshCpu = async () => {
    try {
      const cfg = await fetchCpuConfig();
      setCpuConfig(cfg);
      setCpuThreads(cfg.threads);
      setCpuThrottlePercent(cfg.throttle_percent);
    } catch (err) {
      console.error('Error fetching CPU config:', err);
    }
  };

  useEffect(() => {
    refreshCpu();
  }, []);

  const handleSelectCpuProfile = async (profileId: string) => {
    setCpuLoading(true);
    try {
      const cfg = await updateCpuConfig({ profile: profileId });
      setCpuConfig(cfg);
      setCpuThreads(cfg.threads);
      setCpuThrottlePercent(cfg.throttle_percent);
    } catch (err) {
      console.error('Error setting CPU profile:', err);
    } finally {
      setCpuLoading(false);
    }
  };

  const handleUpdateCustomCpu = async (threads: number, throttlePct: number) => {
    setCpuThreads(threads);
    setCpuThrottlePercent(throttlePct);
    setCpuLoading(true);
    try {
      const cfg = await updateCpuConfig({
        profile: 'custom',
        threads,
        throttle_percent: throttlePct,
      });
      setCpuConfig(cfg);
    } catch (err) {
      console.error('Error updating custom CPU:', err);
    } finally {
      setCpuLoading(false);
    }
  };

  // Synchronize cadence state with arbitrary text input
  const updateCadence = (newMs: number) => {
    const safeMs = Math.max(5, newMs);
    setIntervalMs(safeMs);
    setCustomIntervalInput(safeMs.toString());
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const valStr = e.target.value;
    setCustomIntervalInput(valStr);
    const parsed = parseInt(valStr, 10);
    if (!isNaN(parsed) && parsed > 0) {
      setIntervalMs(parsed);
    }
  };

  const handleModelChange = (modelId: string) => {
    setModelName(modelId);
    if (OPTIMAL_THRESHOLDS[modelId]) {
      setThreshold(OPTIMAL_THRESHOLDS[modelId]);
    }
  };

  // BULLETPROOF STOP SIMULATION - Guaranteed to stop immediately without leaks
  const stopSimulation = useCallback(() => {
    isRunningRef.current = false;
    setIsRunning(false);
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // EXECUTE A SINGLE STEP
  const executeStep = async (isLoopCall = false): Promise<boolean> => {
    if (inFlightRef.current) return false;
    inFlightRef.current = true;
    setLoading(true);

    try {
      // Check if dataset has reached 1200
      const currentIdx = currentIndexRef.current;
      const queue = testQueueRef.current;

      if (currentIdx >= queue.length) {
        if (stopAt1200Ref.current) {
          stopSimulation();
          setIsBatchFinished(true);
          return false;
        } else {
          // Loop mode: reset index to 0
          currentIndexRef.current = 0;
          setCurrentIndex(0);
        }
      }

      // If loop was stopped while waiting, abort
      if (isLoopCall && !isRunningRef.current) {
        return false;
      }

      const itemId = queue[currentIndexRef.current] ?? 0;
      const step = await runSimulationStep(
        modelNameRef.current,
        thresholdRef.current,
        itemId,
        categoryFilterRef.current
      );

      // Post-fetch guard: If user clicked Pause while the HTTP request was over the network, DO NOT CONTINUE!
      if (isLoopCall && !isRunningRef.current) {
        return false;
      }

      // Update state
      setCurrentStep(step);
      setRecentFeed((prev) => [step, ...prev.slice(0, 49)]);

      const isTP = step.classification_type === 'TP';
      const isTN = step.classification_type === 'TN';
      const isFP = step.classification_type === 'FP';
      const isFN = step.classification_type === 'FN';

      // Play audio cue
      if (isFN) {
        playBeep('fn');
      } else if (step.is_defective) {
        playBeep('defect');
      } else {
        playBeep('ok');
      }

      // Advance index in 1200 dataset
      const nextIdx = currentIndexRef.current + 1;
      currentIndexRef.current = nextIdx;
      setCurrentIndex(nextIdx);

      // Update stats
      setStats((prev) => ({
        total: prev.total + 1,
        tp: prev.tp + (isTP ? 1 : 0),
        tn: prev.tn + (isTN ? 1 : 0),
        fp: prev.fp + (isFP ? 1 : 0),
        fn: prev.fn + (isFN ? 1 : 0),
        totalLatency: prev.totalLatency + step.latency_ms,
      }));

      // Check safety auto-stop triggers
      if (stopOnFNRef.current && isFN) {
        stopSimulation();
        return false;
      }
      if (stopOnFPRef.current && isFP) {
        stopSimulation();
        return false;
      }

      // Check 1200 items completion
      if (nextIdx >= queue.length) {
        if (stopAt1200Ref.current) {
          stopSimulation();
          setIsBatchFinished(true);
          return false;
        }
      }

      return true;
    } catch (err) {
      console.error('Simulation step error:', err);
      return false;
    } finally {
      inFlightRef.current = false;
      setLoading(false);
    }
  };

  // DYNAMIC CHIP-COMPENSATED LOOP: Ensures smooth rhythm without request pileup
  const scheduleLoopStep = useCallback(() => {
    if (!isRunningRef.current) return;

    const tStart = performance.now();
    executeStep(true).then((continued) => {
      if (!continued || !isRunningRef.current) return;

      const tElapsed = performance.now() - tStart;
      const targetDelay = intervalMsRef.current;
      // Delay compensation: smoothly deduct inference time from target cadence
      const delay = Math.max(10, targetDelay - tElapsed);

      if (isRunningRef.current) {
        timerRef.current = setTimeout(() => {
          scheduleLoopStep();
        }, delay);
      }
    });
  }, [stopSimulation]);

  // START / PAUSE TOGGLE
  const toggleSimulation = () => {
    if (isRunning) {
      stopSimulation();
    } else {
      // If batch was finished at 1200, restart from 0
      if (currentIndex >= testQueue.length || isBatchFinished) {
        setCurrentIndex(0);
        currentIndexRef.current = 0;
        setIsBatchFinished(false);
        if (shuffleOrder) {
          const fresh = createShuffledIndices(TOTAL_TEST_SET_ITEMS);
          setTestQueue(fresh);
          testQueueRef.current = fresh;
        }
      }
      isRunningRef.current = true;
      setIsRunning(true);
      scheduleLoopStep();
    }
  };

  // FULL RESET
  const handleReset = () => {
    stopSimulation();
    setCurrentStep(null);
    setRecentFeed([]);
    setCurrentIndex(0);
    currentIndexRef.current = 0;
    setIsBatchFinished(false);
    const fresh = shuffleOrder ? createShuffledIndices(TOTAL_TEST_SET_ITEMS) : Array.from({ length: TOTAL_TEST_SET_ITEMS }, (_, i) => i);
    setTestQueue(fresh);
    testQueueRef.current = fresh;
    setStats({
      total: 0,
      tp: 0,
      tn: 0,
      fp: 0,
      fn: 0,
      totalLatency: 0,
    });
  };

  // Reshuffle queue
  const handleReshuffle = () => {
    const fresh = createShuffledIndices(TOTAL_TEST_SET_ITEMS);
    setTestQueue(fresh);
    testQueueRef.current = fresh;
    setCurrentIndex(0);
    currentIndexRef.current = 0;
    setIsBatchFinished(false);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSimulation();
    };
  }, [stopSimulation]);

  const accuracyPercent =
    stats.total > 0
      ? (((stats.tp + stats.tn) / stats.total) * 100).toFixed(1)
      : '0.0';

  const recallPercent =
    stats.tp + stats.fn > 0
      ? ((stats.tp / (stats.tp + stats.fn)) * 100).toFixed(1)
      : '100.0';

  const precisionPercent =
    stats.tp + stats.fp > 0
      ? ((stats.tp / (stats.tp + stats.fp)) * 100).toFixed(1)
      : '100.0';

  const avgLatency =
    stats.total > 0 ? (stats.totalLatency / stats.total).toFixed(1) : '0.0';

  // Items per minute throughput calculation
  const throughputPerMin = intervalMs > 0 ? Math.round((60 * 1000) / intervalMs) : 0;
  const throughputPerSec = intervalMs > 0 ? (1000 / intervalMs).toFixed(1) : '0.0';

  // Speed Tier
  const getSpeedTier = (ms: number) => {
    if (ms <= 50) return { label: 'TURBO EKSTREMALNY', color: 'border-red-500 text-red-400 bg-red-500/10' };
    if (ms <= 200) return { label: 'WYSOKA PRĘDKOŚĆ', color: 'border-amber-500 text-amber-400 bg-amber-500/10' };
    if (ms <= 600) return { label: 'STANDARDOWA TAŚMA', color: 'border-cyan-500 text-cyan-400 bg-cyan-500/10' };
    return { label: 'TRYB DETALICZNY', color: 'border-slate-500 text-slate-300 bg-slate-800/40' };
  };

  const speedTier = getSpeedTier(intervalMs);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col 2xl:flex-row justify-between items-start 2xl:items-center gap-4">
        <div className="max-w-3xl">
          <Badge variant="cyan" className="mb-2.5 gap-1.5">
            <Radio className="w-3.5 h-3.5 animate-pulse" />
            SYMULACJA STRUMIENIA PRODUKCYJNEGO (1200 ODLEWÓW TESTOWYCH)
          </Badge>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white mb-1">
            Automatyczna Inspekcja w Ruchu Ciągłym
          </h1>
          <p className="text-sm text-slate-400">
            Inspekcja w czasie rzeczywistym z płynnym przesyłem obrazów (Zero-Latency Base64) i natychmiastowym wstrzymaniem taśmy.
            Modele sieci neuronowych ładowane są <b>On-Demand</b> (oszczędność pamięci RAM).
          </p>
        </div>

        {/* Live Status indicator, CPU Hardware & RAM Optimization Badge */}
        <div className="flex flex-wrap items-center gap-2.5 sm:gap-3 shrink-0">
          {/* CPU Hardware Status Badge */}
          <div className="flex items-center gap-2.5 bg-slate-900/90 border border-white/10 px-3.5 py-2 rounded-xl text-xs font-mono shadow-md shrink-0 whitespace-nowrap">
            <Cpu className={`w-3.5 h-3.5 shrink-0 ${cpuConfig?.is_throttled ? 'text-amber-400 animate-pulse' : 'text-emerald-400'}`} />
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-400 uppercase font-semibold leading-tight">
                Moc CPU
              </span>
              <div className="text-white font-bold flex items-center gap-1.5 leading-tight mt-0.5">
                <span>{cpuConfig?.threads ?? 1}w &bull; {cpuConfig?.throttle_percent ?? 100}%</span>
                {cpuConfig?.is_throttled ? (
                  <span className="text-amber-400 text-[10px] font-normal">
                    ({cpuConfig?.profile === 'raspberry_pi_4' ? 'RPi 4' : cpuConfig?.profile === 'ultra_low_power' ? 'Ultra IoT' : cpuConfig?.profile_name?.split('(')[0]?.trim() || 'Edge'})
                  </span>
                ) : (
                  <span className="text-emerald-400 text-[10px] font-normal">(100%)</span>
                )}
              </div>
            </div>
          </div>

          {/* RAM Telemetry Badge */}
          <div className="flex items-center gap-2.5 bg-slate-900/90 border border-white/10 px-3.5 py-2 rounded-xl text-xs font-mono shadow-md shrink-0 whitespace-nowrap">
            <Database className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-400 uppercase font-semibold leading-tight">
                RAM Mikroserwisu ML
              </span>
              <div className="text-white font-bold flex items-center gap-1.5 leading-tight mt-0.5">
                <span>{memoryStatus?.rss_memory_mb ? `${memoryStatus.rss_memory_mb} MB` : 'On-Demand'}</span>
                {memoryStatus?.active_model && (
                  <span className="text-cyan-400 text-[10px] font-normal">({memoryStatus.active_model})</span>
                )}
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleUnloadMemory}
              disabled={unloadingMemory || isRunning}
              className="h-6 px-2 text-[10px] text-slate-400 hover:text-red-400 hover:bg-red-500/10 border border-white/5 hover:border-red-500/20 rounded-lg shrink-0 ml-1 transition-colors"
              title="Zwolnij wagi wszystkich modeli z RAM"
            >
              <Trash2 className="w-3 h-3 mr-1" />
              Zwolnij
            </Button>
          </div>

          {/* Andon Conveyor Status */}
          <div className="flex items-center gap-3 bg-slate-900/90 border border-white/10 px-3.5 py-2 rounded-xl shadow-lg shrink-0 whitespace-nowrap">
            <div
              className={`w-3.5 h-3.5 rounded-full shrink-0 transition-all ${
                isRunning
                  ? currentStep?.classification_type === 'FN'
                    ? 'bg-red-500 shadow-lg shadow-red-500 animate-ping'
                    : currentStep?.is_defective
                    ? 'bg-amber-400 shadow-md shadow-amber-400 animate-pulse'
                    : 'bg-emerald-400 shadow-md shadow-emerald-400'
                  : isBatchFinished
                  ? 'bg-cyan-400 shadow-md shadow-cyan-400'
                  : 'bg-slate-600'
              }`}
            />
            <div className="text-xs">
              <div className="font-bold text-white uppercase tracking-wider flex items-center gap-2 leading-tight">
                <span>
                  {isRunning
                    ? 'TAŚMA W RUCHU'
                    : isBatchFinished
                    ? 'PARTIA UKOŃCZONA'
                    : 'TAŚMA WSTRZYMANA'}
                </span>
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-bold border shrink-0 ${speedTier.color}`}>
                  {speedTier.label}
                </span>
              </div>
              <div className="text-slate-400 font-mono text-[11px] flex items-center gap-1.5 leading-tight mt-1">
                <span>KADENCJA: <b className="text-slate-200">{intervalMs} ms</b></span>
                <span className="text-slate-600">&bull;</span>
                <span>~<b className="text-slate-200">{throughputPerMin}</b> szt./min</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Finished 1200 Batch Banner */}
      {isBatchFinished && (
        <div className="p-4 rounded-xl bg-cyan-950/70 border-2 border-cyan-500/60 shadow-2xl flex flex-col md:flex-row items-center justify-between gap-4 animate-in fade-in">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400">
              <Flag className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">
                Zakończono pełną partię kontrolną 1200 odlewów ze zbioru testowego!
              </h2>
              <p className="text-xs text-slate-300">
                Wszystkie 1200 próbek wzorcowych zostało zweryfikowanych. Trafność sesji: <b>{accuracyPercent}%</b> (Wykrywalność: <b>{recallPercent}%</b>).
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="cyan" size="sm" onClick={handleReshuffle} className="gap-1.5 font-bold text-xs">
              <Shuffle className="w-3.5 h-3.5" />
              <span>Przetasuj i Rozpocznij Nowy Cykl (1200)</span>
            </Button>
            <Button variant="outline" size="sm" onClick={handleReset} className="text-xs">
              Resetuj Statystyki
            </Button>
          </div>
        </div>
      )}

      {/* Main Control & Quick Action Bar */}
      <Card className="p-4 bg-slate-950/80 border-white/10 shadow-xl">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
          {/* Main Action buttons */}
          <div className="md:col-span-5 flex flex-wrap gap-2">
            <Button
              variant={isRunning ? 'secondary' : 'cyan'}
              size="default"
              onClick={toggleSimulation}
              className="gap-2 font-bold flex-1"
            >
              {isRunning ? (
                <>
                  <Pause className="w-4 h-4 text-amber-400" />
                  <span>Wstrzymaj Taśmę</span>
                </>
              ) : isBatchFinished ? (
                <>
                  <RotateCcw className="w-4 h-4 fill-white" />
                  <span>Nowy Cykl 1200</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Uruchom Symulację</span>
                </>
              )}
            </Button>

            <Button
              variant="outline"
              size="default"
              disabled={isRunning || loading}
              onClick={() => executeStep(false)}
              className="gap-2 text-xs"
            >
              <SkipForward className="w-4 h-4 text-cyan-400" />
              <span>Następny Odlew</span>
            </Button>

            <Button
              variant="ghost"
              size="default"
              disabled={stats.total === 0}
              onClick={handleReset}
              className="gap-1.5 text-xs text-slate-400 hover:text-white"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </Button>

            {/* Toggle Config Panel Button */}
            <Button
              variant="ghost"
              size="default"
              onClick={() => setShowConfig(!showConfig)}
              className={`gap-1.5 text-xs ${
                showConfig ? 'text-cyan-400 bg-cyan-500/10' : 'text-slate-400 hover:text-white'
              }`}
            >
              <Settings2 className="w-4 h-4" />
              <span>Konfiguracja Napływu</span>
            </Button>
          </div>

          {/* Model Architecture Quick Select (On-Demand) */}
          <div className="md:col-span-4 flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap">
              Model (On-Demand):
            </span>
            <select
              value={modelName}
              onChange={(e) => handleModelChange(e.target.value)}
              className="w-full bg-slate-900 border border-white/10 rounded-lg text-white px-3 py-1.5 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <option value="resnet18">ResNet18 (Transfer Learning - Falsz 0%)</option>
              <option value="mobilenet_v2">MobileNetV2 (Edge AI - 13.8 ms)</option>
              <option value="cnn">Custom CNN (Baseline - 3.8 MB)</option>
            </select>
          </div>

          {/* Sound toggle & Cadence info */}
          <div className="md:col-span-3 flex items-center justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`text-xs gap-1.5 ${
                soundEnabled ? 'border-cyan-500/40 text-cyan-400' : 'text-slate-500'
              }`}
              title="Sygnalizacja dźwiękowa przy detekcji wady i alarmie FN"
            >
              {soundEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
              <span>{soundEnabled ? 'Dźwięk WŁ' : 'Dźwięk WYŁ'}</span>
            </Button>
          </div>
        </div>

        {/* 1200 Items Progress Bar */}
        <div className="mt-3 pt-3 border-t border-white/10 space-y-1.5">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-slate-300 font-semibold flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>
                Postęp partii testowej:{' '}
                <b className="text-white">
                  {Math.min(currentIndex, TOTAL_TEST_SET_ITEMS)} / {TOTAL_TEST_SET_ITEMS}
                </b>{' '}
                odlewów
              </span>
            </span>
            <span className="text-cyan-400 font-bold">
              {((Math.min(currentIndex, TOTAL_TEST_SET_ITEMS) / TOTAL_TEST_SET_ITEMS) * 100).toFixed(1)}%
            </span>
          </div>
          <Progress
            value={(Math.min(currentIndex, TOTAL_TEST_SET_ITEMS) / TOTAL_TEST_SET_ITEMS) * 100}
            className="h-2 bg-slate-900"
          />
        </div>

        {/* Advanced Simulation Configuration Drawer */}
        {showConfig && (
          <div className="mt-4 pt-4 border-t border-white/10 space-y-4 bg-slate-900/50 p-4 rounded-xl">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
                <Gauge className="w-4 h-4" />
                <span>Zaawansowana Konfiguracja Prędkości & Strumienia Odlewów:</span>
              </span>
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="text-slate-400">Przepustowość taśmy:</span>
                <span className="text-cyan-300 font-bold bg-cyan-950/60 border border-cyan-800/40 px-2 py-0.5 rounded">
                  ~{throughputPerMin} odlewów / min ({throughputPerSec} / sek &bull; {(intervalMs / 1000).toFixed(2)}s)
                </span>
              </div>
            </div>

            {/* 1. DOWOLNY CZAS NAPŁYWU: Direct Input, Steppers, Slider & Presets */}
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-white/5 space-y-3">
              <div className="flex flex-col lg:flex-row justify-between lg:items-center gap-3 text-xs">
                <div>
                  <span className="font-semibold text-white flex items-center gap-1.5">
                    <Clock className="w-4 h-4 text-cyan-400" />
                    <span>Czas napływu odlewów (Wybierz dowolną kadencję w ms):</span>
                  </span>
                  <p className="text-[11px] text-slate-400">
                    Wpisz dowolną liczbę milisekund (np. 15, 45, 120, 1500) lub klikaj przyciski.
                  </p>
                </div>

                {/* Direct Number Input + Stepper Controls */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => updateCadence(intervalMs - 500)}
                    className="h-7 px-2 text-[11px] text-slate-300"
                    title="-500 ms"
                  >
                    -500
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => updateCadence(intervalMs - 100)}
                    className="h-7 px-2 text-[11px] text-slate-300"
                    title="-100 ms"
                  >
                    -100
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => updateCadence(intervalMs - 25)}
                    className="h-7 px-2 text-[11px] text-slate-300"
                    title="-25 ms"
                  >
                    -25
                  </Button>

                  {/* Direct Arbitrary Number Input */}
                  <div className="flex items-center gap-1 bg-slate-900 border-2 border-cyan-500/50 rounded-lg px-2 py-0.5 shadow-inner">
                    <input
                      type="number"
                      min={5}
                      max={60000}
                      step={1}
                      value={customIntervalInput}
                      onChange={handleInputChange}
                      className="w-20 bg-transparent text-cyan-300 font-mono font-extrabold text-sm text-right focus:outline-none"
                    />
                    <span className="text-xs text-slate-400 font-mono font-bold">ms</span>
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => updateCadence(intervalMs + 25)}
                    className="h-7 px-2 text-[11px] text-slate-300"
                    title="+25 ms"
                  >
                    +25
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => updateCadence(intervalMs + 100)}
                    className="h-7 px-2 text-[11px] text-slate-300"
                    title="+100 ms"
                  >
                    +100
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => updateCadence(intervalMs + 500)}
                    className="h-7 px-2 text-[11px] text-slate-300"
                    title="+500 ms"
                  >
                    +500
                  </Button>
                </div>
              </div>

              {/* Slider (10ms - dynamic max) */}
              <div className="pt-1">
                <Slider
                  min={10}
                  max={Math.max(3000, Math.ceil(intervalMs / 500) * 500)}
                  step={5}
                  value={[intervalMs]}
                  onValueChange={(vals) => updateCadence(vals[0])}
                  className="py-1"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                  <span>10 ms (100 FPS)</span>
                  <span>100 ms (~600/min)</span>
                  <span>500 ms (~120/min)</span>
                  <span>1000 ms (~60/min)</span>
                  <span>{Math.max(3000, Math.ceil(intervalMs / 500) * 500)} ms</span>
                </div>
              </div>

              {/* Speed Preset Buttons */}
              <div className="flex flex-wrap gap-1.5 pt-1">
                <Button
                  variant={intervalMs === 25 ? 'secondary' : 'outline'}
                  size="sm"
                  onClick={() => updateCadence(25)}
                  className={`text-[11px] py-1 px-2.5 h-7 ${
                    intervalMs === 25 ? 'border-red-500 text-red-400 font-bold' : ''
                  }`}
                >
                  <Zap className="w-3 h-3 mr-1 text-red-400" />
                  Ultra 25 ms (~2400/min)
                </Button>

                <Button
                  variant={intervalMs === 50 ? 'secondary' : 'outline'}
                  size="sm"
                  onClick={() => updateCadence(50)}
                  className={`text-[11px] py-1 px-2.5 h-7 ${
                    intervalMs === 50 ? 'border-amber-500 text-amber-400 font-bold' : ''
                  }`}
                >
                  <Flame className="w-3 h-3 mr-1 text-amber-400" />
                  Turbo 50 ms (~1200/min)
                </Button>

                <Button
                  variant={intervalMs === 100 ? 'secondary' : 'outline'}
                  size="sm"
                  onClick={() => updateCadence(100)}
                  className={`text-[11px] py-1 px-2.5 h-7 ${
                    intervalMs === 100 ? 'border-cyan-500 text-cyan-400 font-bold' : ''
                  }`}
                >
                  <Zap className="w-3 h-3 mr-1 text-cyan-400" />
                  Szybka 100 ms (~600/min)
                </Button>

                <Button
                  variant={intervalMs === 250 ? 'secondary' : 'outline'}
                  size="sm"
                  onClick={() => updateCadence(250)}
                  className={`text-[11px] py-1 px-2.5 h-7 ${
                    intervalMs === 250 ? 'border-cyan-500 text-cyan-400 font-bold' : ''
                  }`}
                >
                  250 ms (~240/min)
                </Button>

                <Button
                  variant={intervalMs === 500 ? 'secondary' : 'outline'}
                  size="sm"
                  onClick={() => updateCadence(500)}
                  className={`text-[11px] py-1 px-2.5 h-7 ${
                    intervalMs === 500 ? 'border-cyan-500 text-cyan-400 font-bold' : ''
                  }`}
                >
                  500 ms (~120/min)
                </Button>

                <Button
                  variant={intervalMs === 1000 ? 'secondary' : 'outline'}
                  size="sm"
                  onClick={() => updateCadence(1000)}
                  className={`text-[11px] py-1 px-2.5 h-7 ${
                    intervalMs === 1000 ? 'border-cyan-500 text-cyan-400 font-bold' : ''
                  }`}
                >
                  Standard 1.0 s (~60/min)
                </Button>

                <Button
                  variant={intervalMs === 2500 ? 'secondary' : 'outline'}
                  size="sm"
                  onClick={() => updateCadence(2500)}
                  className={`text-[11px] py-1 px-2.5 h-7 ${
                    intervalMs === 2500 ? 'border-cyan-500 text-cyan-400 font-bold' : ''
                  }`}
                >
                  Prezentacja 2.5 s (~24/min)
                </Button>
              </div>
            </div>

            {/* 2. KONFIGURACJA MOCY PROCESORA CPU & SYMULACJA SŁABSZEJ MASZYNY (EDGE AI) */}
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-white/5 space-y-3">
              <div className="flex flex-col lg:flex-row justify-between lg:items-center gap-2">
                <div>
                  <span className="font-semibold text-white flex items-center gap-1.5 text-xs">
                    <Cpu className={`w-4 h-4 ${cpuConfig?.is_throttled ? 'text-amber-400' : 'text-cyan-400'}`} />
                    <span>Konfiguracja Mocy Obliczeniowej CPU (Symulacja Słabszego Sprzętu / Edge IoT):</span>
                  </span>
                  <p className="text-[11px] text-slate-400">
                    Wybierz profil maszyny lub obniż taktowanie/wątki procesora, aby przetestować wydajność modeli AI na słabszym sprzęcie przemysłowym (np. Raspberry Pi, mikrokontroler).
                  </p>
                </div>

                <div className="flex items-center gap-2 text-xs font-mono">
                  <span className="text-slate-400">Aktywny procesor:</span>
                  <span className={`px-2 py-0.5 rounded font-bold border ${
                    cpuConfig?.is_throttled
                      ? 'border-amber-500/50 bg-amber-500/10 text-amber-300'
                      : 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300'
                  }`}>
                    {cpuConfig?.threads ?? 1}w @ {cpuConfig?.throttle_percent ?? 100}% mocy
                    {cpuConfig?.is_throttled && (
                      <span className="ml-1 opacity-80">
                        (~{(100 / Math.max(5, cpuConfig?.throttle_percent ?? 100)).toFixed(1)}x wolniej)
                      </span>
                    )}
                  </span>
                </div>
              </div>

              {/* Hardware Profiles Quick Buttons */}
              <div className="space-y-1.5">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                  Gotowe Profile Sprzętowe Maszyn Fabrycznych:
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                  <Button
                    variant={cpuConfig?.profile === 'unthrottled' ? 'secondary' : 'outline'}
                    size="sm"
                    disabled={cpuLoading}
                    onClick={() => handleSelectCpuProfile('unthrottled')}
                    className={`h-auto py-2 px-2.5 flex flex-col items-start text-left ${
                      cpuConfig?.profile === 'unthrottled'
                        ? 'border-emerald-500 text-emerald-300 bg-emerald-500/10'
                        : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    <span className="text-xs font-bold flex items-center gap-1">
                      Serwer / PC
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                      100% &bull; {cpuConfig?.total_system_cores ?? 16} wątków
                    </span>
                  </Button>

                  <Button
                    variant={cpuConfig?.profile === 'industrial_pc' ? 'secondary' : 'outline'}
                    size="sm"
                    disabled={cpuLoading}
                    onClick={() => handleSelectCpuProfile('industrial_pc')}
                    className={`h-auto py-2 px-2.5 flex flex-col items-start text-left ${
                      cpuConfig?.profile === 'industrial_pc'
                        ? 'border-cyan-500 text-cyan-300 bg-cyan-500/10'
                        : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    <span className="text-xs font-bold flex items-center gap-1">
                      Przemysłowy IPC
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                      60% &bull; 4 wątki (i3/Atom)
                    </span>
                  </Button>

                  <Button
                    variant={cpuConfig?.profile === 'jetson_nano' ? 'secondary' : 'outline'}
                    size="sm"
                    disabled={cpuLoading}
                    onClick={() => handleSelectCpuProfile('jetson_nano')}
                    className={`h-auto py-2 px-2.5 flex flex-col items-start text-left ${
                      cpuConfig?.profile === 'jetson_nano'
                        ? 'border-amber-500 text-amber-300 bg-amber-500/10'
                        : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    <span className="text-xs font-bold flex items-center gap-1">
                      Jetson / Edge AI
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                      35% &bull; 2 wątki (~3x)
                    </span>
                  </Button>

                  <Button
                    variant={cpuConfig?.profile === 'raspberry_pi_4' ? 'secondary' : 'outline'}
                    size="sm"
                    disabled={cpuLoading}
                    onClick={() => handleSelectCpuProfile('raspberry_pi_4')}
                    className={`h-auto py-2 px-2.5 flex flex-col items-start text-left ${
                      cpuConfig?.profile === 'raspberry_pi_4'
                        ? 'border-amber-500 text-amber-300 bg-amber-500/10'
                        : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    <span className="text-xs font-bold flex items-center gap-1">
                      Raspberry Pi 4
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                      18% &bull; 2 wątki (~5.5x)
                    </span>
                  </Button>

                  <Button
                    variant={cpuConfig?.profile === 'ultra_low_power' ? 'secondary' : 'outline'}
                    size="sm"
                    disabled={cpuLoading}
                    onClick={() => handleSelectCpuProfile('ultra_low_power')}
                    className={`h-auto py-2 px-2.5 flex flex-col items-start text-left ${
                      cpuConfig?.profile === 'ultra_low_power'
                        ? 'border-red-500 text-red-300 bg-red-500/10'
                        : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    <span className="text-xs font-bold flex items-center gap-1">
                      Ultra Low-Power
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                      8% &bull; 1 wątek (~12x)
                    </span>
                  </Button>

                  <Button
                    variant={cpuConfig?.profile === 'custom' ? 'secondary' : 'outline'}
                    size="sm"
                    disabled={cpuLoading}
                    onClick={() => handleSelectCpuProfile('custom')}
                    className={`h-auto py-2 px-2.5 flex flex-col items-start text-left ${
                      cpuConfig?.profile === 'custom'
                        ? 'border-purple-500 text-purple-300 bg-purple-500/10'
                        : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    <span className="text-xs font-bold flex items-center gap-1">
                      Własna (Custom)
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                      Ręczne suwaki
                    </span>
                  </Button>
                </div>
              </div>

              {/* Sliders for Manual / Fine-tuning Control */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1 border-t border-white/5">
                {/* 1. CPU Cores / Threads Slider */}
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-300 font-semibold flex items-center gap-1">
                      <span>Liczba wątków PyTorch CPU:</span>
                    </span>
                    <span className="text-cyan-400 font-mono font-bold">
                      {cpuThreads} / {cpuConfig?.total_system_cores ?? 16} wątków
                    </span>
                  </div>
                  <Slider
                    min={1}
                    max={cpuConfig?.total_system_cores ?? 16}
                    step={1}
                    value={[cpuThreads]}
                    onValueChange={(vals) => handleUpdateCustomCpu(vals[0], cpuThrottlePercent)}
                    className="py-1"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>1 wątek (IoT)</span>
                    <span>2 wątki (Edge)</span>
                    <span>4 wątki (IPC)</span>
                    <span>{cpuConfig?.total_system_cores ?? 16} wątków (Max)</span>
                  </div>
                </div>

                {/* 2. CPU Power / Clock Throttling Slider */}
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-300 font-semibold flex items-center gap-1">
                      <span>Moc obliczeniowa CPU (Throttling taktowania):</span>
                    </span>
                    <span className="text-amber-400 font-mono font-bold">
                      {cpuThrottlePercent}% mocy (~{(100 / Math.max(5, cpuThrottlePercent)).toFixed(1)}x spowolnienie)
                    </span>
                  </div>
                  <Slider
                    min={5}
                    max={100}
                    step={5}
                    value={[cpuThrottlePercent]}
                    onValueChange={(vals) => handleUpdateCustomCpu(cpuThreads, vals[0])}
                    className="py-1"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>5% (Zator)</span>
                    <span>18% (Raspberry Pi)</span>
                    <span>50% (Pół mocy)</span>
                    <span>100% (Pełna moc)</span>
                  </div>
                </div>
              </div>

              {/* Informational description box */}
              <div className="p-2.5 rounded-lg bg-slate-900/60 border border-white/5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-[11px] text-slate-400">
                <span className="flex items-center gap-1.5">
                  <span className="text-cyan-400">ℹ️</span>
                  <span>
                    Aktualny profil: <b className="text-white">{cpuConfig?.profile_name}</b>.
                    {cpuConfig?.is_throttled
                      ? ' Symulacja osłabionej maszyny spowalnia inferencję, co pozwala zaobserwować wąskie gardło taśmy produkcyjnej.'
                      : ' Procesor pracuje z pełną wydajnością sprzętową komputera hosta.'}
                  </span>
                </span>
                {cpuConfig?.is_throttled && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleSelectCpuProfile('unthrottled')}
                    className="h-6 text-[10px] text-emerald-400 hover:text-emerald-300 whitespace-nowrap"
                  >
                    Przywróć 100% mocy
                  </Button>
                )}
              </div>
            </div>

            {/* 3. Zbiór 1200 elementów, kolejność i zasady zatrzymania */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Dataset Limit and Order Controls */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-white/5 space-y-2">
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Zarządzanie Zbiorem Testowym (1200 sztuk):
                </label>
                <div className="flex flex-col gap-2 text-xs text-slate-300">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={stopAt1200}
                      onChange={(e) => setStopAt1200(e.target.checked)}
                      className="rounded border-white/20 text-cyan-500 focus:ring-cyan-500"
                    />
                    <span className="font-semibold text-cyan-300">
                      Zatrzymaj po zbadaniu 1200 odlewów (Pełna partia testowa)
                    </span>
                  </label>

                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={shuffleOrder}
                      onChange={(e) => {
                        setShuffleOrder(e.target.checked);
                        if (e.target.checked) handleReshuffle();
                      }}
                      className="rounded border-white/20 text-cyan-500 focus:ring-cyan-500"
                    />
                    <span>Losowe tasowanie 1200 odlewów</span>
                  </label>

                  <div className="pt-1 flex items-center justify-between text-[11px] text-slate-400">
                    <span>Pozostało do zbadania:</span>
                    <b className="text-white font-mono">
                      {Math.max(0, TOTAL_TEST_SET_ITEMS - currentIndex)} sztuk
                    </b>
                  </div>
                </div>
              </div>

              {/* Safety Rules */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-white/5 space-y-2">
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Zasady Bezpieczeństwa Jakości:
                </label>
                <div className="flex flex-col gap-2">
                  <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
                    <input
                      type="checkbox"
                      checked={stopOnFN}
                      onChange={(e) => setStopOnFN(e.target.checked)}
                      className="rounded border-white/20 text-cyan-500 focus:ring-cyan-500"
                    />
                    <span>Zatrzymaj natychmiast przy False Negative (FN)</span>
                  </label>

                  <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
                    <input
                      type="checkbox"
                      checked={stopOnFP}
                      onChange={(e) => setStopOnFP(e.target.checked)}
                      className="rounded border-white/20 text-cyan-500 focus:ring-cyan-500"
                    />
                    <span>Zatrzymaj przy False Positive (FP) - fałszywy alarm</span>
                  </label>
                </div>
              </div>
            </div>

            {/* 3. Decision Threshold (tau) Calibration & RAM Optimization Telemetry */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Threshold */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-white/5 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-300 uppercase tracking-wider">
                    Próg Klasyfikacji (&tau;):
                  </span>
                  <span className="font-mono text-cyan-400 font-bold bg-slate-900 px-2 py-0.5 rounded border border-cyan-500/30">
                    {threshold.toFixed(3)}
                  </span>
                </div>
                <Slider
                  min={0.01}
                  max={0.9}
                  step={0.005}
                  value={[threshold]}
                  onValueChange={(vals) => setThreshold(vals[0])}
                  className="py-1"
                />
                <div className="flex gap-2 pt-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setThreshold(OPTIMAL_THRESHOLDS[modelName] || 0.197)}
                    className="text-[11px] py-1 px-2 border-cyan-500/30 text-cyan-400 h-7 flex-1"
                  >
                    Optymalny &tau;* ({(OPTIMAL_THRESHOLDS[modelName] || 0.197).toFixed(3)})
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setThreshold(0.05)}
                    className="text-[11px] py-1 px-2 text-red-400 hover:text-red-300 h-7"
                    title="Zero tolerancji dla wad"
                  >
                    Rygorystyczny 0.05
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setThreshold(0.5)}
                    className="text-[11px] py-1 px-2 text-slate-400 h-7"
                  >
                    0.50
                  </Button>
                </div>
              </div>

              {/* RAM Telemetry & On-Demand Status */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-white/5 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Optymalizacja RAM (On-Demand):</span>
                  </span>
                  <Badge variant="cyan" className="text-[10px] py-0 px-1.5 font-mono">
                    1 MODEL W RAM
                  </Badge>
                </div>
                <div className="text-xs text-slate-400 space-y-1 font-mono">
                  <div className="flex justify-between">
                    <span>Zużycie pamięci (RSS):</span>
                    <b className="text-white">
                      {memoryStatus?.rss_memory_mb ? `${memoryStatus.rss_memory_mb} MB` : 'On-Demand'}
                    </b>
                  </div>
                  <div className="flex justify-between">
                    <span>Załadowany model w RAM:</span>
                    <b className="text-cyan-300">
                      {memoryStatus?.active_model || 'Żaden (Odładowany z RAM)'}
                    </b>
                  </div>
                </div>
                <div className="pt-1 flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={refreshMemory}
                    className="h-7 text-[11px] text-slate-300 flex-1 gap-1"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Odśwież RAM
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={unloadingMemory || isRunning}
                    onClick={handleUnloadMemory}
                    className="h-7 text-[11px] text-red-400 hover:text-red-300 gap-1"
                  >
                    <Trash2 className="w-3 h-3" />
                    Zwolnij Pamięć
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Real-time Conveyor Bottleneck Alert */}
      {currentStep && currentStep.latency_ms > intervalMs && (
        <div className="p-4 bg-amber-950/70 border-2 border-amber-500/70 rounded-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-xl animate-in fade-in">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-400 flex-shrink-0">
              <AlertOctagon className="w-5 h-5 animate-bounce" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-amber-300 flex items-center gap-2">
                <span>WĄSKIE GARDŁO LINII PRODUKCYJNEJ (CPU BOTTLENECK)</span>
                {currentStep.cpu_config?.is_throttled && (
                  <span className="text-[10px] px-2 py-0.5 rounded font-mono bg-amber-500/20 border border-amber-500/30 text-amber-200">
                    {currentStep.cpu_config.profile_name}
                  </span>
                )}
              </h2>
              <p className="text-xs text-amber-100 mt-0.5">
                Czas inferencji na osłabionym CPU (<b>{currentStep.latency_ms.toFixed(1)} ms</b>) przekracza zadany czas napływu taśmy (<b>{intervalMs} ms</b>). 
                W rzeczywistej fabryce taśma musiałaby zwolnić lub detale uległyby spiętrzeniu!
              </p>
            </div>
          </div>
          {modelName === 'resnet18' && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleModelChange('mobilenet_v2')}
              className="text-xs font-bold border-amber-400 text-amber-300 hover:bg-amber-500/20 whitespace-nowrap gap-1.5 shadow-md"
            >
              <Zap className="w-3.5 h-3.5 text-amber-300" />
              <span>Przełącz na lekki MobileNetV2</span>
            </Button>
          )}
        </div>
      )}

      {/* Main Simulation View (Grid: Camera HUD + Comparison Panel) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Camera HUD & Animated Conveyor (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="camera-hud overflow-hidden rounded-xl border border-white/10 shadow-2xl">
            <div className="camera-header-bar flex justify-between items-center bg-slate-900/90 px-3 py-2 border-b border-white/10">
              <div className="flex items-center gap-2">
                <span className={isRunning ? 'red-pulse' : 'live-pulse'} />
                <span className="font-bold text-xs text-white">AOI-CAM-01 &bull; STRUMIEŃ TESTOWY</span>
              </div>
              <div className="font-mono text-[11px] text-cyan-400">
                {currentStep ? `ID: #${currentStep.item_id} (${Math.min(currentIndex, TOTAL_TEST_SET_ITEMS)}/1200)` : 'OCZEKIWANIE'}
              </div>
            </div>

            <div className="relative min-h-[300px] flex items-center justify-center p-4 bg-slate-950 overflow-hidden">
              {currentStep ? (
                <div className="relative w-64 h-64">
                  {/* SMOOTH ZERO-LATENCY IN-MEMORY RENDERING (Base64 data URI avoids secondary network roundtrip) */}
                  <img
                    key={currentStep.item_id}
                    src={currentStep.image_base64 || getSimulationImageUrl(currentStep.item_id)}
                    alt={currentStep.filename}
                    className="w-full h-full object-contain rounded-lg shadow-2xl"
                  />
                  <div className="camera-crosshair" />
                </div>
              ) : (
                <div className="text-slate-500 text-center space-y-2 py-8">
                  <Radio className="w-12 h-12 mx-auto opacity-30 animate-pulse" />
                  <div className="text-sm font-semibold text-slate-300">Taśma produkcyjna zatrzymana</div>
                  <div className="text-xs text-slate-500">
                    Kliknij &quot;Uruchom Symulację&quot; lub &quot;Następny Odlew&quot;
                  </div>
                </div>
              )}

              {/* Status banner in camera corner */}
              {isRunning && (
                <div className="absolute top-2 left-2 bg-slate-950/80 border border-white/10 rounded px-2 py-1 text-[10px] font-mono text-cyan-400">
                  KADENCJA: {intervalMs}ms
                </div>
              )}
            </div>

            {/* Industrial Conveyor Belt Graphic Animation */}
            <div
              className="p-2.5 bg-slate-950 border-t border-white/10"
              style={{ ['--conveyor-speed' as any]: `${Math.max(0.08, intervalMs / 1000)}s` }}
            >
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-1 px-1">
                <span className="flex items-center gap-1">
                  <Settings2 className={`w-3.5 h-3.5 ${isRunning ? 'animate-spin' : ''}`} />
                  <span>ROLKI NAPĘDOWE</span>
                </span>
                <span className="text-cyan-400 font-bold">
                  {isRunning ? `TAŚMA W RUCHU (${intervalMs}ms)` : 'POSTÓJ TAŚMY'}
                </span>
                <span className="flex items-center gap-1">
                  <span>SORTOWNIK</span>
                  <Settings2 className={`w-3.5 h-3.5 ${isRunning ? 'animate-spin' : ''}`} />
                </span>
              </div>

              {/* Conveyor Track Stripe */}
              <div
                className={`h-5 rounded-md border border-cyan-500/30 conveyor-track ${
                  isRunning ? 'running' : ''
                } flex items-center justify-between px-2`}
              >
                <ArrowRight className="w-3 h-3 text-cyan-400 opacity-80" />
                <ArrowRight className="w-3 h-3 text-cyan-400 opacity-80" />
                <ArrowRight className="w-3 h-3 text-cyan-400 opacity-80" />
              </div>
            </div>

            {/* File info footer with CPU latency telemetry */}
            {currentStep && (
              <div className="p-2.5 bg-slate-900 border-t border-white/10 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-1 text-xs text-slate-300 font-mono">
                <span className="truncate max-w-[220px]">{currentStep.filename}</span>
                <div className="flex items-center gap-2">
                  {currentStep.raw_latency_ms && currentStep.raw_latency_ms !== currentStep.latency_ms && (
                    <span className="text-[10px] text-slate-500 line-through" title="Czas bazowy nieograniczonego procesora hosta">
                      {currentStep.raw_latency_ms.toFixed(1)} ms
                    </span>
                  )}
                  <span className={`font-semibold flex items-center gap-1 ${
                    currentStep.latency_ms > intervalMs ? 'text-amber-400 font-bold' : 'text-cyan-400'
                  }`}>
                    <Clock className="w-3 h-3" />
                    {currentStep.latency_ms.toFixed(1)} ms
                    {currentStep.cpu_config?.is_throttled && (
                      <span className="text-[9px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                        {currentStep.cpu_config.profile_name?.split('(')[0]?.trim()}
                      </span>
                    )}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Comparison & Verdict Panel: "Czy model dobrze rozpoznał?" (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {currentStep ? (
            <Card
              className={`p-6 border-2 transition-all duration-300 ${
                currentStep.is_correct
                  ? 'bg-slate-900/90 border-emerald-500/50 shadow-2xl shadow-emerald-500/10'
                  : currentStep.classification_type === 'FN'
                  ? 'bg-red-950/40 border-red-500 shadow-2xl shadow-red-500/30'
                  : 'bg-amber-950/30 border-amber-500 shadow-2xl shadow-amber-500/20'
              }`}
            >
              {/* Top Match Evaluation Header */}
              <div className="flex items-center justify-between pb-4 border-b border-white/10 mb-5">
                <div className="flex items-center gap-3">
                  {currentStep.is_correct ? (
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
                      <CheckCircle2 className="w-6 h-6" />
                    </div>
                  ) : (
                    <div className="w-10 h-10 rounded-xl bg-red-500/20 border border-red-500/40 flex items-center justify-center text-red-400">
                      <AlertOctagon className="w-6 h-6 animate-pulse" />
                    </div>
                  )}
                  <div>
                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Weryfikacja Trafności Modelu AI:
                    </div>
                    <div
                      className={`text-lg md:text-xl font-extrabold ${
                        currentStep.is_correct
                          ? 'text-emerald-300'
                          : currentStep.classification_type === 'FN'
                          ? 'text-red-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {currentStep.classification_badge}
                    </div>
                  </div>
                </div>

                <Badge
                  variant={currentStep.is_correct ? 'emerald' : 'crimson'}
                  className="font-bold text-sm px-3 py-1 font-mono"
                >
                  {currentStep.classification_type}
                </Badge>
              </div>

              {/* Side-by-Side: Ground Truth vs Model Decision */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
                {/* 1. Ground Truth */}
                <div className="p-4 rounded-xl bg-slate-950/80 border border-white/10 space-y-1.5">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                    Stan Faktyczny (Ground Truth):
                  </span>
                  <div
                    className={`text-lg font-extrabold flex items-center gap-2 ${
                      currentStep.ground_truth_label === 1 ? 'text-red-400' : 'text-emerald-400'
                    }`}
                  >
                    {currentStep.ground_truth_label === 1 ? (
                      <>
                        <AlertOctagon className="w-5 h-5" />
                        <span>WADLIWY (WADA ODLEWU)</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="w-5 h-5" />
                        <span>POPRAWNY (BEZ WAD)</span>
                      </>
                    )}
                  </div>
                  <p className="text-xs text-slate-400">
                    Etykieta wzorcowa ze zbioru testowego Technocast.
                  </p>
                </div>

                {/* 2. Model Decision */}
                <div className="p-4 rounded-xl bg-slate-950/80 border border-white/10 space-y-1.5">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                    Decyzja Modelu AI (Sieć {currentStep.model_name}):
                  </span>
                  <div
                    className={`text-lg font-extrabold flex items-center gap-2 ${
                      currentStep.is_defective ? 'text-red-400' : 'text-emerald-400'
                    }`}
                  >
                    {currentStep.is_defective ? (
                      <>
                        <AlertOctagon className="w-5 h-5" />
                        <span>ODRZUCONY (WADA)</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="w-5 h-5" />
                        <span>ZATWIERDZONY (OK)</span>
                      </>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 truncate">
                    {currentStep.recommendation}
                  </p>
                </div>
              </div>

              {/* Confidence Progress Bar */}
              <div className="p-4 rounded-xl bg-slate-950/60 border border-white/5 space-y-2">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-emerald-400">
                    Pewność OK: {(currentStep.prob_ok * 100).toFixed(1)}%
                  </span>
                  <span className="text-slate-400 font-mono">
                    Próg decyzyjny &tau; = {currentStep.threshold.toFixed(3)}
                  </span>
                  <span className="text-red-400">
                    Pewność WADA: {(currentStep.prob_defect * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="h-3 w-full bg-slate-800 rounded-full flex overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 transition-all duration-200"
                    style={{ width: `${(currentStep.prob_ok * 100).toFixed(1)}%` }}
                  />
                  <div
                    className="h-full bg-red-500 transition-all duration-200"
                    style={{ width: `${(currentStep.prob_defect * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>

              {/* Engineering Impact Callout */}
              <div className="mt-4 p-3 rounded-lg bg-slate-950/90 border border-white/5 text-xs text-slate-300">
                {currentStep.classification_type === 'TP' && (
                  <span className="text-emerald-300">
                    <b>Prawidłowa eliminacja wady:</b> Wadliwy wirnik został natychmiast
                    skierowany do odrzutu przed kolejnym etapem montażu.
                  </span>
                )}
                {currentStep.classification_type === 'TN' && (
                  <span className="text-emerald-300">
                    <b>Prawidłowe dopuszczenie:</b> Pełnowartościowy wirnik skierowano do
                    montażu pompy bez zbędnego opóźnienia linii.
                  </span>
                )}
                {currentStep.classification_type === 'FN' && (
                  <span className="text-red-300">
                    <b>KRYTYCZNE ZAGROŻENIE:</b> Wadliwy wirnik przeszedł kontrolę jako sprawny!
                    Zalecenie: obniż próg decyzyjny &tau; do trybu rygorystycznego.
                  </span>
                )}
                {currentStep.classification_type === 'FP' && (
                  <span className="text-amber-300">
                    <b>Fałszywy alarm:</b> Sprawny odlew został odrzucony przez drobny cień lub refleks.
                    Wymagana powtórna weryfikacja.
                  </span>
                )}
              </div>
            </Card>
          ) : (
            <Card className="p-12 text-center flex flex-col items-center justify-center gap-3 text-slate-400">
              <Sparkles className="w-10 h-10 text-cyan-400 opacity-50" />
              <div className="text-base font-bold text-slate-200">
                Oczekiwanie na uruchomienie symulacji taśmy
              </div>
              <p className="text-xs max-w-md text-slate-400 leading-relaxed">
                Wybierz model i dowolną prędkość napływu w panelu u góry, a następnie kliknij &quot;Uruchom Symulację&quot;.
                System przejdzie po kolei przez wszystkie 1200 odlewów ze zbioru testowego, sprawdzając je bez opóźnień
                i automatycznie zatrzyma się po zbadaniu pełnej partii.
              </p>
            </Card>
          )}
        </div>
      </div>

      {/* Session KPI Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3.5">
        <Card className="p-3.5 bg-slate-950/70">
          <div className="text-[11px] uppercase text-slate-400 font-semibold mb-1">
            Zbadano w Partii
          </div>
          <div className="text-2xl font-extrabold font-mono text-white">
            {stats.total} <span className="text-xs text-slate-500 font-normal">/ 1200</span>
          </div>
        </Card>

        <Card className="p-3.5 bg-slate-950/70">
          <div className="text-[11px] uppercase text-slate-400 font-semibold mb-1">
            Trafność Sesji (Acc)
          </div>
          <div className="text-2xl font-extrabold font-mono text-cyan-400">
            {accuracyPercent}%
          </div>
        </Card>

        <Card className="p-3.5 bg-slate-950/70">
          <div className="text-[11px] uppercase text-slate-400 font-semibold mb-1">
            Wykrywalność Wad (Recall)
          </div>
          <div className="text-2xl font-extrabold font-mono text-emerald-400">
            {recallPercent}%
          </div>
        </Card>

        <Card className="p-3.5 bg-slate-950/70">
          <div className="text-[11px] uppercase text-slate-400 font-semibold mb-1">
            Precyzja Odrzutów
          </div>
          <div className="text-2xl font-extrabold font-mono text-cyan-400">
            {precisionPercent}%
          </div>
        </Card>

        <Card className="p-3.5 bg-slate-950/70 border-red-500/30">
          <div className="text-[11px] uppercase text-slate-400 font-semibold mb-1">
            Przeoczone Wady (FN)
          </div>
          <div
            className={`text-2xl font-extrabold font-mono ${
              stats.fn > 0 ? 'text-red-400 animate-pulse' : 'text-slate-400'
            }`}
          >
            {stats.fn}
          </div>
        </Card>

        <Card className="p-3.5 bg-slate-950/70 border-amber-500/30">
          <div className="text-[11px] uppercase text-slate-400 font-semibold mb-1">
            Fałszywe Alarmy (FP)
          </div>
          <div className="text-2xl font-extrabold font-mono text-amber-400">
            {stats.fp}
          </div>
        </Card>
      </div>

      {/* Real-time Production Feed Table */}
      <Card className="overflow-hidden">
        <CardHeader className="p-4 border-b border-white/10 bg-slate-950/60 flex flex-row items-center justify-between">
          <CardTitle className="text-sm font-bold flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Dziennik Napływających Odlewów (Ostatnie {recentFeed.length} sztuk)</span>
          </CardTitle>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="text-slate-400">
              Średnie opóźnienie CPU: <b className="text-white">{avgLatency} ms</b>
            </span>
            <span className="text-slate-400">
              Obciążenie kadencji: <b className="text-cyan-300">{intervalMs > 0 && stats.total > 0 ? `${Math.round((parseFloat(avgLatency) / intervalMs) * 100)}%` : '0%'}</b>
            </span>
          </div>
        </CardHeader>

        {recentFeed.length > 0 ? (
          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/90 sticky top-0 border-b border-white/10 text-slate-400 uppercase text-[10px] tracking-wider">
                <tr>
                  <th className="py-2.5 px-3">#ID Próbki</th>
                  <th className="py-2.5 px-3">Plik Wirnika</th>
                  <th className="py-2.5 px-3">Stan Faktyczny (Ground Truth)</th>
                  <th className="py-2.5 px-3">Decyzja AI</th>
                  <th className="py-2.5 px-3">Zgodność / Typ</th>
                  <th className="py-2.5 px-3">P(Wada)</th>
                  <th className="py-2.5 px-3">Czas CPU</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {recentFeed.map((row, idx) => {
                  const isTP = row.classification_type === 'TP';
                  const isTN = row.classification_type === 'TN';
                  const isFN = row.classification_type === 'FN';
                  return (
                    <tr
                      key={`${row.item_id}-${idx}`}
                      className={`hover:bg-white/[0.02] transition-colors ${
                        isFN ? 'bg-red-500/10' : ''
                      }`}
                    >
                      <td className="py-2 px-3 font-mono text-slate-400">#{row.item_id}</td>
                      <td className="py-2 px-3 font-semibold text-white truncate max-w-[160px]">
                        {row.filename}
                      </td>
                      <td className="py-2 px-3">
                        <Badge
                          variant={row.ground_truth_label === 1 ? 'crimson' : 'emerald'}
                          className="text-[10px] py-0 px-2 font-mono"
                        >
                          {row.ground_truth_name}
                        </Badge>
                      </td>
                      <td className="py-2 px-3">
                        <Badge
                          variant={row.is_defective ? 'crimson' : 'emerald'}
                          className="text-[10px] py-0 px-2 font-mono"
                        >
                          {row.is_defective ? 'WADLIWY' : 'POPRAWNY'}
                        </Badge>
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`inline-flex items-center gap-1 font-semibold text-[11px] font-mono ${
                            isTP || isTN
                              ? 'text-emerald-400'
                              : isFN
                              ? 'text-red-400 font-bold'
                              : 'text-amber-400'
                          }`}
                        >
                          {row.classification_type}
                        </span>
                      </td>
                      <td className="py-2 px-3 font-mono text-slate-300">
                        {(row.prob_defect * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 font-mono text-cyan-400">
                        {row.latency_ms.toFixed(1)} ms
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-10 text-center text-slate-500 text-xs">
            Brak przetworzonych odlewów w bieżącej sesji.
          </div>
        )}
      </Card>
    </div>
  );
}
