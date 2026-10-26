'use client';

import React, { useState, useEffect } from 'react';
import HeroBanner from '@/components/HeroBanner';
import VerdictCard from '@/components/VerdictCard';
import TelemetryGrid from '@/components/TelemetryGrid';
import {
  fetchModels,
  fetchSamples,
  getSampleImageUrl,
  inspectSampleFile,
  inspectUploadedFile,
} from '@/lib/api';
import { ModelInfo, SampleListResponse, InspectionResponse } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import Link from 'next/link';
import {
  Camera,
  Upload,
  Cpu,
  Sliders,
  Play,
  Sparkles,
  Info,
  AlertTriangle,
  Radio,
  ArrowRight,
} from 'lucide-react';

const OPTIMAL_THRESHOLDS: Record<string, number> = {
  resnet18: 0.197,
  mobilenet_v2: 0.099,
  cnn: 0.158,
};

export default function InspectionStationPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('resnet18');
  const [threshold, setThreshold] = useState<number>(0.197);
  const [samples, setSamples] = useState<SampleListResponse>({ def: [], ok: [] });
  const [sampleCategory, setSampleCategory] = useState<'def' | 'ok'>('def');
  const [selectedSample, setSelectedSample] = useState<string>('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [sourceType, setSourceType] = useState<'sample' | 'upload'>('sample');
  const [opticalFilter, setOpticalFilter] = useState<string>('none');
  
  const [loading, setLoading] = useState<boolean>(false);
  const [inspection, setInspection] = useState<InspectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function init() {
      try {
        const [modelsData, samplesData] = await Promise.all([
          fetchModels().catch(() => []),
          fetchSamples().catch(() => ({ def: [], ok: [] })),
        ]);
        setModels(modelsData);
        setSamples(samplesData);
        if (samplesData.def && samplesData.def.length > 0) {
          setSelectedSample(samplesData.def[0].filename);
        }
      } catch (err: any) {
        console.error('Initialization error:', err);
      }
    }
    init();
  }, []);

  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
    if (OPTIMAL_THRESHOLDS[modelId]) {
      setThreshold(OPTIMAL_THRESHOLDS[modelId]);
    }
  };

  const getPreviewUrl = () => {
    if (sourceType === 'upload' && uploadedFile) {
      return URL.createObjectURL(uploadedFile);
    }
    if (selectedSample) {
      return getSampleImageUrl(sampleCategory, selectedSample);
    }
    return null;
  };

  const handleRunInspection = async () => {
    setLoading(true);
    setError(null);
    try {
      let result: InspectionResponse;
      if (sourceType === 'upload' && uploadedFile) {
        result = await inspectUploadedFile(
          uploadedFile,
          selectedModel,
          threshold,
          'Własne zdjęcie przesłane przez operatora'
        );
      } else if (selectedSample) {
        result = await inspectSampleFile(
          sampleCategory,
          selectedSample,
          selectedModel,
          threshold,
          `Próbka wzorcowa Pilot Technocast (${sampleCategory.toUpperCase()})`
        );
      } else {
        throw new Error('Wybierz próbkę lub wgraj plik ze zdjęciem.');
      }
      setInspection(result);
    } catch (err: any) {
      setError(err.message || 'Wystąpił błąd podczas inspekcji.');
    } finally {
      setLoading(false);
    }
  };

  const getFilterStyle = (): React.CSSProperties => {
    switch (opticalFilter) {
      case 'edge':
        return { filter: 'grayscale(100%) invert(100%) contrast(300%)' };
      case 'contrast':
        return { filter: 'contrast(200%) brightness(95%)' };
      case 'mono':
        return { filter: 'grayscale(100%)' };
      case 'sharp':
        return { filter: 'contrast(130%) saturate(140%)' };
      default:
        return {};
    }
  };

  return (
    <div className="space-y-6">
      <HeroBanner />

      {/* Production Conveyor Simulation Callout Banner */}
      <Card className="p-4 bg-gradient-to-r from-cyan-950/40 via-blue-950/30 to-slate-900 border-cyan-500/30 shadow-lg flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center flex-shrink-0 text-cyan-400">
            <Radio className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="text-sm font-bold text-white flex items-center gap-2">
              <span>Symulacja Ciągłego Napływu Odlewów (Live Conveyor Stream)</span>
              <Badge variant="cyan" className="text-[10px] py-0 px-1.5">NOWOŚĆ</Badge>
            </div>
            <p className="text-xs text-slate-300">
              Sprawdź zachowanie modeli w warunkach produkcyjnych na 1200 losowych odlewach z automatyczną oceną Ground Truth (TP, TN, FP, FN).
            </p>
          </div>
        </div>
        <Link href="/simulation" className="w-full sm:w-auto">
          <Button variant="cyan" size="sm" className="w-full sm:w-auto text-xs font-bold gap-2 whitespace-nowrap">
            <span>Uruchom Symulację Produkcji</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Button>
        </Link>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Control Panel (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Hardware & Inference Config */}
          <Card className="p-5">
            <CardHeader className="p-0 pb-4">
              <CardTitle className="text-base flex items-center gap-2">
                <Cpu className="w-5 h-5 text-cyan-400" />
                <span>1. Konfiguracja Inferencji AI</span>
              </CardTitle>
            </CardHeader>

            <CardContent className="p-0 space-y-4">
              {/* Architecture Selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  Architektura Klasyfikatora:
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => handleModelChange(e.target.value)}
                  className="w-full bg-slate-950 border border-white/10 rounded-lg text-white px-3.5 py-2.5 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="resnet18">ResNet18 (Transfer Learning - ImageNet)</option>
                  <option value="mobilenet_v2">MobileNetV2 (Optimum Edge AI)</option>
                  <option value="cnn">Custom CNN (Autorska Sieć Bazowa)</option>
                </select>
              </div>

              {/* Threshold Slider with shadcn Slider */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Próg Decyzyjny Wady (&tau;):
                  </span>
                  <span className="font-mono font-bold text-cyan-400 text-sm">
                    {threshold.toFixed(3)}
                  </span>
                </div>

                <Slider
                  min={0.01}
                  max={0.9}
                  step={0.005}
                  value={[threshold]}
                  onValueChange={(vals) => setThreshold(vals[0])}
                  className="py-2"
                />

                {/* Mode Guidance Callout */}
                <div
                  className={`mt-2 p-2.5 rounded-lg text-xs border ${
                    threshold <= 0.2
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                      : threshold <= 0.5
                      ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
                      : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                  }`}
                >
                  {threshold <= 0.2 ? (
                    <span>
                      <b>Tryb Zero-Defect (Zalecany):</b> Gwarantuje Recall &ge; 0.98. Żadna
                      wada nie trafi do montażu pompy.
                    </span>
                  ) : threshold <= 0.5 ? (
                    <span>
                      <b>Tryb Zrównoważony:</b> Kompromis między wykrywaniem wad a minimalizacją
                      fałszywych alarmów.
                    </span>
                  ) : (
                    <span>
                      <b>Tryb Wysokiej Precyzji:</b> Ryzyko wystąpienia błędu False Negative
                      (przeoczenie pęknięcia)!
                    </span>
                  )}
                </div>

                {/* Quick set optimum buttons */}
                <div className="flex gap-2 mt-3">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 text-xs border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10"
                    onClick={() => setThreshold(OPTIMAL_THRESHOLDS[selectedModel] || 0.197)}
                  >
                    Optymalny &tau;* ({(OPTIMAL_THRESHOLDS[selectedModel] || 0.197).toFixed(3)})
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-xs text-slate-400 hover:text-white"
                    onClick={() => setThreshold(0.5)}
                  >
                    Standard 0.50
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Source Selection Card */}
          <Card className="p-5">
            <CardHeader className="p-0 pb-4">
              <CardTitle className="text-base flex items-center gap-2">
                <Camera className="w-5 h-5 text-cyan-400" />
                <span>2. Rejestracja Obrazu Wirnika</span>
              </CardTitle>
            </CardHeader>

            <CardContent className="p-0 space-y-4">
              {/* Source Switcher */}
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant={sourceType === 'sample' ? 'default' : 'outline'}
                  size="sm"
                  className={sourceType === 'sample' ? 'bg-cyan-500 hover:bg-cyan-600 text-slate-950 font-bold' : ''}
                  onClick={() => setSourceType('sample')}
                >
                  Katalog Próbek
                </Button>
                <Button
                  variant={sourceType === 'upload' ? 'default' : 'outline'}
                  size="sm"
                  className={sourceType === 'upload' ? 'bg-cyan-500 hover:bg-cyan-600 text-slate-950 font-bold' : ''}
                  onClick={() => setSourceType('upload')}
                >
                  Wgraj Plik
                </Button>
              </div>

              {sourceType === 'sample' ? (
                <div className="space-y-3">
                  {/* Category toggle */}
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant={sampleCategory === 'def' ? 'danger' : 'outline'}
                      size="sm"
                      className="text-xs"
                      onClick={() => {
                        setSampleCategory('def');
                        if (samples.def && samples.def.length > 0) {
                          setSelectedSample(samples.def[0].filename);
                        }
                      }}
                    >
                      Wadliwe (DEF)
                    </Button>
                    <Button
                      variant={sampleCategory === 'ok' ? 'emerald' : 'outline'}
                      size="sm"
                      className="text-xs"
                      onClick={() => {
                        setSampleCategory('ok');
                        if (samples.ok && samples.ok.length > 0) {
                          setSelectedSample(samples.ok[0].filename);
                        }
                      }}
                    >
                      Poprawne (OK)
                    </Button>
                  </div>

                  {/* Thumbnail gallery */}
                  <div className="grid grid-cols-5 gap-2 max-h-40 overflow-y-auto p-1 bg-slate-950/60 rounded-xl border border-white/5">
                    {(samples[sampleCategory] || []).map((s) => (
                      <div
                        key={s.filename}
                        onClick={() => setSelectedSample(s.filename)}
                        className={`cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${
                          selectedSample === s.filename
                            ? 'border-cyan-400 shadow-md shadow-cyan-500/50 scale-95'
                            : 'border-white/10 hover:border-white/30'
                        }`}
                      >
                        <img
                          src={getSampleImageUrl(sampleCategory, s.filename)}
                          alt={s.filename}
                          className="w-full h-12 object-cover"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <label className="flex flex-col items-center justify-center border-2 border-dashed border-cyan-500/30 rounded-xl p-6 cursor-pointer bg-cyan-500/5 hover:bg-cyan-500/10 transition-colors">
                  <Upload className="w-8 h-8 text-cyan-400 mb-2" />
                  <span className="text-sm font-semibold text-white">
                    {uploadedFile ? uploadedFile.name : 'Upuść zdjęcie wirnika tutaj'}
                  </span>
                  <span className="text-xs text-slate-400 mt-1">
                    JPG, JPEG, PNG (format 300x300 px)
                  </span>
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setUploadedFile(e.target.files[0]);
                      }
                    }}
                  />
                </label>
              )}

              {/* Optical filter selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  Optyczny Filtr Wizyjny AOI:
                </label>
                <select
                  value={opticalFilter}
                  onChange={(e) => setOpticalFilter(e.target.value)}
                  className="w-full bg-slate-950 border border-white/10 rounded-lg text-white px-3 py-2 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="none">Widok standardowy (RGB)</option>
                  <option value="edge">Wykrywanie Krawędzi (Pęknięcia i rysy)</option>
                  <option value="contrast">Wysoki Kontrast (Niedolewy i pory)</option>
                  <option value="mono">Monochromatyczny (Skala szarości)</option>
                  <option value="sharp">Wyostrzenie Konturów</option>
                </select>
              </div>

              {/* Action Button */}
              <Button
                variant="cyan"
                size="lg"
                disabled={loading}
                onClick={handleRunInspection}
                className="w-full text-sm font-extrabold tracking-wide"
              >
                {loading ? (
                  <>
                    <span className="live-pulse mr-2" />
                    <span>ANALIZOWANIE SIECIĄ {selectedModel.toUpperCase()}...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white mr-2" />
                    <span>URUCHOM INSPEKCJĘ JAKOŚCI</span>
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Right Live Camera Viewport & Results (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {/* Camera Frame */}
          <div className="camera-hud">
            <div className="camera-header-bar">
              <div className="flex items-center gap-2">
                <span className="red-pulse" />
                <span className="font-bold">AOI-CAM-01 &bull; LIVE FEED</span>
              </div>
              <div className="font-mono">RES: 300&times;300 PX</div>
              <div className="text-cyan-400 font-mono truncate max-w-[200px]">
                {sourceType === 'upload' && uploadedFile
                  ? uploadedFile.name
                  : selectedSample || 'BRAK OBRAZU'}
              </div>
            </div>

            <div className="relative min-h-[360px] flex items-center justify-center p-6 bg-slate-950">
              {getPreviewUrl() ? (
                <div className="relative w-80 h-80">
                  <img
                    src={getPreviewUrl()!}
                    alt="Podgląd wirnika"
                    className="w-full h-full object-contain rounded-lg transition-all"
                    style={getFilterStyle()}
                  />
                  <div className="camera-crosshair" />
                </div>
              ) : (
                <div className="text-slate-500 text-center">
                  <Camera className="w-12 h-12 mx-auto opacity-30 mb-2" />
                  <div className="text-sm">Wybierz próbkę wzorcową lub wgraj zdjęcie wirnika</div>
                </div>
              )}
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-500/15 border border-red-500/30 p-4 rounded-xl text-red-400 text-sm flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Results display */}
          {inspection ? (
            <div className="space-y-4">
              <VerdictCard inspection={inspection} />
              <TelemetryGrid inspection={inspection} />

              <Card className="p-4 bg-slate-950/70">
                <div className="flex items-center gap-2 text-xs font-bold text-white mb-2">
                  <Info className="w-4 h-4 text-cyan-400" />
                  <span>Wyjaśnienie Decyzji Inżynierskiej (Decision Rule):</span>
                </div>
                <ul className="text-xs text-slate-400 space-y-1.5 list-disc pl-5 leading-relaxed">
                  <li>
                    <b>Reguła decyzyjna:</b> Jeśli <code className="font-mono text-cyan-300">P(Wada) &ge; &tau;</code>, odlew zostaje oznaczony jako <b>WADLIWY</b> i odrzucony ze stanowiska montażowego.
                  </li>
                  <li>
                    <b>Wartość wyliczona przez sieć neuronową:</b> <code className="font-mono text-cyan-300">P(Wada) = {inspection.prob_defect.toFixed(5)}</code>, próg <code className="font-mono text-cyan-300">&tau; = {threshold.toFixed(3)}</code>.
                  </li>
                  <li>
                    <b>Rekomendacja zero-defect:</b> Dzięki obniżonemu progowi model osiąga <b>100% wykrywalności wad (0 błędów False Negative)</b> na 600 wadliwych próbkach testowych.
                  </li>
                </ul>
              </Card>
            </div>
          ) : (
            <Card className="p-8 text-center flex flex-col items-center justify-center gap-3 text-slate-400">
              <Sparkles className="w-8 h-8 text-cyan-400 opacity-60" />
              <div className="font-bold text-slate-200">
                Stanowisko kontroli oczekuje na sygnał inspekcji
              </div>
              <p className="text-xs max-w-md text-slate-400 leading-relaxed">
                Wybierz parametry w panelu bocznym i kliknij &quot;Uruchom Inspekcję Jakości&quot;, aby
                uzyskać natychmiastowy werdykt AI oraz telemetrię opóźnienia CPU.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
