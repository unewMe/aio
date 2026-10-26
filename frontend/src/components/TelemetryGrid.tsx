'use client';

import React from 'react';
import { Clock, Cpu, Sliders, Database } from 'lucide-react';
import { InspectionResponse } from '@/lib/types';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface TelemetryGridProps {
  inspection: InspectionResponse;
}

export default function TelemetryGrid({ inspection }: TelemetryGridProps) {
  const isTargetMet = inspection.target_met_le_100ms;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5 mt-4">
      {/* Latency */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-1.5 text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <span>Czas Inferencji</span>
          <Clock className="w-4 h-4 text-cyan-400" />
        </div>
        <div className="text-2xl font-extrabold font-mono text-white">
          {inspection.latency_ms.toFixed(1)}{' '}
          <span className="text-sm font-normal text-slate-400">ms</span>
        </div>
        <Badge
          variant={isTargetMet ? 'emerald' : 'crimson'}
          className="mt-2 text-[10px] py-0.5 px-2 font-semibold"
        >
          {isTargetMet ? 'ZGODNE (≤ 100 ms)' : 'POWYŻEJ 100 ms'}
        </Badge>
      </Card>

      {/* Model Name */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-1.5 text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <span>Model AI</span>
          <Cpu className="w-4 h-4 text-indigo-400" />
        </div>
        <div className="text-xl font-extrabold text-white capitalize truncate">
          {inspection.model_name}
        </div>
        <Badge variant="cyan" className="mt-2 text-[10px] py-0.5 px-2 font-semibold">
          Deep Learning
        </Badge>
      </Card>

      {/* Threshold */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-1.5 text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <span>Próg Decyzyjny (τ)</span>
          <Sliders className="w-4 h-4 text-amber-400" />
        </div>
        <div className="text-2xl font-extrabold font-mono text-white">
          {inspection.threshold.toFixed(3)}
        </div>
        <Badge variant="amber" className="mt-2 text-[10px] py-0.5 px-2 font-semibold">
          {inspection.threshold <= 0.20 ? 'Zero-Defect' : 'Zrównoważony'}
        </Badge>
      </Card>

      {/* Audit Log Record */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-1.5 text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <span>Dziennik Audytu</span>
          <Database className="w-4 h-4 text-sky-400" />
        </div>
        <div className="text-2xl font-extrabold font-mono text-white">
          #{inspection.id || 'LIVE'}
        </div>
        <Badge variant="outline" className="mt-2 text-[10px] py-0.5 px-2 font-semibold text-sky-400 border-sky-400/30">
          Zapisano w SQLite
        </Badge>
      </Card>
    </div>
  );
}
