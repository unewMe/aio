'use client';

import React from 'react';
import { AlertOctagon, CheckCircle2 } from 'lucide-react';
import { InspectionResponse } from '@/lib/types';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

interface VerdictCardProps {
  inspection: InspectionResponse;
}

export default function VerdictCard({ inspection }: VerdictCardProps) {
  const isDef = inspection.is_defective;
  const probDefPercent = Number((inspection.prob_defect * 100).toFixed(2));
  const probOkPercent = Number((inspection.prob_ok * 100).toFixed(2));

  return (
    <Card
      className={`p-6 text-center relative overflow-hidden transition-all duration-300 ${
        isDef
          ? 'bg-gradient-to-br from-red-950/40 via-red-900/20 to-slate-950/80 border-red-500/50 shadow-2xl shadow-red-500/20'
          : 'bg-gradient-to-br from-emerald-950/40 via-emerald-900/20 to-slate-950/80 border-emerald-500/50 shadow-2xl shadow-emerald-500/20'
      }`}
    >
      <div className="flex items-center justify-center gap-3 mb-2">
        {isDef ? (
          <AlertOctagon className="w-8 h-8 text-red-500 animate-pulse" />
        ) : (
          <CheckCircle2 className="w-8 h-8 text-emerald-400" />
        )}
        <h2
          className={`text-2xl md:text-3xl font-extrabold tracking-tight ${
            isDef ? 'text-red-400' : 'text-emerald-300'
          }`}
        >
          {isDef ? 'ODLEW WADLIWY (ODRZUCONY)' : 'ODLEW POPRAWNY (ZATWIERDZONY)'}
        </h2>
      </div>

      <p className="text-sm md:text-base text-slate-300 max-w-xl mx-auto mb-4">
        {isDef
          ? `Wykryto niezgodność strukturalną (pęknięcie lub ubytek metalu). Prawdopodobieństwo wady: ${probDefPercent}%`
          : `Struktura wirnika prawidłowa. Odlew spełnia normy metalurgiczne Pilot Technocast.`}
      </p>

      {/* Recommendation Pill */}
      <div className="mb-6">
        <Badge
          className={`text-xs md:text-sm py-1.5 px-5 font-bold uppercase tracking-wider shadow-lg ${
            isDef
              ? 'bg-red-600 text-white shadow-red-600/30 hover:bg-red-500'
              : 'bg-emerald-600 text-white shadow-emerald-600/30 hover:bg-emerald-500'
          }`}
        >
          {inspection.recommendation}
        </Badge>
      </div>

      {/* Dual Confidence Progress Bar */}
      <div className="bg-slate-950/70 border border-white/10 rounded-xl p-4 text-left">
        <div className="flex justify-between text-xs font-semibold mb-2">
          <span className="text-emerald-400">Pewność OK: {probOkPercent}%</span>
          <span className="text-slate-400 font-mono">
            Próg &tau; = {inspection.threshold.toFixed(3)}
          </span>
          <span className="text-red-400">Pewność WADA: {probDefPercent}%</span>
        </div>

        <div className="h-3 w-full bg-slate-800 rounded-full flex overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all duration-500 ease-out"
            style={{ width: `${probOkPercent}%` }}
          />
          <div
            className="h-full bg-red-500 transition-all duration-500 ease-out"
            style={{ width: `${probDefPercent}%` }}
          />
        </div>
      </div>
    </Card>
  );
}
