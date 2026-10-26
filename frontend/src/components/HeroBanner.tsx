'use client';

import React from 'react';
import { ShieldCheck, Target, Zap } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export default function HeroBanner() {
  return (
    <Card className="p-7 md:p-8 mb-7 bg-gradient-to-br from-slate-900/90 via-slate-900/60 to-slate-950/90 border-white/10 shadow-2xl relative overflow-hidden">
      <div className="flex items-center gap-2 mb-3">
        <Badge variant="cyan" className="gap-2 py-1 px-3">
          <span className="live-pulse" />
          PILOT TECHNOCAST &bull; AUTOMATED OPTICAL INSPECTION (AOI)
        </Badge>
      </div>

      <h1 className="text-2xl md:text-3xl lg:text-4xl font-extrabold tracking-tight text-white mb-2 leading-tight">
        System Kontroli Jakości Odlewów Wirników Pomp
      </h1>

      <p className="text-sm md:text-base text-slate-400 max-w-3xl leading-relaxed mb-5">
        Wizyjna inspekcja odlewów wirników pomp zatapialnych (impeller 300&times;300 px) z wykorzystaniem 
        architektur Deep Learning. System gwarantuje eliminację krytycznych wad (Recall &ge; 0.98) 
        przy czasie inferencji &le; 100 ms na procesorze CPU.
      </p>

      {/* Target Engineering Specs */}
      <div className="flex flex-wrap gap-3">
        <Badge variant="emerald" className="gap-1.5 py-1.5 px-3 text-xs">
          <ShieldCheck className="w-4 h-4" />
          <span>Cel Krytyczny: Recall &ge; 0.98 (Zero FN)</span>
        </Badge>

        <Badge variant="cyan" className="gap-1.5 py-1.5 px-3 text-xs">
          <Target className="w-4 h-4" />
          <span>Precision &ge; 0.90 (Max Precyzji)</span>
        </Badge>

        <Badge variant="amber" className="gap-1.5 py-1.5 px-3 text-xs">
          <Zap className="w-4 h-4" />
          <span>Opóźnienie CPU &le; 100 ms (Edge AI)</span>
        </Badge>
      </div>
    </Card>
  );
}
