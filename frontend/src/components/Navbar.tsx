'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Camera, Cpu, BarChart3, History, Activity, Radio } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export default function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { label: 'Stacja Inspekcji AOI', href: '/', icon: Camera },
    { label: 'Symulacja Produkcji', href: '/simulation', icon: Radio },
    { label: 'Modele & Benchmark', href: '/models', icon: Cpu },
    { label: 'Ewaluacja & Krzywe PR', href: '/analytics', icon: BarChart3 },
    { label: 'Dziennik Audytu', href: '/history', icon: History },
  ];

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-white/10 bg-slate-950/80 backdrop-blur-md px-6 py-3.5 flex items-center justify-between">
      {/* Brand */}
      <div className="flex items-center gap-3.5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/30">
          <Activity className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-extrabold text-base tracking-tight text-white">
              PILOT TECHNOCAST
            </span>
            <Badge variant="cyan" className="text-[10px] py-0 px-2">
              AOI v1.0
            </Badge>
          </div>
          <div className="text-xs text-slate-400">
            Automatyczna Wizyjna Kontrola Jakości Odlewów Wirników Pomp
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <div className="hidden md:flex items-center gap-1.5">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-white/10 text-cyan-400 border border-cyan-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </div>

      {/* Status indicator */}
      <div className="flex items-center gap-2.5">
        <div className="flex items-center gap-2 bg-slate-900/80 border border-white/10 px-3 py-1.5 rounded-full text-xs text-slate-300">
          <span className="live-pulse" />
          <span className="font-semibold text-white">EDGE AI CPU</span>
          <span className="opacity-30">|</span>
          <span className="text-emerald-400 font-bold">ONLINE</span>
        </div>
      </div>
    </nav>
  );
}
