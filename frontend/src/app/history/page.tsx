'use client';

import React, { useState, useEffect } from 'react';
import {
  fetchInspectionHistory,
  fetchInspectionStats,
  clearInspectionHistory,
} from '@/lib/api';
import { InspectionHistoryItem, InspectionStats } from '@/lib/types';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  History,
  Trash2,
  RefreshCw,
  AlertOctagon,
  CheckCircle2,
  Database,
} from 'lucide-react';

export default function HistoryAuditPage() {
  const [history, setHistory] = useState<InspectionHistoryItem[]>([]);
  const [stats, setStats] = useState<InspectionStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const [hData, sData] = await Promise.all([
        fetchInspectionHistory(0, 100).catch(() => []),
        fetchInspectionStats().catch(() => null),
      ]);
      setHistory(hData);
      setStats(sData);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleClear = async () => {
    if (!window.confirm('Czy na pewno chcesz usunąć całą historię audytu z bazy SQLite?')) {
      return;
    }
    setClearing(true);
    try {
      await clearInspectionHistory();
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <Badge variant="cyan" className="mb-2.5 gap-1.5">
            <Database className="w-3.5 h-3.5" />
            DZIENNIK AUDYTU & REJESTR PRODUKCYJNY
          </Badge>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white mb-1">
            Historia Inspekcji Wizyjnych (SQLAlchemy / SQLite)
          </h1>
          <p className="text-sm text-slate-400">
            Zapis telemetrii, werdyktów klasyfikacji i zastosowanych progów decyzyjnych dla każdego wirnika.
          </p>
        </div>

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={loadData}
            disabled={loading}
            className="gap-2"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Odśwież</span>
          </Button>

          <Button
            variant="destructive"
            size="sm"
            onClick={handleClear}
            disabled={clearing || history.length === 0}
            className="gap-2"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Wyczyść Historię</span>
          </Button>
        </div>
      </div>

      {/* Aggregate Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3.5">
          <Card className="p-4">
            <div className="text-xs uppercase text-slate-400 mb-1 font-semibold">
              Łącznie Zbadano
            </div>
            <div className="text-2xl md:text-3xl font-extrabold font-mono text-white">
              {stats.total_inspections}
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-xs uppercase text-slate-400 mb-1 font-semibold">
              Zatwierdzone (OK)
            </div>
            <div className="text-2xl md:text-3xl font-extrabold font-mono text-emerald-400">
              {stats.total_approved}
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-xs uppercase text-slate-400 mb-1 font-semibold">
              Odrzucone (Wada)
            </div>
            <div className="text-2xl md:text-3xl font-extrabold font-mono text-red-400">
              {stats.total_defects}
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-xs uppercase text-slate-400 mb-1 font-semibold">
              Wskaźnik Wadliwości
            </div>
            <div className="text-2xl md:text-3xl font-extrabold font-mono text-amber-400">
              {stats.defect_rate_percent}%
            </div>
          </Card>

          <Card className="p-4 col-span-2 lg:col-span-1">
            <div className="text-xs uppercase text-slate-400 mb-1 font-semibold">
              Średnie Opóźnienie
            </div>
            <div className="text-2xl md:text-3xl font-extrabold font-mono text-cyan-400">
              {stats.average_latency_ms.toFixed(1)} ms
            </div>
          </Card>
        </div>
      )}

      {/* History Table */}
      <Card className="overflow-hidden">
        {history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs md:text-sm">
              <thead className="bg-slate-950/80 border-b border-white/10 text-slate-400 uppercase text-[11px] tracking-wider">
                <tr>
                  <th className="py-3 px-4">#ID</th>
                  <th className="py-3 px-4">Data / Czas</th>
                  <th className="py-3 px-4">Plik Wirnika</th>
                  <th className="py-3 px-4">Model</th>
                  <th className="py-3 px-4">Próg &tau;</th>
                  <th className="py-3 px-4">P(Wada)</th>
                  <th className="py-3 px-4">Werdykt AI</th>
                  <th className="py-3 px-4">Opóźnienie</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {history.map((row) => {
                  const isDef = row.is_defective;
                  return (
                    <tr key={row.id} className="hover:bg-white/[0.02] transition-colors">
                      <td className="py-3 px-4 font-mono text-slate-500">#{row.id}</td>
                      <td className="py-3 px-4 text-slate-300">
                        {new Date(row.created_at).toLocaleString('pl-PL')}
                      </td>
                      <td className="py-3 px-4 font-semibold text-white">{row.filename}</td>
                      <td className="py-3 px-4 capitalize text-slate-300">{row.model_name}</td>
                      <td className="py-3 px-4 font-mono text-cyan-400">
                        {row.threshold.toFixed(3)}
                      </td>
                      <td className="py-3 px-4 font-mono text-white">
                        {(row.prob_defect * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4">
                        <Badge
                          variant={isDef ? 'crimson' : 'emerald'}
                          className="gap-1 text-[11px]"
                        >
                          {isDef ? (
                            <AlertOctagon className="w-3 h-3" />
                          ) : (
                            <CheckCircle2 className="w-3 h-3" />
                          )}
                          {isDef ? 'WADLIWY' : 'POPRAWNY'}
                        </Badge>
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-400">
                        {row.latency_ms.toFixed(1)} ms
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-16 text-center text-slate-500 space-y-2">
            <History className="w-10 h-10 mx-auto opacity-30 mb-2" />
            <div className="text-sm font-semibold text-slate-400">
              Brak zarejestrowanych inspekcji w bazie danych.
            </div>
            <div className="text-xs">
              Przejdź do zakładki &quot;Stacja Inspekcji AOI&quot; i przeprowadź pierwszą analizę odlewu.
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
