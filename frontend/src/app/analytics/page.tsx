'use client';

import React from 'react';
import { getFigureImageUrl } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { BarChart3, TrendingUp, Grid, Sliders, AlertTriangle, BookOpen } from 'lucide-react';

export default function AnalyticsReportsPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <Badge variant="cyan" className="mb-2.5 gap-1.5">
          <BarChart3 className="w-3.5 h-3.5" />
          RAPORTY BADAWCZE & METRYKI NAUKOWE
        </Badge>
        <h1 className="text-2xl md:text-3xl font-extrabold text-white mb-2">
          Ewaluacja Statystyczna & Wykresy Diagnostyczne
        </h1>
        <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
          Wszystkie wykresy wygenerowano na podstawie 1200 próbnego zbioru testowego z zachowaniem
          pełnego rygoru statystycznego (Repeated Stratified 5x2-fold Cross-Validation wg Thomasa G. Diettericha).
        </p>
      </div>

      {/* Tabs Layout using shadcn Tabs */}
      <Tabs defaultValue="roc" className="w-full">
        <TabsList className="grid grid-cols-2 md:grid-cols-6 w-full h-auto p-1.5 gap-1 bg-slate-950/80">
          <TabsTrigger value="roc" className="gap-2 py-2">
            <TrendingUp className="w-4 h-4" />
            <span>Krzywe ROC & PR</span>
          </TabsTrigger>
          <TabsTrigger value="cm" className="gap-2 py-2">
            <Grid className="w-4 h-4" />
            <span>Macierze Pomyłek</span>
          </TabsTrigger>
          <TabsTrigger value="thresholds" className="gap-2 py-2">
            <Sliders className="w-4 h-4" />
            <span>Strojenie Progów (τ)</span>
          </TabsTrigger>
          <TabsTrigger value="loss" className="gap-2 py-2">
            <BarChart3 className="w-4 h-4" />
            <span>Przebieg Uczenia</span>
          </TabsTrigger>
          <TabsTrigger value="errors" className="gap-2 py-2">
            <AlertTriangle className="w-4 h-4" />
            <span>Analiza Błędów</span>
          </TabsTrigger>
          <TabsTrigger value="methodology" className="gap-2 py-2">
            <BookOpen className="w-4 h-4" />
            <span>Metodologia 5x2</span>
          </TabsTrigger>
        </TabsList>

        {/* ROC Curves */}
        <TabsContent value="roc">
          <Card className="p-6 md:p-8">
            <h3 className="text-lg font-bold text-white mb-1">
              Krzywe ROC (Receiver Operating Characteristic) & Precision-Recall
            </h3>
            <p className="text-xs md:text-sm text-slate-400 mb-5">
              Krzywe ilustrują doskonałą separowalność klas: AUC &gt; 0.9999 dla wszystkich trzech architektur.
            </p>
            <div className="bg-slate-950 rounded-xl p-4 flex items-center justify-center border border-white/5">
              <img
                src={getFigureImageUrl('test_roc_pr_curves.png')}
                alt="Krzywe ROC oraz Precision-Recall"
                className="max-w-full h-auto rounded-lg shadow-lg"
              />
            </div>
          </Card>
        </TabsContent>

        {/* Confusion Matrices */}
        <TabsContent value="cm">
          <Card className="p-6 md:p-8">
            <h3 className="text-lg font-bold text-white mb-1">
              Macierze Pomyłek na Zbiorze Testowym (1200 obrazów)
            </h3>
            <p className="text-xs md:text-sm text-slate-400 mb-5">
              Asymetryczny koszt błędu: kluczowa eliminacja False Negatives (wad przeoczonych) do poziomu 0.
            </p>
            <div className="bg-slate-950 rounded-xl p-4 flex items-center justify-center border border-white/5">
              <img
                src={getFigureImageUrl('test_confusion_matrices.png')}
                alt="Macierze Pomyłek"
                className="max-w-full h-auto rounded-lg shadow-lg"
              />
            </div>
          </Card>
        </TabsContent>

        {/* Threshold Tuning */}
        <TabsContent value="thresholds">
          <Card className="p-6 md:p-8">
            <h3 className="text-lg font-bold text-white mb-1">
              Strojenie Progów Decyzyjnych pod Krytyczny Recall &ge; 0.98
            </h3>
            <p className="text-xs md:text-sm text-slate-400 mb-5">
              Kalibracja optymalnego progu &tau;* gwarantuje zerową liczbę pękniętych wirników
              kierowanych do montażu pomp.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-950 p-3 rounded-xl border border-white/5">
                <Badge variant="cyan" className="mb-2">ResNet18: &tau;* = 0.197</Badge>
                <img
                  src={getFigureImageUrl('threshold_tuning_resnet18.png')}
                  alt="ResNet18 Tuning"
                  className="w-full rounded-lg"
                />
              </div>

              <div className="bg-slate-950 p-3 rounded-xl border border-white/5">
                <Badge variant="cyan" className="mb-2">MobileNetV2: &tau;* = 0.099</Badge>
                <img
                  src={getFigureImageUrl('threshold_tuning_mobilenet_v2.png')}
                  alt="MobileNetV2 Tuning"
                  className="w-full rounded-lg"
                />
              </div>

              <div className="bg-slate-950 p-3 rounded-xl border border-white/5">
                <Badge variant="cyan" className="mb-2">Custom CNN: &tau;* = 0.158</Badge>
                <img
                  src={getFigureImageUrl('threshold_tuning_cnn.png')}
                  alt="Custom CNN Tuning"
                  className="w-full rounded-lg"
                />
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Loss Curves */}
        <TabsContent value="loss">
          <Card className="p-6 md:p-8">
            <h3 className="text-lg font-bold text-white mb-1">
              Przebieg Uczenia Modeli (BCE Loss & Walidacja)
            </h3>
            <p className="text-xs md:text-sm text-slate-400 mb-5">
              Dynamika zbieżności funkcji straty oraz metryk walidacyjnych w kolejnych epokach treningowych.
            </p>
            <div className="bg-slate-950 rounded-xl p-4 flex items-center justify-center border border-white/5">
              <img
                src={getFigureImageUrl('training_curves.png')}
                alt="Krzywe Treningu"
                className="max-w-full h-auto rounded-lg shadow-lg"
              />
            </div>
          </Card>
        </TabsContent>

        {/* Errors Analysis */}
        <TabsContent value="errors">
          <Card className="p-6 md:p-8 space-y-6">
            <h3 className="text-lg font-bold text-white">
              Asymetria Kosztu Błędów w Kontroli Jakości Odlewów
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-red-950/20 border-2 border-red-500/60 rounded-xl p-5 space-y-2">
                <Badge variant="crimson">False Negative (Błąd Krytyczny - Przeoczenie Wady)</Badge>
                <p className="text-xs text-red-300 leading-relaxed">
                  Pęknięty lub niedolany wirnik zostaje zaklasyfikowany jako poprawny i trafia do pompy.
                </p>
                <div className="text-xs text-red-200 pt-2 border-t border-red-500/20">
                  <b>Koszt:</b> Krytycznie wysoki (awaria pompy na obiekcie, zniszczenie silnika).<br />
                  <b>Wynik systemu:</b> <b className="text-white">0 błędów FN (100% wykrywalności).</b>
                </div>
              </div>

              <div className="bg-amber-950/20 border-2 border-amber-500/60 rounded-xl p-5 space-y-2">
                <Badge variant="amber">False Positive (Błąd Umiarkowany - Fałszywy Alarm)</Badge>
                <p className="text-xs text-amber-300 leading-relaxed">
                  Poprawny wirnik zostaje oznaczony jako wadliwy przez cień lub refleks świetlny.
                </p>
                <div className="text-xs text-amber-200 pt-2 border-t border-amber-500/20">
                  <b>Koszt:</b> Niski (wymaga ponownej inspekcji wzrokowej przez operatora).<br />
                  <b>Wynik systemu:</b> Zminimalizowany (Precision &ge; 98.2% - 99.5%).
                </div>
              </div>
            </div>

            <div className="pt-2">
              <h4 className="text-sm font-bold text-white mb-3">
                Wizualizacja Egzemplarzy Błędnie Zaklasyfikowanych (False Positives)
              </h4>
              <div className="bg-slate-950 rounded-xl p-4 flex items-center justify-center border border-white/5">
                <img
                  src={getFigureImageUrl('error_fp_resnet18.png')}
                  alt="False Positives ResNet18"
                  className="max-w-full h-auto rounded-lg shadow-lg"
                />
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Methodology */}
        <TabsContent value="methodology">
          <Card className="p-6 md:p-8 space-y-5">
            <h3 className="text-lg font-bold text-white">
              Procedura Badawcza: Repeated Stratified 5x2-fold Cross-Validation
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              Standardowa walidacja K-fold cierpi na wysoką korelację między foldami testowymi,
              co drastycznie zawyża błąd I rodzaju w testach istotności hipotez statystycznych.
              W celu rzetelnego porównania architektur Custom CNN, ResNet18 oraz MobileNetV2
              zastosowano procedurę <b>5&times;2-fold Cross-Validation</b> zaproponowaną przez
              prof. Thomasa G. Diettericha (1998):
            </p>

            <div className="bg-slate-950 border border-white/10 rounded-xl p-4 font-mono text-xs md:text-sm text-cyan-400">
              Statystyka testowa Diettericha: t = p1_1 / sqrt( 1/5 * sum(s_i^2) ) ~ t(df=5, alpha=0.05)
            </div>

            <ul className="text-xs md:text-sm text-slate-400 space-y-2 list-disc pl-5 leading-relaxed">
              <li>
                <b>10 niezależnych trenowań i ewaluacji:</b> Zbiór 6 000 obrazów dzielony jest 5-krotnie
                na dwa równe podzbiory 50%/50% ze stratyfikacją klas.
              </li>
              <li>
                <b>Stabilność estymatora wariancji:</b> Wykorzystanie różnic w skuteczności z obu foldów
                każdego powtórzenia do wyznaczenia nieobciążonej wariancji.
              </li>
              <li>
                <b>Korekta Nadeau & Bengio:</b> Uwzględnienie nakładania się prób treningowych w estymacji
                istotności statystycznej.
              </li>
            </ul>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
