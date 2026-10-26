import React from 'react';
import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';

export const metadata: Metadata = {
  title: 'System Kontroli Jakości Odlewów | Pilot Technocast AI',
  description: 'Automatyczna inspekcja optyczna (AOI) wirników pomp zatapialnych z wykorzystaniem sieci Deep Learning (ResNet18, MobileNetV2, Custom CNN).',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pl">
      <body>
        <Navbar />
        <main style={{ maxWidth: '1440px', margin: '0 auto', padding: '24px 28px' }}>
          {children}
        </main>
        <footer
          style={{
            borderTop: '1px solid rgba(255, 255, 255, 0.06)',
            padding: '24px 28px',
            textAlign: 'center',
            color: '#64748b',
            fontSize: '0.82rem',
            marginTop: '40px',
          }}
        >
          <div style={{ marginBottom: '6px' }}>
            System Automatycznej Wizyjnej Kontroli Jakości Odlewów &bull; Pilot Technocast AI
          </div>
          <div>
            Autorzy: <b>Dawid Chudzicki (272715)</b>, <b>Piotr Szczerbiak (294722)</b> &bull; Uczenie Maszynowe
          </div>
        </footer>
      </body>
    </html>
  );
}
