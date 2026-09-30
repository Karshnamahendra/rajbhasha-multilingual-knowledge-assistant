import React from 'react';
import { Sparkles, Database, PanelLeft } from 'lucide-react';
import { OllamaStatusInfo } from './chatTypes';

interface HeaderProps {
  ollamaStatus: OllamaStatusInfo;
  /** Small screens: opens the document drawer. */
  onMenuClick?: () => void;
}

/** Slim app bar: name on the left, service status on the right. */
export const Header: React.FC<HeaderProps> = ({ ollamaStatus, onMenuClick }) => (
  <header className="shrink-0 h-14 flex items-center gap-3 px-3 md:px-5 border-b border-slate-800 bg-slate-950/80 backdrop-blur">
    {onMenuClick && (
      <button
        type="button"
        onClick={onMenuClick}
        className="md:hidden p-2 -ml-1 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 cursor-pointer"
        aria-label="दस्तावेज़ दिखाएँ"
      >
        <PanelLeft className="w-5 h-5" />
      </button>
    )}

    <div className="w-8 h-8 shrink-0 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center">
      <Sparkles className="w-4 h-4 text-indigo-300" />
    </div>
    <div className="min-w-0 leading-tight">
      <h1 className="text-[15px] font-semibold text-slate-100 truncate">राजभाषा ज्ञान सहायक</h1>
      <p className="text-[11px] text-slate-500 truncate">Rajbhasha Knowledge Assistant · पत्रिकाएँ और रिपोर्ट</p>
    </div>

    <div className="ml-auto flex items-center gap-2 text-[11px]">
      <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full border border-slate-800 px-2.5 py-1 text-slate-400">
        <Database className="w-3.5 h-3.5" /> Qdrant
      </span>
      <span
        className="inline-flex items-center gap-1.5 rounded-full border border-slate-800 px-2.5 py-1 text-slate-400"
        title={ollamaStatus.available ? 'भाषा मॉडल चल रहा है' : 'भाषा मॉडल (Ollama) से संपर्क नहीं हो पा रहा'}
      >
        <span
          className={`w-2 h-2 rounded-full ${ollamaStatus.available ? 'bg-emerald-400' : 'bg-rose-500'}`}
          aria-hidden="true"
        />
        {ollamaStatus.available ? ollamaStatus.model : 'Ollama बंद'}
      </span>
    </div>
  </header>
);