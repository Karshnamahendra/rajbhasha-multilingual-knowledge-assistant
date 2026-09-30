import React from 'react';
import { Sparkles, Database } from 'lucide-react';
import { OllamaStatusInfo } from './chatTypes';

interface HeaderProps {
  ollamaStatus: OllamaStatusInfo;
}

export const Header: React.FC<HeaderProps> = ({ ollamaStatus }) => (
  <header className="bg-slate-900/90 border border-slate-800 rounded-3xl p-4 md:p-5 flex flex-col sm:flex-row items-center justify-between gap-4 shadow-xl backdrop-blur-md">
    <div className="flex items-center gap-3 text-center sm:text-left">
      <div className="w-11 h-11 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 shadow-inner">
        <Sparkles className="w-5 h-5" />
      </div>
      <div>
        <h1 className="text-xl md:text-2xl font-extrabold text-white tracking-tight">
          DocuMind • Rajbhasha Multilingual Assistant
        </h1>
        <p className="text-xs text-slate-400">
          Autonomous Document Intelligence • Qdrant Vector DB &amp; Llama 3.2
        </p>
      </div>
    </div>

    <div className="flex items-center gap-2.5">
      <div className="flex items-center gap-2 bg-slate-950 px-3 py-1.5 rounded-full border border-slate-800 text-xs">
        <Database className="w-3.5 h-3.5 text-indigo-400" />
        <span className="text-slate-300 font-medium">Qdrant Active</span>
      </div>

      <div className="flex items-center gap-2 bg-slate-950 px-3.5 py-1.5 rounded-full border border-slate-800 text-xs">
        <span className={`w-2 h-2 rounded-full ${ollamaStatus.available ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
        <span className="text-slate-300 font-medium">
          {ollamaStatus.available ? `Ollama (${ollamaStatus.model})` : 'Ollama Standby'}
        </span>
      </div>
    </div>
  </header>
);