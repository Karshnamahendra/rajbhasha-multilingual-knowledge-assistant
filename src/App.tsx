import React from 'react';
import { MainChatbot } from './components/MainChatbot';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      <main className="flex-1 w-full">
        <MainChatbot />
      </main>
      
      <footer className="py-3 text-center text-[11px] text-slate-500 border-t border-slate-900">
        Rajbhasha Vibhag Knowledge Assistant • Grounded PDF RAG Pipeline
      </footer>
    </div>
  );
}
