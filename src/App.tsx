import React from 'react';
import { MainChatbot } from './components/MainChatbot';

export default function App() {
  return (
    // Exactly one screen tall on every device; the chat and the file list scroll inside
    <div className="h-dvh overflow-hidden bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500 selection:text-white">
      <MainChatbot />
    </div>
  );
}