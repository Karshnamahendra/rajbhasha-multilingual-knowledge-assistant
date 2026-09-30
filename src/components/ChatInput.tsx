import React from 'react';
import { Send, RefreshCw } from 'lucide-react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  isQuerying: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ value, onChange, onSubmit, isQuerying }) => (
  <div className="bg-slate-950/70 border-t border-slate-800/80 p-4 md:p-5">
    <form onSubmit={onSubmit} className="space-y-3">
      <div className="relative">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Ask in Hindi, English, or Hinglish (e.g. 2024 पत्रिका के मुख्य लेख, कविताएं, रिपोर्ट के आंकड़े)..."
          className="w-full bg-slate-900/90 border border-slate-800 rounded-2xl px-4 py-3 text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all"
          disabled={isQuerying}
        />
      </div>

      <div className="flex justify-center">
        <button
          type="submit"
          disabled={isQuerying || !value.trim()}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs md:text-sm font-semibold px-8 py-2 rounded-2xl flex items-center gap-2 shadow-lg shadow-indigo-600/25 active:scale-95 transition-all cursor-pointer"
        >
          {isQuerying ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
          <span>Ask Assistant</span>
        </button>
      </div>
    </form>
  </div>
);