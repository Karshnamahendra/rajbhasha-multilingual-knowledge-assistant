import React, { useEffect, useRef } from 'react';
import { ArrowUp, RefreshCw } from 'lucide-react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  isQuerying: boolean;
  /** What the next question will search, e.g. "सभी दस्तावेज़" or "2 फ़ाइलें" (shown under the box). */
  scopeLabel?: string;
}

/** One-row composer: Enter sends, Shift+Enter adds a new line, grows up to ~6 lines. */
export const ChatInput: React.FC<ChatInputProps> = ({ value, onChange, onSubmit, isQuerying, scopeLabel }) => {
  const formRef = useRef<HTMLFormElement>(null);
  const boxRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the text
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  // Back to the box after an answer arrives
  useEffect(() => {
    if (!isQuerying) boxRef.current?.focus();
  }, [isQuerying]);

  const canSend = !isQuerying && value.trim().length > 0;

  return (
    <div className="shrink-0 px-3 md:px-6 pb-3 md:pb-4 pt-2">
      <form
        ref={formRef}
        onSubmit={onSubmit}
        className="mx-auto max-w-4xl flex items-end gap-2 rounded-2xl border border-slate-700/80 bg-slate-900 focus-within:border-indigo-500/70 px-3 py-2 shadow-lg shadow-black/20 transition-colors"
      >
        <textarea
          ref={boxRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              if (canSend) formRef.current?.requestSubmit();
            }
          }}
          placeholder="अपना सवाल लिखें… (हिंदी, English या Hinglish)"
          className="flex-1 resize-none bg-transparent py-1.5 text-sm leading-6 text-slate-100 placeholder:text-slate-500 outline-none max-h-40"
          disabled={isQuerying}
          aria-label="आपका सवाल"
        />
        <button
          type="submit"
          disabled={!canSend}
          className="shrink-0 w-9 h-9 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white flex items-center justify-center transition-colors cursor-pointer disabled:cursor-not-allowed"
          aria-label="पूछें"
          title="पूछें (Enter)"
        >
          {isQuerying ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ArrowUp className="w-4 h-4" />}
        </button>
      </form>
      <p className="mx-auto max-w-4xl mt-1.5 px-1 text-[10px] text-slate-600">
        {scopeLabel ? <>खोज: {scopeLabel} · </> : null}
        जवाब केवल अपलोड किए गए दस्तावेज़ों से · Enter से भेजें, Shift+Enter से नई पंक्ति
      </p>
    </div>
  );
};