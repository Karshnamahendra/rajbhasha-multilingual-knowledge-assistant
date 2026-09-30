import React from 'react';
import { BookOpen, X, Copy, Check } from 'lucide-react';
import { HighlightText } from './HighlightText';
import { SourceItem } from './chatTypes';

interface SourceInspectorProps {
  chunk: SourceItem | null;
  copiedId: string | null;
  onCopy: (text: string, id: string) => void;
  onClose: () => void;
}

export const SourceInspector: React.FC<SourceInspectorProps> = ({ chunk, copiedId, onCopy, onClose }) => {
  if (!chunk) return null;

  return (
    <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-2xl w-full p-5 md:p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-indigo-400" />
            <div>
              <h3 className="text-sm font-bold text-slate-100">
                Source Chunk Inspector • Page {chunk.page_number}
              </h3>
              <span className="text-[11px] text-slate-400">
                Content Type:{' '}
                <span className="text-indigo-300 font-semibold uppercase">
                  {chunk.content_type || chunk.collection_type || 'content'}
                </span>{' '}
                | Doc: {chunk.document_name || chunk.document_id || 'Current Document'}
                {chunk.year ? ` • Year ${chunk.year}` : ''}
              </span>
            </div>
          </div>
          <button
            onClick={() => onClose()}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-xl transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="bg-slate-950 border border-slate-800/80 rounded-2xl p-4 max-h-96 overflow-y-auto font-mono text-xs md:text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
          <HighlightText
            text={chunk.text || 'No raw chunk text available in payload.'}
            highlight={chunk.highlightText || chunk.text?.slice(0, 100)}
            searchQuery=""
          />
        </div>

        <div className="flex justify-end gap-2 pt-2">
          <button
            onClick={() => onCopy(chunk.text || '', 'modal-chunk')}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-xl flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            {copiedId === 'modal-chunk' ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span>Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy Text</span>
              </>
            )}
          </button>
          <button
            onClick={() => onClose()}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl transition-colors cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};