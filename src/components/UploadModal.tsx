import React from 'react';
import { UploadCloud, X, BookOpen, FileText, RefreshCw, Check, FileCheck2, AlertCircle } from 'lucide-react';
import { UploadSummary } from './chatTypes';

interface UploadModalProps {
  isOpen: boolean;
  isUploading: boolean;
  category: 'magazine' | 'report';
  progressStep: number;
  statusText: string;
  error: string | null;
  successSummary: UploadSummary | null;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onStartUpload: (category: 'magazine' | 'report') => void;
  onFilesSelected: (files: FileList | null) => void;
  onClose: () => void;
  onDone: () => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  isUploading,
  category,
  progressStep,
  statusText,
  error,
  successSummary,
  fileInputRef,
  onStartUpload,
  onFilesSelected,
  onClose,
  onDone,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-lg w-full p-5 md:p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <UploadCloud className="w-5 h-5 text-indigo-400" />
            <h3 className="text-base font-bold text-slate-100">Upload Document</h3>
          </div>
          <button
            onClick={onClose}
            disabled={isUploading}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-xl transition-colors cursor-pointer disabled:opacity-30"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.doc,.docx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          className="hidden"
          onChange={(e) => onFilesSelected(e.target.files)}
        />

        {!isUploading && !successSummary && (
          <div className="space-y-4">
            <p className="text-xs text-slate-300">
              Which type of document are you uploading?
            </p>

            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => onStartUpload('magazine')}
                className="p-4 rounded-2xl border border-slate-800 bg-slate-950 hover:bg-slate-800/80 hover:border-indigo-500/60 transition-all flex flex-col items-center text-center space-y-2 group cursor-pointer"
              >
                <div className="w-12 h-12 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 group-hover:scale-110 transition-transform">
                  <BookOpen className="w-6 h-6" />
                </div>
                <span className="font-bold text-xs text-slate-100">📖 MAGAZINES</span>
                <span className="text-[10px] text-slate-400">
                  Annual/periodical editions, literary articles, poems, TOC
                </span>
              </button>

              <button
                type="button"
                onClick={() => onStartUpload('report')}
                className="p-4 rounded-2xl border border-slate-800 bg-slate-950 hover:bg-slate-800/80 hover:border-emerald-500/60 transition-all flex flex-col items-center text-center space-y-2 group cursor-pointer"
              >
                <div className="w-12 h-12 rounded-2xl bg-emerald-600/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform">
                  <FileText className="w-6 h-6" />
                </div>
                <span className="font-bold text-xs text-slate-100">📊 REPORTS</span>
                <span className="text-[10px] text-slate-400">
                  Quarterly proformas, official metrics, data tables
                </span>
              </button>
            </div>

            <div className="text-[11px] text-slate-500 text-center">
              Supports PDF (.pdf) and Microsoft Word (.docx, .doc).
            </div>
          </div>
        )}

        {/* Ingestion Progress Display */}
        {isUploading && (
          <div className="space-y-4 py-4">
            <div className="text-center space-y-1">
              <RefreshCw className="w-8 h-8 animate-spin text-indigo-400 mx-auto mb-2" />
              <h4 className="text-sm font-semibold text-slate-200">
                Processing {category === 'magazine' ? 'Magazine' : 'Report'}
              </h4>
              <p className="text-xs text-indigo-300 font-medium">{statusText}</p>
            </div>

            {/* 5-Step Pipeline Progress Indicator */}
            <div className="space-y-2 pt-2">
              {[
                { step: 1, label: 'Uploading document file' },
                { step: 2, label: 'Extracting text and tables' },
                { step: 3, label: 'Detecting structure (Index / Content / Tables)' },
                { step: 4, label: 'Generating multilingual embeddings (384-D)' },
                { step: 5, label: 'Stored in Qdrant Vector DB' }
              ].map((s) => {
                const isDone = progressStep > s.step;
                const isCurrent = progressStep === s.step;
                return (
                  <div
                    key={s.step}
                    className={`flex items-center gap-2.5 text-xs p-2 rounded-xl transition-colors ${
                      isDone
                        ? 'bg-emerald-950/30 text-emerald-300 border border-emerald-500/20'
                        : isCurrent
                        ? 'bg-indigo-950/40 text-indigo-200 border border-indigo-500/30 font-semibold'
                        : 'text-slate-600'
                    }`}
                  >
                    {isDone ? (
                      <Check className="w-4 h-4 text-emerald-400 shrink-0" />
                    ) : isCurrent ? (
                      <RefreshCw className="w-4 h-4 animate-spin text-indigo-400 shrink-0" />
                    ) : (
                      <div className="w-4 h-4 rounded-full border border-slate-700 flex items-center justify-center text-[10px] shrink-0">
                        {s.step}
                      </div>
                    )}
                    <span>{s.label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Upload Success Summary */}
        {successSummary && (
          <div className="space-y-4 py-2">
            <div className="bg-emerald-950/40 border border-emerald-500/30 rounded-2xl p-4 space-y-2">
              <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs">
                <FileCheck2 className="w-4 h-4" />
                <span>Successfully Ingested into Qdrant!</span>
              </div>
              <p className="text-xs text-slate-200 font-medium">{successSummary.name}</p>
              <div className="grid grid-cols-3 gap-2 text-[11px] text-slate-400 pt-2 border-t border-emerald-500/20">
                <div>Category: <span className="text-white font-medium capitalize">{successSummary.type}</span></div>
                <div>Year: <span className="text-emerald-300 font-bold">{successSummary.year || 'Auto'}</span></div>
                <div>Pages: <span className="text-white font-medium">{successSummary.pages}</span></div>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                onClick={onDone}
                className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl transition-colors cursor-pointer"
              >
                Done
              </button>
            </div>
          </div>
        )}

        {/* Upload Error */}
        {error && (
          <div className="bg-rose-950/50 border border-rose-500/40 p-3 rounded-2xl text-xs text-rose-300 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
};