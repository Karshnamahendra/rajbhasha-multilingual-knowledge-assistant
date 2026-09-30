import React, { useMemo, useState } from 'react';
import { BookOpen, BarChart3, FileText, Search, Trash2, Upload, CheckSquare, MinusSquare, Square, X, Layers } from 'lucide-react';
import { UploadedDocument } from '../types';

type Tab = 'magazines' | 'reports';

interface DocumentSelectorProps {
  documents: UploadedDocument[];
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  onUploadClick: () => void;
  onDelete: (docId: string, e: React.MouseEvent) => void;
  /** Small screens: the sidebar is a drawer; this closes it. */
  onClose?: () => void;
}

const isReport = (d: UploadedDocument) => d.document_type === 'report';

/** "Abhivyakti_Sixeen edition 2024.docx" -> "Abhivyakti Sixeen edition 2024" */
const displayName = (d: UploadedDocument) =>
  (d.document_name || d.fileName || d.id).replace(/\.(pdf|docx?)$/i, '').replace(/_/g, ' ');

const yearOf = (d: UploadedDocument): string | null => {
  if (d.year) return String(d.year);
  const m = (d.fileName || '').match(/(?:^|\D)((?:19|20)\d{2})(?!\d)/);
  return m ? m[1] : null;
};

/**
 * Left sidebar (replaces the old wide document picker): pick which documents the next question searches.
 * Nothing selected = search every document.
 */
export const DocumentSelector: React.FC<DocumentSelectorProps> = ({
  documents,
  selectedIds,
  onSelectionChange,
  activeTab,
  onTabChange,
  onUploadClick,
  onDelete,
  onClose,
}) => {
  const [search, setSearch] = useState('');
  const [year, setYear] = useState<string>('all');

  const magazines = documents.filter((d) => !isReport(d));
  const reports = documents.filter(isReport);
  const tabDocs = activeTab === 'magazines' ? magazines : reports;

  const years = useMemo(
    () => Array.from(new Set(tabDocs.map(yearOf).filter((y): y is string => !!y))).sort().reverse(),
    [tabDocs],
  );

  const visible = tabDocs
    .filter((d) => year === 'all' || yearOf(d) === year)
    .filter((d) => !search.trim() || `${displayName(d)} ${d.report_period ?? ''}`.toLowerCase().includes(search.trim().toLowerCase()))
    .sort((a, b) => (yearOf(a) ?? '').localeCompare(yearOf(b) ?? '') || displayName(a).localeCompare(displayName(b)));

  const selected = new Set(selectedIds);
  const selectedInTab = (list: UploadedDocument[]) => list.filter((d) => selected.has(d.id)).length;
  const visibleSelected = visible.filter((d) => selected.has(d.id)).length;
  const allVisibleSelected = visible.length > 0 && visibleSelected === visible.length;

  const toggle = (id: string) =>
    onSelectionChange(selected.has(id) ? selectedIds.filter((x) => x !== id) : [...selectedIds, id]);

  const toggleAllVisible = () => {
    const ids = visible.map((d) => d.id);
    onSelectionChange(
      allVisibleSelected
        ? selectedIds.filter((id) => !ids.includes(id))
        : Array.from(new Set([...selectedIds, ...ids])),
    );
  };

  const selectedDocs = documents.filter((d) => selected.has(d.id));

  const TabButton: React.FC<{ tab: Tab; label: string; icon: React.ReactNode; list: UploadedDocument[] }> = ({
    tab,
    label,
    icon,
    list,
  }) => {
    const active = activeTab === tab;
    const n = selectedInTab(list);
    return (
      <button
        type="button"
        onClick={() => onTabChange(tab)}
        className={`flex-1 inline-flex items-center justify-center gap-1.5 rounded-lg px-2 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
          active ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
        }`}
      >
        {icon}
        {label}
        <span
          className={`rounded-full px-1.5 text-[10px] tabular-nums ${
            active ? 'bg-white/20' : 'bg-slate-800 text-slate-400'
          }`}
        >
          {n > 0 ? `${n}/${list.length}` : list.length}
        </span>
      </button>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Title */}
      <div className="shrink-0 flex items-center justify-between px-4 pt-4 pb-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-100">दस्तावेज़</h2>
          <p className="text-[11px] text-slate-500">जिनमें खोजना है, उन्हें चुनें</p>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="md:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 cursor-pointer"
            aria-label="बंद करें"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="shrink-0 px-3">
        <div className="flex gap-1 rounded-xl bg-slate-950/70 border border-slate-800 p-1">
          <TabButton tab="magazines" label="पत्रिकाएँ" icon={<BookOpen className="w-3.5 h-3.5" />} list={magazines} />
          <TabButton tab="reports" label="रिपोर्ट" icon={<BarChart3 className="w-3.5 h-3.5" />} list={reports} />
        </div>
      </div>

      {/* Search + year */}
      <div className="shrink-0 px-3 pt-3 flex gap-2">
        <label className="flex-1 flex items-center gap-2 rounded-lg bg-slate-950/70 border border-slate-800 focus-within:border-indigo-500/60 px-2.5">
          <Search className="w-3.5 h-3.5 text-slate-500 shrink-0" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="फ़ाइल खोजें"
            className="w-full bg-transparent py-1.5 text-xs text-slate-200 placeholder:text-slate-600 outline-none"
          />
        </label>
        {years.length > 1 && (
          <select
            value={year}
            onChange={(e) => setYear(e.target.value)}
            className="rounded-lg bg-slate-950/70 border border-slate-800 px-2 text-xs text-slate-300 outline-none cursor-pointer"
            aria-label="वर्ष"
          >
            <option value="all">सभी वर्ष</option>
            {years.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Select all in view */}
      {visible.length > 0 && (
        <div className="shrink-0 px-4 pt-3 pb-1 flex items-center justify-between text-[11px]">
          <button
            type="button"
            onClick={toggleAllVisible}
            className="inline-flex items-center gap-1.5 text-slate-400 hover:text-slate-200 cursor-pointer"
          >
            {allVisibleSelected ? (
              <CheckSquare className="w-3.5 h-3.5 text-indigo-400" />
            ) : visibleSelected > 0 ? (
              <MinusSquare className="w-3.5 h-3.5 text-indigo-400" />
            ) : (
              <Square className="w-3.5 h-3.5" />
            )}
            {allVisibleSelected ? 'सभी हटाएँ' : 'सभी चुनें'}
          </button>
          <span className="text-slate-600">{visible.length} फ़ाइलें</span>
        </div>
      )}

      {/* Document list */}
      <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-2">
        {visible.length === 0 ? (
          <div className="px-3 py-8 text-center text-xs text-slate-500">
            {tabDocs.length === 0
              ? activeTab === 'magazines'
                ? 'अभी कोई पत्रिका नहीं है।'
                : 'अभी कोई रिपोर्ट नहीं है।'
              : 'इस खोज से कोई फ़ाइल नहीं मिली।'}
          </div>
        ) : (
          <ul className="space-y-1">
            {visible.map((d) => {
              const on = selected.has(d.id);
              const y = yearOf(d);
              return (
                <li key={d.id}>
                  <div
                    role="checkbox"
                    aria-checked={on}
                    tabIndex={0}
                    onClick={() => toggle(d.id)}
                    onKeyDown={(e) => {
                      if (e.key === ' ' || e.key === 'Enter') {
                        e.preventDefault();
                        toggle(d.id);
                      }
                    }}
                    className={`group flex items-start gap-2.5 rounded-xl px-2.5 py-2 cursor-pointer border transition-colors outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/60 ${
                      on
                        ? 'bg-indigo-500/10 border-indigo-500/40'
                        : 'border-transparent hover:bg-slate-800/50'
                    }`}
                  >
                    <span
                      className={`mt-0.5 w-4 h-4 shrink-0 rounded border flex items-center justify-center ${
                        on ? 'bg-indigo-500 border-indigo-500' : 'border-slate-600'
                      }`}
                    >
                      {on && (
                        <svg viewBox="0 0 12 12" className="w-3 h-3 text-white" aria-hidden="true">
                          <path d="M2.5 6.2l2.3 2.3 4.7-4.9" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      )}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-[13px] leading-snug text-slate-200 break-words" title={d.fileName}>
                        {displayName(d)}
                      </span>
                      <span className="mt-1 flex items-center gap-2 text-[11px] text-slate-500">
                        {isReport(d) ? (
                          <FileText className="w-3 h-3 text-emerald-400" aria-hidden="true" />
                        ) : (
                          <BookOpen className="w-3 h-3 text-indigo-400" aria-hidden="true" />
                        )}
                        {isReport(d) && d.report_period ? (
                          <span className="rounded-md bg-slate-800 px-1.5 py-px font-medium text-slate-300">{d.report_period}</span>
                        ) : y && (
                          <span className="rounded-md bg-slate-800 px-1.5 py-px font-medium text-slate-300">{y}</span>
                        )}
                        {d.totalPages ? <span>{d.totalPages} पृष्ठ</span> : null}
                      </span>
                    </span>
                    <button
                      type="button"
                      onClick={(e) => onDelete(d.id, e)}
                      className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 -m-0.5 rounded-md text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-opacity cursor-pointer"
                      title="हटाएँ"
                      aria-label={`${displayName(d)} हटाएँ`}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Scope + upload */}
      <div className="shrink-0 border-t border-slate-800 p-3 space-y-2.5">
        <div className="flex items-start gap-2 text-[11px]">
          <Layers className="w-3.5 h-3.5 mt-px shrink-0 text-indigo-400" />
          <div className="min-w-0 flex-1">
            <span className="text-slate-500">खोज का दायरा: </span>
            <span className="text-slate-200">
              {selectedDocs.length === 0 ? 'सभी दस्तावेज़' : `${selectedDocs.length} फ़ाइलें चुनी गईं`}
            </span>
          </div>
          {selectedDocs.length > 0 && (
            <button
              type="button"
              onClick={() => onSelectionChange([])}
              className="shrink-0 text-slate-500 hover:text-slate-200 cursor-pointer"
            >
              साफ़ करें
            </button>
          )}
        </div>
        <button
          type="button"
          onClick={onUploadClick}
          className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 px-3 py-2 text-sm font-medium text-white transition-colors cursor-pointer"
        >
          <Upload className="w-4 h-4" />
          दस्तावेज़ अपलोड करें
        </button>
      </div>
    </div>
  );
};