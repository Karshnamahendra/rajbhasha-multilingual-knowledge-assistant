import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpen,
  FileText,
  UploadCloud,
  Trash2,
  Calendar,
  Search,
  Check,
  X,
  Layers,
} from 'lucide-react';
import { UploadedDocument } from '../types';

type CategoryTab = 'magazines' | 'reports';

interface DocumentSelectorProps {
  documents: UploadedDocument[];
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
  activeTab: CategoryTab;
  onTabChange: (tab: CategoryTab) => void;
  onUploadClick: () => void;
  onDelete: (docId: string, e: React.MouseEvent) => void;
}

const docTypeOf = (d: UploadedDocument) => d.document_type || 'magazine';

const docLabel = (d: UploadedDocument) => d.document_name || d.fileName || d.id;

function TabCheckbox({ checked, indeterminate, onChange, disabled }: {
  checked: boolean;
  indeterminate: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = indeterminate;
  }, [indeterminate]);
  return (
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      onChange={onChange}
      disabled={disabled}
      className="w-4 h-4 accent-indigo-500 cursor-pointer disabled:cursor-not-allowed"
    />
  );
}

export const DocumentSelector: React.FC<DocumentSelectorProps> = ({
  documents,
  selectedIds,
  onSelectionChange,
  activeTab,
  onTabChange,
  onUploadClick,
  onDelete,
}) => {
  const [search, setSearch] = useState('');
  const [yearFilter, setYearFilter] = useState<string>('all');

  const selected = useMemo(() => new Set(selectedIds), [selectedIds]);
  const docById = useMemo(() => new Map(documents.map((d) => [d.id, d])), [documents]);

  const magazines = documents.filter((d) => docTypeOf(d) === 'magazine');
  const reports = documents.filter((d) => docTypeOf(d) === 'report');
  const tabDocs = activeTab === 'magazines' ? magazines : reports;

  const years = useMemo(
    () =>
      [...new Set(tabDocs.map((d) => d.year).filter((y): y is number => !!y))].sort((a, b) => b - a),
    [tabDocs]
  );

  // Reset the year filter if it no longer exists in the new tab
  useEffect(() => {
    if (yearFilter !== 'all' && !years.includes(Number(yearFilter))) setYearFilter('all');
  }, [years, yearFilter]);

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    return tabDocs.filter(
      (d) =>
        (yearFilter === 'all' || String(d.year) === yearFilter) &&
        (!q || `${docLabel(d)} ${d.fileName} ${d.report_period ?? ''}`.toLowerCase().includes(q))
    );
  }, [tabDocs, search, yearFilter]);

  const setMany = (ids: string[], on: boolean) => {
    const next = new Set(selected);
    ids.forEach((id) => (on ? next.add(id) : next.delete(id)));
    onSelectionChange([...next]);
  };

  const visibleIds = visible.map((d) => d.id);
  const visibleSelectedCount = visibleIds.filter((id) => selected.has(id)).length;
  const allVisibleSelected = visibleIds.length > 0 && visibleSelectedCount === visibleIds.length;

  const selectedMagCount = selectedIds.filter((id) => docById.get(id) && docTypeOf(docById.get(id)!) === 'magazine').length;
  const selectedRepCount = selectedIds.filter((id) => docById.get(id) && docTypeOf(docById.get(id)!) === 'report').length;

  const tabButton = (tab: CategoryTab, icon: React.ReactNode, label: string, total: number, picked: number) => (
    <button
      onClick={() => onTabChange(tab)}
      className={`flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold transition-all cursor-pointer ${
        activeTab === tab
          ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 ring-1 ring-indigo-400/40'
          : 'bg-slate-950 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-slate-800'
      }`}
    >
      {icon}
      <span>{label}</span>
      <span className="bg-indigo-950/80 text-indigo-300 px-1.5 py-0.5 rounded-full text-[10px] ml-0.5">
        {picked > 0 ? `${picked}/${total}` : total}
      </span>
    </button>
  );

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-4 md:p-5 space-y-4 shadow-lg">
      {/* Tabs + Upload */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          {tabButton('magazines', <BookOpen className="w-4 h-4" />, '📖 MAGAZINES', magazines.length, selectedMagCount)}
          {tabButton('reports', <FileText className="w-4 h-4" />, '📊 REPORTS', reports.length, selectedRepCount)}
        </div>

        <button
          onClick={onUploadClick}
          className="flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs px-5 py-2.5 rounded-2xl transition-all shadow-md shadow-indigo-600/30 active:scale-95 cursor-pointer"
        >
          <UploadCloud className="w-4 h-4" />
          <span>Upload Document</span>
        </button>
      </div>

      {/* Toolbar: select-all, search, year */}
      {tabDocs.length > 0 && (
        <div className="flex flex-col md:flex-row md:items-center gap-2.5">
          <label className="flex items-center gap-2 text-xs text-slate-300 bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 cursor-pointer select-none shrink-0">
            <TabCheckbox
              checked={allVisibleSelected}
              indeterminate={visibleSelectedCount > 0 && !allVisibleSelected}
              onChange={() => setMany(visibleIds, !allVisibleSelected)}
              disabled={!visibleIds.length}
            />
            <span>{allVisibleSelected ? 'सभी हटाएँ' : 'सभी चुनें'} ({visibleIds.length})</span>
          </label>

          <div className="relative flex-1">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="फ़ाइल खोजें / Search files…"
              className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-8 pr-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
          </div>

          {years.length > 0 && (
            <select
              value={yearFilter}
              onChange={(e) => setYearFilter(e.target.value)}
              className="bg-slate-950 border border-slate-800 text-slate-200 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="all">सभी वर्ष / All years</option>
              {years.map((y) => (
                <option key={y} value={String(y)}>
                  {y}
                </option>
              ))}
            </select>
          )}
        </div>
      )}

      {/* Document cards */}
      <div className="space-y-2">
        {tabDocs.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-xs">
            {activeTab === 'magazines' ? (
              <BookOpen className="w-8 h-8 mx-auto text-slate-600 mb-2 opacity-50" />
            ) : (
              <FileText className="w-8 h-8 mx-auto text-slate-600 mb-2 opacity-50" />
            )}
            <p>No {activeTab} uploaded yet.</p>
            <p className="text-[11px] text-slate-600 mt-1">
              Click "Upload Document" and select {activeTab === 'magazines' ? 'MAGAZINES' : 'REPORTS'}.
            </p>
          </div>
        ) : visible.length === 0 ? (
          <p className="text-center py-6 text-slate-500 text-xs">कोई फ़ाइल नहीं मिली / No matching files</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-[340px] overflow-y-auto pr-1">
            {visible.map((doc) => {
              const isSelected = selected.has(doc.id);
              const isReport = docTypeOf(doc) === 'report';
              return (
                <div
                  key={doc.id}
                  role="checkbox"
                  aria-checked={isSelected}
                  tabIndex={0}
                  onClick={() => setMany([doc.id], !isSelected)}
                  onKeyDown={(e) => {
                    if (e.key === ' ' || e.key === 'Enter') {
                      e.preventDefault();
                      setMany([doc.id], !isSelected);
                    }
                  }}
                  className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between space-y-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                    isSelected
                      ? 'bg-indigo-950/40 border-indigo-500/70 shadow-md shadow-indigo-950/40 ring-1 ring-indigo-500/40'
                      : 'bg-slate-950/70 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/60'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2 overflow-hidden">
                      <span
                        className={`w-4 h-4 rounded-md border flex items-center justify-center shrink-0 transition-colors ${
                          isSelected ? 'bg-indigo-500 border-indigo-400' : 'border-slate-600 bg-slate-900'
                        }`}
                      >
                        {isSelected && <Check className="w-3 h-3 text-white" strokeWidth={3} />}
                      </span>
                      {isReport ? (
                        <FileText className="w-4 h-4 text-emerald-400 shrink-0" />
                      ) : (
                        <BookOpen className="w-4 h-4 text-indigo-400 shrink-0" />
                      )}
                      <span className="font-semibold text-xs text-slate-200 truncate" title={doc.fileName}>
                        {docLabel(doc)}
                      </span>
                    </div>
                    <button
                      onClick={(e) => onDelete(doc.id, e)}
                      title="Remove Document"
                      className="text-slate-500 hover:text-rose-400 p-1 rounded-lg hover:bg-rose-950/40 transition-colors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/60">
                    <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                      <Calendar className="w-3 h-3" />
                      {isReport
                        ? doc.report_period || (doc.year ? `Year ${doc.year}` : 'Official Report')
                        : doc.year
                        ? `Year ${doc.year}`
                        : 'Auto-detected'}
                    </span>
                    <span>{doc.totalPages || 1} Pages</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Selection summary (replaces the old single-scope dropdown) */}
      <div className="pt-3 border-t border-slate-800/60 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2 text-slate-400">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-medium">Query Scope:</span>
            <span className="text-slate-200">
              {selectedIds.length === 0
                ? 'सभी दस्तावेज़ / All documents'
                : `${selectedIds.length} ${selectedIds.length === 1 ? 'फ़ाइल चुनी गई' : 'फ़ाइलें चुनी गईं'} / selected`}
            </span>
          </div>
          {selectedIds.length > 0 && (
            <button
              onClick={() => onSelectionChange([])}
              className="text-[11px] text-slate-400 hover:text-rose-300 px-2 py-1 rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
            >
              सब साफ़ करें / Clear
            </button>
          )}
        </div>

        {selectedIds.length > 0 && (
          <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
            {selectedIds.map((id) => {
              const d = docById.get(id);
              const isReport = d && docTypeOf(d) === 'report';
              return (
                <span
                  key={id}
                  className={`inline-flex items-center gap-1 max-w-[260px] pl-2.5 pr-1 py-0.5 rounded-full text-[11px] border ${
                    isReport
                      ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-200'
                      : 'bg-indigo-950/50 border-indigo-500/30 text-indigo-200'
                  }`}
                >
                  <span className="truncate">{d ? docLabel(d) : id}</span>
                  <button
                    onClick={() => setMany([id], false)}
                    className="p-0.5 rounded-full hover:bg-slate-800/80 cursor-pointer"
                    aria-label="Remove from selection"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};