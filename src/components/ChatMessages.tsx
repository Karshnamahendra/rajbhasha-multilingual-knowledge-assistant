import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpen,
  FileText,
  ExternalLink,
  Copy,
  Check,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  Table2,
  Sparkles,
  Layers,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import {
  ChatMessage,
  SourceItem,
  RawChartData,
  ComparisonData,
  ComparisonRow,
  BackendComparisonRow,
} from './chatTypes';

interface ChatMessagesProps {
  messages: ChatMessage[];
  isQuerying: boolean;
  copiedId: string | null;
  onCopy: (text: string, id: string) => void;
  onSourceClick: (source: SourceItem) => void;
  /** Ask a suggested question from the welcome screen. */
  onSuggestion?: (question: string) => void;
  /** What the next question will search, shown on the welcome screen. */
  scopeLabel?: string;
}

// ---------------------------------------------------------------------------
// chart_data normalisation
// ---------------------------------------------------------------------------

/** Turn "1,234.5", "45%", " 12 " etc. into a number; anything else becomes null. */
const toNumber = (v: unknown): number | null => {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string') {
    const cleaned = v.replace(/[,%\s₹]/g, '');
    if (cleaned === '') return null;
    const n = Number(cleaned);
    return Number.isFinite(n) ? n : null;
  }
  return null;
};

const NAME_KEYS = ['metric', 'name', 'label', 'parameter', 'category', 'item', 'title', 'मद', 'विवरण'];
const SKIP_KEYS = ['change', 'change_pct', 'changepct', 'badlav', 'diff', 'difference', 'percent_change', 'pct_change', 'unit'];

const buildRow = (metric: string, values: (number | null)[]): ComparisonRow => {
  const first = values[0];
  const last = values[values.length - 1];
  const change = first !== null && last !== null && values.length > 1 ? last - first : null;
  const changePct = change !== null && first !== null && first !== 0 ? (change / Math.abs(first)) * 100 : null;
  return { metric, values, change, changePct };
};

/** Rows like [{ metric: "X", "2024": 10, "2025": 12 }]. Period columns are any other numeric keys. */
const fromRows = (rows: Record<string, unknown>[]): ComparisonData | null => {
  if (!rows.length) return null;
  const nameKey =
    Object.keys(rows[0]).find((k) => NAME_KEYS.includes(k.toLowerCase())) ??
    Object.keys(rows[0]).find((k) => typeof rows[0][k] === 'string' && toNumber(rows[0][k]) === null);
  if (!nameKey) return null;

  // Collect period columns in first-seen order; sort year-like keys ascending.
  // A column counts if it looks like a year, or holds a number in at least one row
  // (so a year with missing values still gets its own column showing "—").
  const periodSet: string[] = [];
  rows.forEach((r) =>
    Object.keys(r).forEach((k) => {
      if (k === nameKey || SKIP_KEYS.includes(k.toLowerCase()) || periodSet.includes(k)) return;
      const looksLikeYear = /^\d{4}(-\d{2,4})?$/.test(k);
      const hasNumber = rows.some((row) => toNumber(row[k]) !== null);
      if (looksLikeYear || hasNumber) periodSet.push(k);
    }),
  );
  const allYears = periodSet.every((p) => /^\d{4}(-\d{2,4})?$/.test(p));
  const periods = allYears ? [...periodSet].sort() : periodSet;
  if (!periods.length) return null;

  return {
    periods,
    rows: rows.map((r) => buildRow(String(r[nameKey] ?? ''), periods.map((p) => toNumber(r[p])))),
  };
};

export const normalizeChartData = (raw: RawChartData | null | undefined): ComparisonData | null => {
  if (raw === null || raw === undefined) return null;

  let data: unknown = raw;
  if (typeof data === 'string') {
    try {
      data = JSON.parse(data);
    } catch {
      return null;
    }
  }

  if (Array.isArray(data)) return fromRows(data as Record<string, unknown>[]);
  if (typeof data !== 'object' || data === null) return null;

  const obj = data as Exclude<RawChartData, string | Record<string, unknown>[]>;
  const meta = {
    title: typeof obj.title === 'string' ? obj.title : undefined,
    unit: typeof obj.unit === 'string' ? obj.unit : undefined,
  };

  // 1. Backend format: labels = metrics, one `series` entry per period
  if (Array.isArray(obj.labels) && Array.isArray(obj.series) && obj.series.length) {
    const periods = obj.series.map((s, i) => String(s.name ?? `Series ${i + 1}`));
    const rows = obj.labels.map((label, li) =>
      buildRow(
        String(label),
        obj.series!.map((s) => toNumber(s.values?.[li])),
      ),
    );
    return { ...meta, periods, rows };
  }

  // 1b. Chart.js style: labels = metrics, one dataset per period
  if (Array.isArray(obj.labels) && Array.isArray(obj.datasets) && obj.datasets.length) {
    const periods = obj.datasets.map((d, i) => String(d.label ?? `Series ${i + 1}`));
    const rows = obj.labels.map((label, li) =>
      buildRow(
        String(label),
        obj.datasets!.map((d) => toNumber(d.data?.[li])),
      ),
    );
    return { ...meta, periods, rows };
  }

  // 3. { years: [...], metrics: [{ name, values: [...] }] }
  if (Array.isArray(obj.years) && Array.isArray(obj.metrics)) {
    const periods = obj.years.map(String);
    const rows = obj.metrics.map((m) =>
      buildRow(String(m.name ?? m.metric ?? m.label ?? ''), periods.map((_, i) => toNumber(m.values?.[i]))),
    );
    return { ...meta, periods, rows };
  }

  // 2. Rows nested under a key
  const nested = obj.rows ?? obj.data ?? obj.items;
  if (Array.isArray(nested)) {
    const result = fromRows(nested);
    return result ? { ...meta, ...result } : null;
  }

  return null;
};

/**
 * The backend's `comparison` rows (report_metrics / magazine_metrics). Preferred over
 * `chart_data` for the table: it keeps the region, percent flag and backend-computed change.
 */
export const fromBackendComparison = (
  rows: BackendComparisonRow[] | null | undefined,
  periods: string[] | null | undefined,
  unit?: string,
): ComparisonData | null => {
  if (!Array.isArray(rows) || !rows.length) return null;
  const cols =
    Array.isArray(periods) && periods.length
      ? periods.map(String)
      : Array.from(new Set(rows.flatMap((r) => Object.keys(r.values || {})))).sort();
  if (!cols.length) return null;

  const out: ComparisonRow[] = rows.map((r) => {
    const values = cols.map((p) => toNumber(r.values?.[p]));
    const base = buildRow([r.region ? `${r.region} क्षेत्र` : '', r.metric || ''].filter(Boolean).join(' – '), values);
    return {
      ...base,
      isPercent: !!r.is_percent,
      // Use the backend's numbers when present; they were computed from the source tables
      change: r.change !== undefined ? toNumber(r.change) : base.change,
      changePct: r.is_percent ? null : r.change_pct !== undefined ? toNumber(r.change_pct) : base.changePct,
    };
  });
  return { periods: cols, rows: out, unit };
};

// ---------------------------------------------------------------------------
// Fallback: read the comparison straight from the answer text
// Handles lines like "तकनीकी लेख: 7 → 15 (+8)" or "- **Kavita**: ८ -> १५".
// Used only when the backend sent no chart_data.
// ---------------------------------------------------------------------------

/** Devanagari digits (०-९) → ASCII so they can be parsed as numbers. */
const asciiDigits = (s: string): string => s.replace(/[०-९]/g, (d) => String('०१२३४५६७८९'.indexOf(d)));

const ARROW_LINE =
  /^\s*(?:[-*•]|\d+[.)])?\s*(.+?)\s*[:：]\s*([+-]?[\d.,]+)\s*%?\s*(?:→|->|=>|⟶|➝)\s*([+-]?[\d.,]+)/;

export const parseComparisonFromText = (text: string): ComparisonData | null => {
  if (!text) return null;
  const lines = asciiDigits(text).split(/\r?\n/);

  const rows: ComparisonRow[] = [];
  for (const line of lines) {
    const m = line.match(ARROW_LINE);
    if (!m) continue;
    const metric = m[1].replace(/[*_`#]/g, '').trim();
    const a = toNumber(m[2]);
    const b = toNumber(m[3]);
    if (!metric || a === null || b === null) continue;
    rows.push(buildRow(metric, [a, b]));
  }
  // A single arrow line is probably not a comparison table
  if (rows.length < 2) return null;

  // Period names: the first two distinct years mentioned in the text, else generic labels
  const years = Array.from(new Set(asciiDigits(text).match(/\b(?:19|20)\d{2}\b/g) ?? []));
  const periods = years.length >= 2 ? [years[0], years[1]] : ['Pehle', 'Baad'];

  return { periods, rows };
};

// ---------------------------------------------------------------------------
// Formatting + colours
// ---------------------------------------------------------------------------

const fmt = (n: number | null, digits = 2): string =>
  n === null ? '—' : n.toLocaleString('en-IN', { maximumFractionDigits: digits });

/** A table/graph value: adds "%" for percentage rows. */
const fmtVal = (n: number | null, isPercent?: boolean): string => (n === null ? '—' : `${fmt(n)}${isPercent ? '%' : ''}`);

/** Include every period supplied by the backend, including all selected quarters. */
const graphPeriodIndexes = (count: number): number[] => Array.from({ length: count }, (_, i) => i);
/** Spread hues across any number of periods instead of relying on a fixed 2–3 color map. */
const periodColor = (index: number, count: number): string =>
  `hsl(${Math.round((index * 360) / Math.max(count, 1))} 76% 56%)`;

// ---------------------------------------------------------------------------
// Change cell (Badlav)
// ---------------------------------------------------------------------------

const ChangeCell: React.FC<{ row: ComparisonRow }> = ({ row }) => {
  if (row.change === null) return <span className="text-slate-500">—</span>;
  const up = row.change > 0;
  const flat = row.change === 0;
  const Icon = flat ? Minus : up ? TrendingUp : TrendingDown;
  const color = flat ? 'text-slate-400' : up ? 'text-emerald-400' : 'text-rose-400';
  const sign = up ? '+' : '';
  return (
    <span className={`inline-flex items-center gap-1 font-medium ${color}`}>
      <Icon className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
      {row.isPercent ? (
        // Percent rows: the change is in percentage points, not a % of a %
        <span title="percentage points">
          {sign}
          {fmt(row.change)} pp
        </span>
      ) : (
        <span>
          {sign}
          {fmt(row.change)}
          {row.changePct !== null && (
            <span className="text-[11px] opacity-80">
              {' '}
              ({sign}
              {fmt(row.changePct, 1)}%)
            </span>
          )}
        </span>
      )}
    </span>
  );
};

// ---------------------------------------------------------------------------
// Comparison table
// ---------------------------------------------------------------------------

const ComparisonTable: React.FC<{ data: ComparisonData }> = ({ data }) => {
  // "Badlav" only means something when there are at least two periods
  const showChange = data.periods.length >= 2;
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-800">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="bg-slate-900/80 text-slate-300">
            <th scope="col" className="text-left font-semibold px-3 py-2">मद</th>
            {data.periods.map((p) => (
              <th key={p} scope="col" className="text-right font-semibold px-3 py-2 whitespace-nowrap">
                {p}
              </th>
            ))}
            {showChange && <th scope="col" className="text-right font-semibold px-3 py-2">बदलाव</th>}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, ri) => (
            <tr key={`${row.metric}-${ri}`} className="border-t border-slate-800/80 hover:bg-slate-900/50">
              <th scope="row" className="text-left font-normal text-slate-200 px-3 py-2">
                {row.metric}
              </th>
              {row.values.map((v, vi) => (
                <td key={vi} className="text-right tabular-nums text-slate-200 px-3 py-2 whitespace-nowrap">
                  {fmtVal(v, row.isPercent)}
                </td>
              ))}
              {showChange && (
                <td className="text-right tabular-nums px-3 py-2 whitespace-nowrap">
                  <ChangeCell row={row} />
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Bar graph
// Vertical grouped bars: one group per metric, one bar per period, all on one
// shared scale so the metrics can be compared with each other. Count rows and
// percentage rows are never mixed on one axis (counts are drawn when present).
// ---------------------------------------------------------------------------

/** A round axis maximum (1, 2, 2.5, 5 × 10^n) at or above `v`. */
const niceMax = (v: number): number => {
  if (v <= 0) return 1;
  const exp = Math.pow(10, Math.floor(Math.log10(v)));
  const step = [1, 2, 2.5, 5, 10].find((s) => s * exp >= v) ?? 10;
  return step * exp;
};

const PLOT_HEIGHT = 220;
const TICK_COUNT = 4;

const ComparisonBars: React.FC<{ data: ComparisonData }> = ({ data }) => {
  const [hover, setHover] = useState<{ r: number; p: number } | null>(null);
  const idx = graphPeriodIndexes(data.periods.length);
  const colors = idx.map((_, i) => periodColor(i, idx.length));

  const allDrawable = data.rows.filter((r) => idx.some((i) => r.values[i] !== null));
  const counts = allDrawable.filter((r) => !r.isPercent);
  const pool = counts.length ? counts : allDrawable;
  const drawable = pool;
  if (!drawable.length) return null;
  const isPercent = !counts.length;

  const dataMax = Math.max(
    ...drawable.flatMap((r) => idx.map((i) => r.values[i] ?? 0)).map((v) => (v > 0 ? v : 0)),
    0,
  );
  const axisMax = isPercent ? Math.min(100, niceMax(dataMax)) || 100 : niceMax(dataMax);
  const ticks = Array.from({ length: TICK_COUNT + 1 }, (_, i) => (axisMax / TICK_COUNT) * i);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3" role="img" aria-label="Bar graph of the table values">
      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mb-3 text-[11px] text-slate-300">
        {idx.map((pi, ci) => (
          <span key={pi} className="inline-flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: colors[ci] }} />
            {data.periods[pi]}
          </span>
        ))}
        {counts.length > 0 && counts.length < allDrawable.length && (
          <span className="text-slate-500">(ग्राफ़ में संख्या वाले मद; प्रतिशत वाले मद तालिका में)</span>
        )}
      </div>

      {/* Hovered bar */}
      <div className="h-4 mb-2 text-[11px] text-slate-300 truncate">
        {hover ? (
          <>
            {drawable[hover.r].metric} · {data.periods[idx[hover.p]]}:{' '}
            <span className="font-semibold text-slate-100">{fmtVal(drawable[hover.r].values[idx[hover.p]], drawable[hover.r].isPercent)}</span>
            {data.unit && !drawable[hover.r].isPercent ? ` ${data.unit}` : ''}
          </>
        ) : (
          <span className="text-slate-500">किसी बार पर माउस ले जाएँ</span>
        )}
      </div>

      <div className="flex">
        {/* Y axis */}
        <div className="relative shrink-0 w-10 mt-3 text-[10px] tabular-nums text-slate-500" style={{ height: PLOT_HEIGHT }}>
          {ticks.map((t) => (
            <span key={t} className="absolute right-1.5 translate-y-1/2" style={{ bottom: `${(t / axisMax) * 100}%` }}>
              {fmt(t, 1)}{isPercent ? '%' : ''}
            </span>
          ))}
        </div>

        {/* Plot + labels (scrolls sideways when there are many metrics) */}
        <div className="flex-1 min-w-0 overflow-x-auto">
          <div style={{ minWidth: drawable.length * Math.max(96, idx.length * 24 + 32) }}>
            <div className="relative border-l border-b border-slate-700 mt-3" style={{ height: PLOT_HEIGHT }}>
              {/* Grid lines */}
              {ticks.slice(1).map((t) => (
                <div
                  key={t}
                  className="absolute left-0 right-0 border-t border-dashed border-slate-800"
                  style={{ bottom: `${(t / axisMax) * 100}%` }}
                />
              ))}

              {/* Bar groups */}
              <div className="absolute inset-0 flex items-end">
                {drawable.map((row, ri) => (
                  <div key={`${row.metric}-${ri}`} className="flex-1 h-full flex items-end justify-center gap-0.5 px-1.5">
                    {idx.map((pi, ci) => {
                      const v = row.values[pi];
                      const h = v !== null && axisMax > 0 ? Math.min((Math.max(v, 0) / axisMax) * 100, 100) : 0;
                      const isHover = hover?.r === ri && hover?.p === ci;
                      const dimmed = hover !== null && !isHover;
                      return (
                        <div
                          key={pi}
                          className="relative flex-1 max-w-7 h-full flex items-end cursor-default"
                          onMouseEnter={() => setHover({ r: ri, p: ci })}
                          onMouseLeave={() => setHover(null)}
                        >
                          <div
                            className="w-full rounded-t transition-[height,opacity] duration-500"
                            style={{
                              height: `${v !== null && v > 0 ? Math.max(h, 0.8) : 0}%`,
                              background: colors[ci],
                              opacity: dimmed ? 0.4 : 1,
                            }}
                          />
                          {v !== null && (
                            <span
                              className="absolute left-1/2 -translate-x-1/2 text-[10px] tabular-nums whitespace-nowrap pointer-events-none"
                              style={{ bottom: `calc(${h}% + 2px)`, color: isHover ? '#f1f5f9' : '#94a3b8' }}
                            >
                              {fmtVal(v, row.isPercent)}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            {/* Metric labels under each group */}
            <div className="flex pt-1.5">
              {drawable.map((row, ri) => {
                const label = row.metric;
                return (
                  <div
                    key={`${row.metric}-label-${ri}`}
                    className="flex-1 px-1 text-center text-[10px] leading-tight text-slate-400 line-clamp-3 break-words"
                    title={label}
                  >
                    {label}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Block shown under an answer: title + table + graph
// Data source, best first: backend `comparison` rows → backend `chart_data` → the answer text.
// ---------------------------------------------------------------------------

const ComparisonBlock: React.FC<{
  raw: RawChartData | null | undefined;
  comparison?: BackendComparisonRow[] | null;
  periods?: string[] | null;
  text: string;
}> = ({ raw, comparison, periods, text }) => {
  const [view, setView] = useState<'table' | 'graph'>('table');
  const data = useMemo(() => {
    const chart = normalizeChartData(raw);
    return fromBackendComparison(comparison, periods, chart?.unit) ?? chart ?? parseComparisonFromText(text);
  }, [raw, comparison, periods, text]);
  const graphData = useMemo(() => {
    if (!data) return null;
    const chart = normalizeChartData(raw);
    if (!chart?.rows.length) return data;
    const requestedLabels = new Set(chart.rows.map((row) => row.metric));
    const requestedRows = data.rows.filter((row) => requestedLabels.has(row.metric));
    return requestedRows.length ? { ...data, rows: requestedRows, unit: chart.unit ?? data.unit } : data;
  }, [data, raw]);
  if (!data || !data.rows.length) return null;

  const heading = data.title || (data.periods.length >= 2 ? 'तुलना' : 'आँकड़े');
  const tabClass = (on: boolean) =>
    `inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors cursor-pointer ${
      on ? 'bg-slate-700 text-slate-100' : 'text-slate-400 hover:text-slate-200'
    }`;
  return (
    <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-2.5">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold text-slate-300 flex items-center gap-1.5">
          <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
          {heading}
          {data.unit && <span className="font-normal text-slate-500">· {data.unit}</span>}
        </span>
        {/* One view at a time keeps long answers short */}
        <div className="ml-auto inline-flex rounded-lg bg-slate-900 border border-slate-800 p-0.5" role="tablist">
          <button type="button" role="tab" aria-selected={view === 'table'} onClick={() => setView('table')} className={tabClass(view === 'table')}>
            <Table2 className="w-3.5 h-3.5" /> तालिका
          </button>
          <button type="button" role="tab" aria-selected={view === 'graph'} onClick={() => setView('graph')} className={tabClass(view === 'graph')}>
            <BarChart3 className="w-3.5 h-3.5" /> ग्राफ़
          </button>
        </div>
      </div>
      {view === 'table' ? <ComparisonTable data={data} /> : <ComparisonBars data={graphData ?? data} />}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Referenced sources, grouped by document
// One card per document (name + year shown once), with its pages as small chips.
// ---------------------------------------------------------------------------

/** How many page chips a document shows before "+N और". */
const PAGE_CHIP_LIMIT = 6;

const docKey = (s: SourceItem): string => s.document_id || s.document_name || s.docName || 'unknown';
const docName = (s: SourceItem): string =>
  (s.document_name || s.docName || s.document_id || 'Document').replace(/\.(pdf|docx?)$/i, '').replace(/_/g, ' ');
const docYear = (s: SourceItem): string | null => {
  if (s.year) return String(s.year);
  // Not \b: names like "Edition_2025" have "_" (a word char) right before the year
  const m = (s.document_name || s.docName || '').match(/(?:^|\D)((?:19|20)\d{2})(?!\d)/);
  return m ? m[1] : null;
};
const pageOf = (s: SourceItem): number | undefined => s.page_number ?? s.pageNumber;
const kindOf = (s: SourceItem): string => s.content_type || s.collection_type || 'content';

interface SourceGroup {
  key: string;
  name: string;
  year: string | null;
  isReport: boolean;
  items: SourceItem[];
}

const groupSources = (sources: SourceItem[]): SourceGroup[] => {
  const groups = new Map<string, SourceGroup>();
  for (const s of sources) {
    const key = docKey(s);
    let g = groups.get(key);
    if (!g) {
      g = { key, name: docName(s), year: docYear(s), isReport: s.document_type === 'report', items: [] };
      groups.set(key, g);
    }
    g.year = g.year ?? docYear(s);
    // Drop exact duplicates (same page + same kind)
    if (!g.items.some((x) => pageOf(x) === pageOf(s) && kindOf(x) === kindOf(s))) g.items.push(s);
  }
  // Pages in reading order inside each document; documents by year
  const list = [...groups.values()];
  list.forEach((g) => g.items.sort((a, b) => (pageOf(a) ?? 0) - (pageOf(b) ?? 0)));
  return list.sort((a, b) => (a.year ?? '').localeCompare(b.year ?? ''));
};

const DocumentSources: React.FC<{ group: SourceGroup; onSourceClick: (s: SourceItem) => void }> = ({
  group,
  onSourceClick,
}) => {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? group.items : group.items.slice(0, PAGE_CHIP_LIMIT);
  const hidden = group.items.length - shown.length;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 px-3 py-2">
      <div className="flex items-center gap-2 min-w-0 mb-1.5">
        {group.isReport ? (
          <FileText className="w-3.5 h-3.5 shrink-0 text-emerald-400" />
        ) : (
          <BookOpen className="w-3.5 h-3.5 shrink-0 text-indigo-400" />
        )}
        <span className="text-[12px] text-slate-200 truncate" title={group.name}>
          {group.name}
        </span>
        {group.year && (
          <span className="shrink-0 rounded-md bg-indigo-500/15 border border-indigo-500/30 px-1.5 py-px text-[10px] font-semibold text-indigo-200">
            {group.year}
          </span>
        )}
        <span className="shrink-0 ml-auto text-[10px] text-slate-500">
          {group.items.length} पृष्ठ
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {shown.map((src, i) => {
          const kind = kindOf(src);
          return (
            <button
              key={`${pageOf(src)}-${kind}-${i}`}
              type="button"
              onClick={() => onSourceClick(src)}
              title={`${group.name}${group.year ? ` (${group.year})` : ''} — पृष्ठ ${pageOf(src) ?? '?'} (${kind}). क्लिक करके सबूत देखें।`}
              className="inline-flex items-center gap-1 text-[11px] bg-slate-950 hover:bg-indigo-950/70 border border-slate-800 hover:border-indigo-500/50 px-2 py-0.5 rounded-md text-indigo-300 hover:text-indigo-200 transition-colors cursor-pointer group"
            >
              {kind === 'table' && <FileText className="w-3 h-3 text-emerald-400" />}
              <span>पृष्ठ {pageOf(src) ?? '?'}</span>
              {kind !== 'content' && kind !== 'text' && <span className="text-slate-500">· {kind}</span>}
              <ExternalLink className="w-2.5 h-2.5 opacity-50 group-hover:opacity-100" />
            </button>
          );
        })}
        {hidden > 0 && (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="text-[11px] px-2 py-0.5 rounded-md text-slate-400 hover:text-slate-200 border border-dashed border-slate-700 hover:border-slate-500 cursor-pointer transition-colors"
          >
            +{hidden} और
          </button>
        )}
        {expanded && group.items.length > PAGE_CHIP_LIMIT && (
          <button
            type="button"
            onClick={() => setExpanded(false)}
            className="text-[11px] px-2 py-0.5 rounded-md text-slate-500 hover:text-slate-300 cursor-pointer"
          >
            कम दिखाएँ
          </button>
        )}
      </div>
    </div>
  );
};

const SourceGroups: React.FC<{ sources: SourceItem[]; onSourceClick: (s: SourceItem) => void }> = ({
  sources,
  onSourceClick,
}) => {
  const groups = useMemo(() => groupSources(sources), [sources]);
  return (
    <div className="mt-3.5 pt-3 border-t border-slate-800/80 space-y-2">
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
        स्रोत · पृष्ठ पर क्लिक करके सबूत देखें
      </span>
      <div className="space-y-2">
        {groups.map((g) => (
          <DocumentSources key={g.key} group={g} onSourceClick={onSourceClick} />
        ))}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Welcome screen (before the first question)
// ---------------------------------------------------------------------------

const SUGGESTIONS: { text: string; hint: string }[] = [
  { text: '2024 और 2025 अंक में लेखों की तुलना कीजिए', hint: 'पत्रिका · तुलना + ग्राफ़' },
  { text: 'किस लेखक ने सबसे ज़्यादा लिखा?', hint: 'पत्रिका · लेखक' },
  { text: 'AI पर कितने लेख हैं?', hint: 'पत्रिका · विषय' },
  { text: '2024 और 2025 की रिपोर्ट की तुलना कीजिए', hint: 'रिपोर्ट · तुलना + ग्राफ़' },
  { text: 'तीनों क्षेत्रों को कुल कितने पत्र भेजे गए?', hint: 'रिपोर्ट · गणना' },
  { text: 'पत्रिका की कविताएँ कौन-सी हैं?', hint: 'पत्रिका · सूची' },
];

const Welcome: React.FC<{ onSuggestion?: (q: string) => void; scopeLabel?: string }> = ({ onSuggestion, scopeLabel }) => (
  <div className="h-full flex flex-col items-center justify-center text-center px-4 py-10">
    <div className="w-12 h-12 rounded-2xl bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center mb-4">
      <Sparkles className="w-6 h-6 text-indigo-300" />
    </div>
    <h2 className="text-xl md:text-2xl font-semibold text-slate-100">नमस्ते! क्या जानना चाहेंगे?</h2>
    <p className="mt-2 max-w-lg text-sm text-slate-400 leading-relaxed">
      राजभाषा पत्रिकाओं और रिपोर्टों से पूछिए — हिंदी, English या Hinglish में। हर जवाब के साथ स्रोत पृष्ठ मिलेंगे,
      और तुलना वाले सवालों पर तालिका व ग्राफ़।
    </p>
    {scopeLabel && (
      <p className="mt-2 text-[11px] text-slate-500">
        अभी खोज: <span className="text-slate-300">{scopeLabel}</span> · दस्तावेज़ सूची से फ़ाइलें चुनें
      </p>
    )}
    {onSuggestion && (
      <div className="mt-6 w-full max-w-2xl grid grid-cols-1 sm:grid-cols-2 gap-2 text-left">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.text}
            type="button"
            onClick={() => onSuggestion(s.text)}
            className="group text-left rounded-xl border border-slate-800 bg-slate-900/60 hover:bg-slate-900 hover:border-indigo-500/50 px-3.5 py-2.5 transition-colors cursor-pointer"
          >
            <span className="block text-[13px] text-slate-200 group-hover:text-white">{s.text}</span>
            <span className="block mt-0.5 text-[10px] text-slate-500">{s.hint}</span>
          </button>
        ))}
      </div>
    )}
  </div>
);

// ---------------------------------------------------------------------------
// Message list
// ---------------------------------------------------------------------------

export const ChatMessages: React.FC<ChatMessagesProps> = ({
  messages,
  isQuerying,
  copiedId,
  onCopy,
  onSourceClick,
  onSuggestion,
  scopeLabel,
}) => {
  const chatEndRef = useRef<HTMLDivElement>(null);
  const lastMsgRef = useRef<HTMLDivElement>(null);

  // The welcome text is shown as the welcome screen, never as a bubble
  const shown = messages.filter((m) => m.id !== 'welcome');

  // Keep the newest message in view. Long answers (table + graph) are scrolled
  // to their start, so the reader sees the answer first.
  useEffect(() => {
    const last = shown[shown.length - 1];
    if (!isQuerying && last?.sender === 'assistant' && lastMsgRef.current) {
      lastMsgRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages, isQuerying]); // eslint-disable-line react-hooks/exhaustive-deps

  if (shown.length === 0 && !isQuerying) {
    return (
      <div className="flex-1 min-h-0 overflow-y-auto">
        <Welcome onSuggestion={onSuggestion} scopeLabel={scopeLabel} />
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-3 md:px-6 py-6 space-y-6">
        {shown.map((msg, i) =>
          msg.sender === 'user' ? (
            <div key={msg.id} ref={i === shown.length - 1 ? lastMsgRef : undefined} className="flex flex-col items-end scroll-mt-4">
              <div className="max-w-[85%] rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2.5 text-sm leading-relaxed text-white whitespace-pre-wrap">
                {msg.text}
              </div>
              <div className="mt-1 px-1 flex items-center gap-1.5 text-[10px] text-slate-500">
                <span>{msg.timestamp}</span>
                <span>·</span>
                <span className="inline-flex items-center gap-1">
                  <Layers className="w-3 h-3" /> {msg.scopeLabel || 'सभी दस्तावेज़'}
                </span>
              </div>
            </div>
          ) : (
            <div key={msg.id} ref={i === shown.length - 1 ? lastMsgRef : undefined} className="flex gap-3 scroll-mt-4">
              <div className="w-7 h-7 shrink-0 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center mt-0.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="prose prose-invert prose-sm max-w-none text-slate-200 leading-relaxed">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>

                {/* Comparison table / bar graph (backend data, else read from the answer text) */}
                <ComparisonBlock
                  raw={msg.chartData ?? msg.chart_data}
                  comparison={msg.comparison}
                  periods={msg.periods}
                  text={msg.text}
                />

                {/* Referenced sources, grouped by document */}
                {msg.sources && msg.sources.length > 0 && (
                  <SourceGroups sources={msg.sources} onSourceClick={onSourceClick} />
                )}

                <div className="mt-2 flex items-center gap-3 text-[10px] text-slate-500">
                  <span>{msg.timestamp}</span>
                  <button
                    type="button"
                    onClick={() => onCopy(msg.text, msg.id)}
                    className="inline-flex items-center gap-1 hover:text-slate-300 cursor-pointer transition-colors"
                  >
                    {copiedId === msg.id ? (
                      <>
                        <Check className="w-3 h-3 text-emerald-400" />
                        <span className="text-emerald-400">कॉपी हो गया</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        <span>कॉपी करें</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          ),
        )}

        {isQuerying && (
          <div className="flex gap-3" aria-live="polite">
            <div className="w-7 h-7 shrink-0 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center">
              <RefreshCw className="w-3.5 h-3.5 text-indigo-300 animate-spin" />
            </div>
            <div className="flex-1 pt-1 space-y-2">
              <p className="text-xs text-slate-400">दस्तावेज़ों में खोज रहा हूँ…</p>
              <div className="h-2.5 w-3/4 rounded bg-slate-800 animate-pulse" />
              <div className="h-2.5 w-1/2 rounded bg-slate-800 animate-pulse" />
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
    </div>
  );
};
