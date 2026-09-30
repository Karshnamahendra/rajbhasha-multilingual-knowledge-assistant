import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bot,
  User,
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
    const base = buildRow((r.region ? `${r.region} क्षेत्र – ` : '') + (r.metric || ''), values);
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

/**
 * Period colours, fixed by position (first period → amber, last → indigo).
 * Validated for colour-blind separation and 3:1 contrast on the dark slate-950 surface.
 */
const PERIOD_COLORS: Record<number, string[]> = {
  1: ['#6366f1'],
  2: ['#d97706', '#6366f1'],
  3: ['#d97706', '#db2777', '#6366f1'],
};
/** The graph draws at most 3 periods; with more, it shows first vs last (the table keeps all). */
const graphPeriodIndexes = (count: number): number[] =>
  count <= 3 ? Array.from({ length: count }, (_, i) => i) : [0, count - 1];

/** The graph shows at most this many metrics; the table always has every row. */
const GRAPH_ROW_LIMIT = 12;

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
            <th scope="col" className="text-left font-semibold px-3 py-2">Metric</th>
            {data.periods.map((p) => (
              <th key={p} scope="col" className="text-right font-semibold px-3 py-2 whitespace-nowrap">
                {p}
              </th>
            ))}
            {showChange && <th scope="col" className="text-right font-semibold px-3 py-2">Badlav</th>}
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
// Each metric gets its own row and its own scale, so metrics with very
// different sizes (e.g. a count and a percentage) stay readable side by side.
// ---------------------------------------------------------------------------

const ComparisonBars: React.FC<{ data: ComparisonData }> = ({ data }) => {
  const [hover, setHover] = useState<{ r: number; p: number } | null>(null);
  const idx = graphPeriodIndexes(data.periods.length);
  const colors = PERIOD_COLORS[idx.length] ?? PERIOD_COLORS[2];

  const allDrawable = data.rows.filter((r) => idx.some((i) => r.values[i] !== null));
  const drawable = allDrawable.slice(0, GRAPH_ROW_LIMIT);
  if (!drawable.length) return null;

  // Several periods: each metric has its own scale (compares years within a metric).
  // One period (e.g. works per author): one shared scale, so the rows can be compared.
  const sharedMax =
    idx.length === 1
      ? Math.max(...drawable.map((r) => r.values[idx[0]] ?? 0).map((v) => (v > 0 ? v : 0)), 0)
      : null;

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
        {data.periods.length > 3 && (
          <span className="text-slate-500">(graph mein pehla aur aakhri saal; baaki table mein)</span>
        )}
        {allDrawable.length > drawable.length && (
          <span className="text-slate-500">
            (graph mein pehle {drawable.length} mad; saare {allDrawable.length} table mein)
          </span>
        )}
      </div>

      <div className="space-y-3">
        {drawable.map((row, ri) => {
          const vals = idx.map((i) => row.values[i]);
          const max = sharedMax ?? Math.max(...vals.map((v) => (v !== null && v > 0 ? v : 0)), 0);
          return (
            <div key={`${row.metric}-${ri}`}>
              <div className="text-[11px] text-slate-300 mb-1 truncate" title={row.metric}>
                {row.metric}
              </div>
              <div className="flex flex-col gap-0.5">
                {vals.map((v, ci) => {
                  const pct = v !== null && max > 0 ? Math.max((Math.max(v, 0) / max) * 100, v > 0 ? 1 : 0) : 0;
                  const isHover = hover?.r === ri && hover?.p === ci;
                  const dimmed = hover !== null && hover.r === ri && !isHover;
                  return (
                    <div
                      key={ci}
                      className="relative flex items-center gap-2 h-4 cursor-default"
                      onMouseEnter={() => setHover({ r: ri, p: ci })}
                      onMouseLeave={() => setHover(null)}
                    >
                      <div className="flex-1 h-3">
                        <div
                          className="h-full rounded-r transition-[width,opacity] duration-500"
                          style={{
                            width: `${pct}%`,
                            background: colors[ci],
                            opacity: dimmed ? 0.45 : 1,
                          }}
                        />
                      </div>
                      <span className="w-20 shrink-0 text-right text-[11px] tabular-nums text-slate-300">
                        {fmtVal(v, row.isPercent)}
                      </span>
                      {isHover && (
                        <div className="absolute left-0 -top-7 z-10 pointer-events-none whitespace-nowrap rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-[11px] text-slate-100 shadow-lg">
                          {row.metric} · {data.periods[idx[ci]]}:{' '}
                          <span className="font-semibold">{fmtVal(v, row.isPercent)}</span>
                          {data.unit && !row.isPercent ? ` ${data.unit}` : ''}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
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
  const data = useMemo(() => {
    const chart = normalizeChartData(raw);
    return fromBackendComparison(comparison, periods, chart?.unit) ?? chart ?? parseComparisonFromText(text);
  }, [raw, comparison, periods, text]);
  if (!data || !data.rows.length) return null;

  const heading = data.title || (data.periods.length >= 2 ? 'Tulna (Comparison)' : 'Aankde (Figures)');
  return (
    <div className="mt-3.5 pt-3 border-t border-slate-800/80 space-y-3">
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
        <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
        {heading}
        {data.unit && <span className="normal-case font-normal text-slate-500">· {data.unit}</span>}
      </span>
      <ComparisonTable data={data} />
      <ComparisonBars data={data} />
    </div>
  );
};

// ---------------------------------------------------------------------------
// Referenced sources, grouped by document
// One card per document (name + year shown once), with its pages as small chips.
// ---------------------------------------------------------------------------

/** How many page chips a document shows before "+N aur". */
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
          {group.items.length} {group.items.length === 1 ? 'page' : 'pages'}
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
              title={`${group.name}${group.year ? ` (${group.year})` : ''} — Page ${pageOf(src) ?? '?'} (${kind}). Click to inspect.`}
              className="inline-flex items-center gap-1 text-[11px] bg-slate-950 hover:bg-indigo-950/70 border border-slate-800 hover:border-indigo-500/50 px-2 py-0.5 rounded-md text-indigo-300 hover:text-indigo-200 transition-colors cursor-pointer group"
            >
              {kind === 'table' && <FileText className="w-3 h-3 text-emerald-400" />}
              <span>Page {pageOf(src) ?? '?'}</span>
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
            +{hidden} aur
          </button>
        )}
        {expanded && group.items.length > PAGE_CHIP_LIMIT && (
          <button
            type="button"
            onClick={() => setExpanded(false)}
            className="text-[11px] px-2 py-0.5 rounded-md text-slate-500 hover:text-slate-300 cursor-pointer"
          >
            kam dikhayein
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
        Srot (Sources) · page par click karke saboot dekhein
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
// Message list
// ---------------------------------------------------------------------------

export const ChatMessages: React.FC<ChatMessagesProps> = ({
  messages,
  isQuerying,
  copiedId,
  onCopy,
  onSourceClick,
}) => {
  const chatEndRef = useRef<HTMLDivElement>(null);
  const lastMsgRef = useRef<HTMLDivElement>(null);

  // Keep the newest message in view. Long answers (table + graph) are scrolled
  // to their start, so the reader sees the answer first rather than the graph's end.
  useEffect(() => {
    const last = messages[messages.length - 1];
    if (!isQuerying && last?.sender === 'assistant' && lastMsgRef.current) {
      lastMsgRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages, isQuerying]);

  return (
    <div className="p-4 md:p-6 min-h-[300px] max-h-[70vh] overflow-y-auto space-y-4">
      {messages.map((msg, i) => (
        <div
          key={msg.id}
          ref={i === messages.length - 1 ? lastMsgRef : undefined}
          className={`flex flex-col scroll-mt-2 ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
        >
          {/* Sender Label */}
          <div className="flex items-center gap-2 text-[11px] text-slate-400 mb-1 px-1">
            {msg.sender === 'assistant' ? (
              <span className="text-indigo-400 font-semibold flex items-center gap-1">
                <Bot className="w-3.5 h-3.5" /> Assistant
              </span>
            ) : (
              <span className="text-slate-300 font-semibold flex items-center gap-1">
                <User className="w-3.5 h-3.5" /> You
              </span>
            )}
            <span>• {msg.timestamp}</span>
            {msg.sender === 'user' && (
              <span className="text-indigo-300/80">• 📎 {msg.scopeLabel || 'All documents'}</span>
            )}
          </div>

          {/* Message Bubble */}
          <div
            className={`max-w-[88%] rounded-2xl p-4 text-sm leading-relaxed ${
              msg.sender === 'user'
                ? 'bg-indigo-600 text-white rounded-tr-sm shadow-md'
                : 'bg-slate-950/90 border border-slate-800 text-slate-200 rounded-tl-sm'
            }`}
          >
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            </div>

            {/* Comparison table + bar graph (backend data, else read from the answer text) */}
            {msg.sender === 'assistant' && <ComparisonBlock
                raw={msg.chartData ?? msg.chart_data}
                comparison={msg.comparison}
                periods={msg.periods}
                text={msg.text}
              />}

            {/* Referenced sources, grouped by document */}
            {msg.sources && msg.sources.length > 0 && (
              <SourceGroups sources={msg.sources} onSourceClick={onSourceClick} />
            )}

            {/* Copy button */}
            {msg.sender === 'assistant' && (
              <div className="mt-2 flex justify-end">
                <button
                  onClick={() => onCopy(msg.text, msg.id)}
                  className="text-slate-500 hover:text-slate-300 text-[11px] flex items-center gap-1 cursor-pointer transition-colors"
                >
                  {copiedId === msg.id ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>
      ))}

      {isQuerying && (
        <div className="flex items-center gap-2 text-xs text-indigo-400 animate-pulse p-2">
          <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-400" />
          <span>Searching Qdrant Vector Store &amp; Synthesizing Grounded Answer...</span>
        </div>
      )}
      <div ref={chatEndRef} />
    </div>
  );
};