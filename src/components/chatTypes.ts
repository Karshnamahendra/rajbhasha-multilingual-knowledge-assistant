// Shapes used by the chat UI components (MainChatbot and its children).

export interface SourceItem {
  chunk_id?: string;
  chunkId?: string;
  document_id?: string;
  docName?: string;
  document_name?: string;
  document_type?: 'magazine' | 'report';
  year?: number | null;
  content_type?: string;
  page_number?: number;
  pageNumber?: number;
  collection_type?: string;
  table_id?: string;
  extractionConfidence?: number;
  text?: string;
  quote?: string;
  highlightText?: string;
  section?: string;
  relevanceScore?: number;
}

// ---------------------------------------------------------------------------
// Comparison chart data
// ---------------------------------------------------------------------------

/**
 * Raw `chart_data` exactly as the backend sends it. The frontend accepts the
 * common shapes below and normalises them with `normalizeChartData()` in
 * ChatMessages.tsx, so the backend format can change without breaking the UI:
 *
 *  1. Chart.js style:  { labels: ["Metric A", ...],
 *                        datasets: [{ label: "2024", data: [..] }, { label: "2025", data: [..] }] }
 *  2. Rows:            [{ metric: "Metric A", "2024": 12, "2025": 18 }, ...]
 *                      (also accepted inside { rows | data | items: [...] })
 *  3. Years+metrics:   { years: [2024, 2025], metrics: [{ name: "Metric A", values: [12, 18] }] }
 */
export type RawChartData =
  | {
      labels?: (string | number)[];
      datasets?: { label?: string | number; data?: (number | string | null)[] }[];
      years?: (string | number)[];
      metrics?: { name?: string; metric?: string; label?: string; values?: (number | string | null)[] }[];
      rows?: Record<string, unknown>[];
      data?: Record<string, unknown>[];
      items?: Record<string, unknown>[];
      title?: string;
      unit?: string;
      [key: string]: unknown;
    }
  | Record<string, unknown>[]
  | string;

/** One row of the normalised comparison table. */
export interface ComparisonRow {
  metric: string;
  /** Value for each period, in the same order as `ComparisonData.periods`. */
  values: (number | null)[];
  /** Absolute change from the first to the last period (null if either is missing). */
  change: number | null;
  /** Percentage change from the first to the last period (null if base is 0/missing). */
  changePct: number | null;
}

/** Normalised data the table and bar graph render from. */
export interface ComparisonData {
  title?: string;
  unit?: string;
  /** Column labels, e.g. ["2024", "2025"]. */
  periods: string[];
  rows: ComparisonRow[];
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  sources?: SourceItem[];
  detectedScript?: string;
  scopeLabel?: string;
  /** Comparison data from the backend's `chart_data` field (either spelling is read). */
  chartData?: RawChartData | null;
  chart_data?: RawChartData | null;
}

export interface OllamaStatusInfo {
  available: boolean;
  model: string;
}

export interface UploadSummary {
  name: string;
  type?: string;
  year?: number | null;
  pages?: number;
  chunks?: number;
}