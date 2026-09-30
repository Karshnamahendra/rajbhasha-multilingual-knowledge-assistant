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
 * Raw `chart_data` as the backend sends it (report_metrics.py / magazine_metrics.py):
 *
 *    { type: "bar", labels: ["Metric A", ...],
 *      series: [{ name: "2024", values: [..] }, { name: "2025", values: [..] }], unit: "" }
 *
 * `normalizeChartData()` in ChatMessages.tsx also accepts a few other common shapes
 * (Chart.js `datasets`, row lists, `{years, metrics}`) so a format change does not break the UI.
 */
export type RawChartData =
  | {
      labels?: (string | number)[];
      type?: string;
      series?: { name?: string | number; values?: (number | string | null)[] }[];
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

/**
 * One row of the backend's `comparison` list (report / magazine comparison).
 * `values` is keyed by period, e.g. { "2024": 12, "2025": 18 }.
 */
export interface BackendComparisonRow {
  metric: string;
  region?: string | null;
  section_no?: string;
  section?: string;
  is_percent?: boolean;
  values: Record<string, number | null>;
  change?: number | null;
  change_pct?: number | null;
}

/** One row of the normalised comparison table. */
export interface ComparisonRow {
  metric: string;
  /** True when the values are percentages (shown with %, change in percentage points). */
  isPercent?: boolean;
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
  /** Graph data from the backend's `chart_data` field (either spelling is read). */
  chartData?: RawChartData | null;
  chart_data?: RawChartData | null;
  /** Comparison rows from the backend's `comparison` field (preferred source for the table). */
  comparison?: BackendComparisonRow[] | null;
  /** Period labels from the backend's `periods` field, oldest first. */
  periods?: string[] | null;
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