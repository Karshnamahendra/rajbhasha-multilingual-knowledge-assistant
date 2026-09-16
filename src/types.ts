export interface DocumentPage {
  page_number?: number;
  pageNumber: number;
  text: string;
  char_count?: number;
  charCount: number;
  word_count?: number;
  wordCount: number;
}

export interface DocumentChunk {
  id: string;
  chunk_id?: string;
  docId: string;
  document_id?: string;
  docName: string;
  document_name?: string;
  pageNumber: number;
  page_number?: number;
  chunkIndex: number;
  chunk_index?: number;
  text: string;
  char_count?: number;
  charCount?: number;
  word_count?: number;
  wordCount?: number;
  startChar?: number;
  endChar?: number;
  embedding?: number[];
  tokenCount?: number;
  metadata?: any;
}

export interface UploadedDocument {
  id: string;
  document_id?: string;
  fileName: string;
  document_name?: string;
  fileSize: number;
  fileType?: string;
  document_type?: 'magazine' | 'report';
  year?: number | null;
  quarter?: string | null;
  report_period?: string | null;
  totalPages: number;
  totalWords: number;
  totalChars?: number;
  uploadDate: string;
  pages: DocumentPage[];
  fullText: string;
  chunkCount?: number;
  chunks?: DocumentChunk[];
  chunkSize?: number;
  chunkOverlap?: number;
  dataUrl?: string;
}

export interface HighlightSpan {
  start: number;
  end: number;
  text: string;
}

export interface SourceQuote {
  quote: string;
  text?: string;
  pageNumber: number;
  page_number?: number;
  chunkId: string;
  chunk_id?: string;
  docName: string;
  document_name?: string;
  document_id?: string;
  document_type?: 'magazine' | 'report';
  year?: number | null;
  content_type?: string;
  relevanceScore: number;
  score?: number;
  similarity?: number;
  highlightText?: string;
  highlighted_spans?: HighlightSpan[];
  collection_type?: string;
  table_id?: string;
  extractionConfidence?: number;
}

export interface QueryResponse {
  answer: string;
  grounded: boolean;
  confidenceScore: number;
  sources: SourceQuote[];
  retrievedChunks: {
    chunk: DocumentChunk;
    similarity: number;
    distance?: number;
  }[];
  modelUsed: string;
  processingTimeMs: number;
  originalQuery?: string;
  normalizedQuery?: string;
  detectedLanguage?: string;
  detectedIntent?: string;
  concepts?: string[];
  embeddingModel?: string;
  embeddingDimension?: number;
  queryEmbeddingDimension?: number;
  contextSentToLLM?: string;
  // Stage 1: all retrieved candidates (broad net)
  retrievedCandidates?: { chunk: DocumentChunk; similarity: number; distance?: number }[];
  // Stage 2: best evidence selected (sent to LLM)
  bestEvidenceChunks?: { chunk: DocumentChunk; similarity: number; distance?: number }[];
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  sourceDoc?: string;
  sourcePage?: number;
  confidenceScore?: number;
  queryResponse?: QueryResponse;
  queryDetails?: any;
  highlightTarget?: {
    pageNumber: number;
    quote: string;
    chunkId?: string;
  };
}

export interface OllamaStatus {
  available: boolean;
  models: string[];
  currentModel: string;
  baseUrl: string;
  error?: string;
}

export type LLMEngine = 'ollama' | 'gemini';

