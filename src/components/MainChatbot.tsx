import React, { useState, useEffect, useRef } from 'react';
import {
  UploadCloud,
  Send,
  Sparkles,
  FileText,
  Bot,
  User,
  AlertCircle,
  RefreshCw,
  ChevronDown,
  Database,
  Trash2,
  BookOpen,
  Copy,
  Check,
  X,
  ExternalLink,
  Calendar,
  FileCheck2,
  FolderOpen
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { UploadedDocument } from '../types';
import { HighlightText } from './HighlightText';

interface SourceItem {
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

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  sources?: SourceItem[];
  detectedScript?: string;
}

export const MainChatbot: React.FC = () => {
  const conversationIdRef = useRef(`conversation-${crypto.randomUUID()}`);
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  
  // Active filter: 'all' | 'all_magazines' | 'all_reports' | specific doc id
  const [activeScope, setActiveScope] = useState<string>('all');
  
  // Document category tab: 'magazines' | 'reports'
  const [activeCategoryTab, setActiveCategoryTab] = useState<'magazines' | 'reports'>('magazines');

  // Upload modal state
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [uploadCategory, setUploadCategory] = useState<'magazine' | 'report'>('magazine');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgressStep, setUploadProgressStep] = useState<number>(0);
  const [uploadStatusText, setUploadStatusText] = useState<string>('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccessSummary, setUploadSuccessSummary] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Chat Query state
  const [inputQuery, setInputQuery] = useState<string>('');
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Active Source Chunk Modal Inspection State
  const [selectedChunk, setSelectedChunk] = useState<SourceItem | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: 'नमस्ते! Hello! I am your Rajbhasha Multilingual Knowledge Assistant.\n\nI dynamically search across your **Magazines** and **Reports** stored in the Qdrant Vector Database. Ask me anything in **Hindi, English, or Hinglish** — about articles, authors, poetry, quarterly reports, tables, or proformas.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  // Ollama status
  const [ollamaStatus, setOllamaStatus] = useState<{ available: boolean; model: string }>({
    available: false,
    model: 'llama3.2'
  });

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchDocuments();
      checkOllama();
    }, 400);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isQuerying]);

  const checkOllama = async () => {
    try {
      const res = await fetch('/api/ollama/status');
      const data = await res.json();
      if (data && (data.available || data.status === 'connected')) {
        setOllamaStatus({ available: true, model: data.model || data.currentModel || 'llama3.2' });
      } else {
        setOllamaStatus({ available: false, model: 'llama3.2' });
      }
    } catch {
      setOllamaStatus({ available: false, model: 'llama3.2' });
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) return;
      const data = await res.json();
      const docList: UploadedDocument[] = Array.isArray(data) ? data : (data.documents || []);
      setDocuments(docList);
    } catch {
      console.warn('[fetchDocuments] Backend not reachable yet.');
    }
  };

  const handleDeleteDocument = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to remove this document and its Qdrant vectors?')) return;
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' });
      if (res.ok) {
        const rem = documents.filter((d) => d.id !== docId);
        setDocuments(rem);
        if (activeScope === docId) {
          setActiveScope('all');
        }
      }
    } catch (err) {
      console.error('Failed to delete document:', err);
    }
  };

  const handleStartUpload = (category: 'magazine' | 'report') => {
    setUploadCategory(category);
    setUploadError(null);
    setUploadSuccessSummary(null);
    setUploadProgressStep(0);
    setUploadStatusText('');
    fileInputRef.current?.click();
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const file = files[0];
    setUploadError(null);
    setUploadSuccessSummary(null);
    setIsUploading(true);

    try {
      // Step 1: Uploading
      setUploadProgressStep(1);
      setUploadStatusText('Uploading document file...');

      const formData = new FormData();
      formData.append('file', file);
      formData.append('document_type', uploadCategory);
      formData.append('category', uploadCategory);

      // Step 2: Extracting
      setUploadProgressStep(2);
      setUploadStatusText('Extracting text and tables...');

      const uploadPromise = fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      // Simulation steps for user visual feedback while server processes
      const stepTimer1 = setTimeout(() => {
        setUploadProgressStep(3);
        setUploadStatusText('Detecting structure (Index / Content / Tables)...');
      }, 1500);

      const stepTimer2 = setTimeout(() => {
        setUploadProgressStep(4);
        setUploadStatusText('Generating multilingual embeddings (384-D)...');
      }, 3500);

      const res = await uploadPromise;
      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Upload failed: ${errText}`);
      }

      setUploadProgressStep(5);
      setUploadStatusText('Stored in Qdrant Vector DB!');

      const data = await res.json();

      const richDoc: UploadedDocument = {
        id: data.document_id || data.id || file.name,
        fileName: data.fileName || data.name || file.name,
        fileSize: file.size,
        document_type: data.document_type || uploadCategory,
        year: data.year || null,
        quarter: data.quarter || null,
        report_period: data.report_period || null,
        totalPages: data.totalPages || (data.pages ? data.pages.length : 1),
        totalWords: data.totalWords || 0,
        totalChars: data.totalChars || (data.fullText ? data.fullText.length : 0),
        uploadDate: new Date().toISOString(),
        pages: data.pages || [],
        fullText: data.fullText || '',
        chunkCount: data.chunkCount || (data.chunks ? data.chunks.length : 0),
        chunks: data.chunks || [],
      };

      setUploadSuccessSummary({
        name: richDoc.fileName,
        type: richDoc.document_type,
        year: richDoc.year,
        pages: richDoc.totalPages,
        chunks: richDoc.chunkCount
      });

      setDocuments((prev) => {
        const existing = prev.find((d) => d.id === richDoc.id);
        if (existing) return prev.map((d) => (d.id === richDoc.id ? richDoc : d));
        return [...prev, richDoc];
      });

      // Auto-switch tab to the category just uploaded
      setActiveCategoryTab(uploadCategory === 'magazine' ? 'magazines' : 'reports');

    } catch (err: any) {
      console.error('[Upload Error]', err);
      setUploadError(err.message || 'Document upload failed. Select a valid PDF or DOCX file.');
      setUploadProgressStep(0);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || isQuerying) return;

    const userQueryText = inputQuery.trim();

    // Determine target document or category filter
    let docIdFilter: string | undefined = undefined;
    let docTypeFilter: string | undefined = undefined;

    if (activeScope === 'all') {
      docIdFilter = undefined;
      docTypeFilter = undefined;
    } else if (activeScope === 'all_magazines') {
      docIdFilter = undefined;
      docTypeFilter = 'magazine';
    } else if (activeScope === 'all_reports') {
      docIdFilter = undefined;
      docTypeFilter = 'report';
    } else {
      docIdFilter = activeScope;
      const targetDoc = documents.find((d) => d.id === activeScope);
      if (targetDoc?.document_type) {
        docTypeFilter = targetDoc.document_type;
      }
    }

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: userQueryText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery('');
    setIsQuerying(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQueryText,
          document_id: docIdFilter || '',
          document_type: docTypeFilter,
          query_mode: 'AUTO',
          conversation_id: conversationIdRef.current
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();

      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: data.answer || 'No relevant information found in the documents.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        sources: data.sources || [],
        detectedScript: data.detected_script
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: 'Error processing your request. Please ensure the backend and Ollama are running.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsQuerying(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const magazines = documents.filter((d) => (d.document_type || 'magazine') === 'magazine');
  const reports = documents.filter((d) => d.document_type === 'report');

  return (
    <div className="w-full max-w-5xl mx-auto flex flex-col space-y-4 py-4 px-3 md:px-4">

      {/* ───────────────────────────────────────────────────────────── */}
      {/* Header & Branding */}
      {/* ───────────────────────────────────────────────────────────── */}
      <header className="bg-slate-900/90 border border-slate-800 rounded-3xl p-4 md:p-5 flex flex-col sm:flex-row items-center justify-between gap-4 shadow-xl backdrop-blur-md">
        <div className="flex items-center gap-3 text-center sm:text-left">
          <div className="w-11 h-11 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 shadow-inner">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl md:text-2xl font-extrabold text-white tracking-tight">
              DocuMind • Rajbhasha Multilingual Assistant
            </h1>
            <p className="text-xs text-slate-400">
              Autonomous Document Intelligence • Qdrant Vector DB &amp; Llama 3.2
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-2 bg-slate-950 px-3 py-1.5 rounded-full border border-slate-800 text-xs">
            <Database className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-slate-300 font-medium">Qdrant Active</span>
          </div>

          <div className="flex items-center gap-2 bg-slate-950 px-3.5 py-1.5 rounded-full border border-slate-800 text-xs">
            <span className={`w-2 h-2 rounded-full ${ollamaStatus.available ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
            <span className="text-slate-300 font-medium">
              {ollamaStatus.available ? `Ollama (${ollamaStatus.model})` : 'Ollama Standby'}
            </span>
          </div>
        </div>
      </header>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* Document Management Section: Magazines & Reports */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-4 md:p-5 space-y-4 shadow-lg">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
          {/* Category Tabs */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveCategoryTab('magazines')}
              className={`flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold transition-all cursor-pointer ${
                activeCategoryTab === 'magazines'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 ring-1 ring-indigo-400/40'
                  : 'bg-slate-950 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-slate-800'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              <span>📖 MAGAZINES</span>
              <span className="bg-indigo-950/80 text-indigo-300 px-1.5 py-0.5 rounded-full text-[10px] ml-0.5">
                {magazines.length}
              </span>
            </button>

            <button
              onClick={() => setActiveCategoryTab('reports')}
              className={`flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold transition-all cursor-pointer ${
                activeCategoryTab === 'reports'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 ring-1 ring-indigo-400/40'
                  : 'bg-slate-950 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-slate-800'
              }`}
            >
              <FileText className="w-4 h-4" />
              <span>📊 REPORTS</span>
              <span className="bg-indigo-950/80 text-indigo-300 px-1.5 py-0.5 rounded-full text-[10px] ml-0.5">
                {reports.length}
              </span>
            </button>
          </div>

          {/* Upload Button */}
          <button
            onClick={() => {
              setIsUploadModalOpen(true);
              setUploadError(null);
              setUploadSuccessSummary(null);
            }}
            className="flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs px-5 py-2.5 rounded-2xl transition-all shadow-md shadow-indigo-600/30 active:scale-95 cursor-pointer"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Upload Document</span>
          </button>
        </div>

        {/* Document List under Active Category */}
        <div className="space-y-2">
          {activeCategoryTab === 'magazines' ? (
            magazines.length === 0 ? (
              <div className="text-center py-8 text-slate-500 text-xs">
                <BookOpen className="w-8 h-8 mx-auto text-slate-600 mb-2 opacity-50" />
                <p>No magazines uploaded yet.</p>
                <p className="text-[11px] text-slate-600 mt-1">Click "Upload Document" and select MAGAZINES to add an edition.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {magazines.map((doc) => {
                  const isSelected = activeScope === doc.id;
                  return (
                    <div
                      key={doc.id}
                      onClick={() => setActiveScope(isSelected ? 'all' : doc.id)}
                      className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between space-y-2 ${
                        isSelected
                          ? 'bg-indigo-950/40 border-indigo-500/70 shadow-md shadow-indigo-950/40 ring-1 ring-indigo-500/40'
                          : 'bg-slate-950/70 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/60'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2 overflow-hidden">
                          <BookOpen className="w-4 h-4 text-indigo-400 shrink-0" />
                          <span className="font-semibold text-xs text-slate-200 truncate" title={doc.fileName}>
                            {doc.fileName}
                          </span>
                        </div>
                        <button
                          onClick={(e) => handleDeleteDocument(doc.id, e)}
                          title="Remove Document"
                          className="text-slate-500 hover:text-rose-400 p-1 rounded-lg hover:bg-rose-950/40 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/60">
                        <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                          <Calendar className="w-3 h-3" />
                          {doc.year ? `Year ${doc.year}` : 'Auto-detected'}
                        </span>
                        <span>{doc.totalPages || 1} Pages</span>
                        <span className="bg-slate-800 text-slate-300 text-[10px] px-1.5 py-0.5 rounded-md">
                          Ready
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )
          ) : (
            reports.length === 0 ? (
              <div className="text-center py-8 text-slate-500 text-xs">
                <FileText className="w-8 h-8 mx-auto text-slate-600 mb-2 opacity-50" />
                <p>No reports uploaded yet.</p>
                <p className="text-[11px] text-slate-600 mt-1">Click "Upload Document" and select REPORTS to add official proformas.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {reports.map((doc) => {
                  const isSelected = activeScope === doc.id;
                  return (
                    <div
                      key={doc.id}
                      onClick={() => setActiveScope(isSelected ? 'all' : doc.id)}
                      className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between space-y-2 ${
                        isSelected
                          ? 'bg-indigo-950/40 border-indigo-500/70 shadow-md shadow-indigo-950/40 ring-1 ring-indigo-500/40'
                          : 'bg-slate-950/70 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/60'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2 overflow-hidden">
                          <FileText className="w-4 h-4 text-emerald-400 shrink-0" />
                          <span className="font-semibold text-xs text-slate-200 truncate" title={doc.fileName}>
                            {doc.fileName}
                          </span>
                        </div>
                        <button
                          onClick={(e) => handleDeleteDocument(doc.id, e)}
                          title="Remove Document"
                          className="text-slate-500 hover:text-rose-400 p-1 rounded-lg hover:bg-rose-950/40 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/60">
                        <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                          <Calendar className="w-3 h-3" />
                          {doc.report_period || (doc.year ? `Year ${doc.year}` : 'Official Report')}
                        </span>
                        <span>{doc.totalPages || 1} Pages</span>
                        <span className="bg-slate-800 text-slate-300 text-[10px] px-1.5 py-0.5 rounded-md">
                          Ready
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )
          )}
        </div>

        {/* Scope Filter Dropdown */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-800/60 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 font-medium">Active Query Scope:</span>
            <div className="relative">
              <select
                value={activeScope}
                onChange={(e) => setActiveScope(e.target.value)}
                className="bg-slate-950 border border-slate-800 text-white rounded-xl px-3 py-1.5 appearance-none focus:outline-none focus:border-indigo-500 cursor-pointer pr-7 text-xs"
              >
                <option value="all">📚 All Documents (Magazines &amp; Reports)</option>
                <option value="all_magazines">📖 All Magazines Only</option>
                <option value="all_reports">📊 All Reports Only</option>
                {documents.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.document_type === 'report' ? '📊 ' : '📖 '} {d.fileName}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2 top-2 pointer-events-none" />
            </div>
          </div>

          <span className="text-[11px] text-slate-500">
            {activeScope === 'all'
              ? 'Querying entire knowledge base'
              : activeScope === 'all_magazines'
              ? 'Filtering to magazines'
              : activeScope === 'all_reports'
              ? 'Filtering to reports'
              : 'Focused on 1 document'}
          </span>
        </div>
      </div>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* Unified Chat History & Input */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div className="bg-[#0f172a]/80 border border-slate-800 rounded-3xl shadow-2xl flex flex-col overflow-hidden">

        {/* Messages Stream */}
        <div className="p-4 md:p-6 min-h-[300px] max-h-[480px] overflow-y-auto space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
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

                {/* Referenced Sources Pills */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3.5 pt-3 border-t border-slate-800/80 space-y-2">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                      Referenced Sources (Click to inspect evidence chunk):
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {msg.sources.map((src, idx) => (
                        <button
                          key={idx}
                          type="button"
                          onClick={() => setSelectedChunk(src)}
                          className="inline-flex items-center gap-1.5 text-[11px] bg-slate-900 hover:bg-indigo-950/70 border border-slate-800 hover:border-indigo-500/50 px-2.5 py-1 rounded-lg text-indigo-300 hover:text-indigo-200 transition-all cursor-pointer shadow-sm group"
                        >
                          {src.content_type === 'table' ? (
                            <FileText className="w-3.5 h-3.5 text-emerald-400 group-hover:scale-110 transition-transform" />
                          ) : (
                            <BookOpen className="w-3.5 h-3.5 text-indigo-400 group-hover:scale-110 transition-transform" />
                          )}
                          <span>
                            {src.document_name ? `${src.document_name.slice(0, 18)}... ` : ''}Page {src.page_number} ({src.content_type || src.collection_type || 'content'})
                          </span>
                          <ExternalLink className="w-3 h-3 opacity-60 group-hover:opacity-100" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Copy button */}
                {msg.sender === 'assistant' && (
                  <div className="mt-2 flex justify-end">
                    <button
                      onClick={() => copyToClipboard(msg.text, msg.id)}
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

        {/* Input Footer (No manual query mode selector buttons) */}
        <div className="bg-slate-950/70 border-t border-slate-800/80 p-4 md:p-5">
          <form onSubmit={handleSendMessage} className="space-y-3">
            <div className="relative">
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                placeholder="Ask in Hindi, English, or Hinglish (e.g. 2024 पत्रिका के मुख्य लेख, कविताएं, रिपोर्ट के आंकड़े)..."
                className="w-full bg-slate-900/90 border border-slate-800 rounded-2xl px-4 py-3 text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all"
                disabled={isQuerying}
              />
            </div>

            <div className="flex justify-center">
              <button
                type="submit"
                disabled={isQuerying || !inputQuery.trim()}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs md:text-sm font-semibold px-8 py-2 rounded-2xl flex items-center gap-2 shadow-lg shadow-indigo-600/25 active:scale-95 transition-all cursor-pointer"
              >
                {isQuerying ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                <span>Ask Assistant</span>
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* Upload Modal (Category Selection & File Upload) */}
      {/* ───────────────────────────────────────────────────────────── */}
      {isUploadModalOpen && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-lg w-full p-5 md:p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <UploadCloud className="w-5 h-5 text-indigo-400" />
                <h3 className="text-base font-bold text-slate-100">Upload Document</h3>
              </div>
              <button
                onClick={() => {
                  if (!isUploading) {
                    setIsUploadModalOpen(false);
                    setUploadError(null);
                    setUploadSuccessSummary(null);
                  }
                }}
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
              onChange={(e) => handleFileUpload(e.target.files)}
            />

            {!isUploading && !uploadSuccessSummary && (
              <div className="space-y-4">
                <p className="text-xs text-slate-300">
                  Which type of document are you uploading?
                </p>

                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => handleStartUpload('magazine')}
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
                    onClick={() => handleStartUpload('report')}
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
                    Processing {uploadCategory === 'magazine' ? 'Magazine' : 'Report'}
                  </h4>
                  <p className="text-xs text-indigo-300 font-medium">{uploadStatusText}</p>
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
                    const isDone = uploadProgressStep > s.step;
                    const isCurrent = uploadProgressStep === s.step;
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
            {uploadSuccessSummary && (
              <div className="space-y-4 py-2">
                <div className="bg-emerald-950/40 border border-emerald-500/30 rounded-2xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs">
                    <FileCheck2 className="w-4 h-4" />
                    <span>Successfully Ingested into Qdrant!</span>
                  </div>
                  <p className="text-xs text-slate-200 font-medium">{uploadSuccessSummary.name}</p>
                  <div className="grid grid-cols-3 gap-2 text-[11px] text-slate-400 pt-2 border-t border-emerald-500/20">
                    <div>Category: <span className="text-white font-medium capitalize">{uploadSuccessSummary.type}</span></div>
                    <div>Year: <span className="text-emerald-300 font-bold">{uploadSuccessSummary.year || 'Auto'}</span></div>
                    <div>Pages: <span className="text-white font-medium">{uploadSuccessSummary.pages}</span></div>
                  </div>
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => {
                      setIsUploadModalOpen(false);
                      setUploadSuccessSummary(null);
                    }}
                    className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl transition-colors cursor-pointer"
                  >
                    Done
                  </button>
                </div>
              </div>
            )}

            {/* Upload Error */}
            {uploadError && (
              <div className="bg-rose-950/50 border border-rose-500/40 p-3 rounded-2xl text-xs text-rose-300 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* Chunk Text Inspection Popup Modal */}
      {/* ───────────────────────────────────────────────────────────── */}
      {selectedChunk && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-2xl w-full p-5 md:p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-indigo-400" />
                <div>
                  <h3 className="text-sm font-bold text-slate-100">
                    Source Chunk Inspector • Page {selectedChunk.page_number}
                  </h3>
                  <span className="text-[11px] text-slate-400">
                    Content Type:{' '}
                    <span className="text-indigo-300 font-semibold uppercase">
                      {selectedChunk.content_type || selectedChunk.collection_type || 'content'}
                    </span>{' '}
                    | Doc: {selectedChunk.document_name || selectedChunk.document_id || 'Current Document'}
                    {selectedChunk.year ? ` • Year ${selectedChunk.year}` : ''}
                  </span>
                </div>
              </div>
              <button
                onClick={() => setSelectedChunk(null)}
                className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-xl transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="bg-slate-950 border border-slate-800/80 rounded-2xl p-4 max-h-96 overflow-y-auto font-mono text-xs md:text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
              <HighlightText
                text={selectedChunk.text || 'No raw chunk text available in payload.'}
                highlight={selectedChunk.highlightText || selectedChunk.text?.slice(0, 100)}
                searchQuery=""
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => copyToClipboard(selectedChunk.text || '', 'modal-chunk')}
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
                onClick={() => setSelectedChunk(null)}
                className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl transition-colors cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
