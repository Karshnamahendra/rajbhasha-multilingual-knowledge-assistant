import React, { useState, useEffect, useRef } from 'react';
import { UploadedDocument } from '../types';
import { ChatMessage, SourceItem, OllamaStatusInfo, UploadSummary } from './chatTypes';
import { Header } from './Header';
import { DocumentSelector } from './DocumentSelector';
import { ChatMessages } from './ChatMessages';
import { ChatInput } from './ChatInput';
import { UploadModal } from './UploadModal';
import { SourceInspector } from './SourceInspector';

/**
 * Page container: owns all state and API calls, and composes the UI pieces.
 * Visual changes belong in the child components, not here.
 */
export const MainChatbot: React.FC = () => {
  const conversationIdRef = useRef(`conversation-${crypto.randomUUID()}`);
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  
  // Selected documents for the query. Empty = search all documents.
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  // Small screens: the document sidebar opens as a drawer
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);
  
  // Document category tab: 'magazines' | 'reports'
  const [activeCategoryTab, setActiveCategoryTab] = useState<'magazines' | 'reports'>('magazines');

  // Upload modal state
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [uploadCategory, setUploadCategory] = useState<'magazine' | 'report'>('magazine');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgressStep, setUploadProgressStep] = useState<number>(0);
  const [uploadStatusText, setUploadStatusText] = useState<string>('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccessSummary, setUploadSuccessSummary] = useState<UploadSummary | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Chat Query state
  const [inputQuery, setInputQuery] = useState<string>('');
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Active Source Chunk Modal Inspection State
  const [selectedChunk, setSelectedChunk] = useState<SourceItem | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      // Shown as the welcome screen with suggested questions (ChatMessages), not as a chat bubble
      text: 'नमस्ते! मैं राजभाषा पत्रिकाओं और रिपोर्टों से आपके सवालों के जवाब देता हूँ।',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  // Ollama status
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatusInfo>({
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
        setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
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
      // Make follow-up questions target the document just indexed. An empty
      // selection means “search all documents”, which can otherwise surface an
      // older report with a similar table label.
      setSelectedDocIds([richDoc.id]);

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

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    sendQuery(inputQuery);
  };

  /** Ask a question: from the input box, or from a suggested-question chip. */
  const sendQuery = async (text: string) => {
    if (!text.trim() || isQuerying) return;

    const userQueryText = text.trim();
    setSidebarOpen(false);

    // Selected documents → backend scope. Drop ids of documents that no longer exist.
    const scopeIds = selectedDocIds.filter((id) => documents.some((d) => d.id === id));
    const scopeTypes = new Set(
      scopeIds.map((id) => documents.find((d) => d.id === id)?.document_type || 'magazine')
    );
    // Only narrow by type when every selected file has the same type
    const docTypeFilter = scopeTypes.size === 1 ? [...scopeTypes][0] : undefined;
    const scopeLabel =
      scopeIds.length === 0
        ? undefined
        : scopeIds.length === 1
        ? documents.find((d) => d.id === scopeIds[0])?.fileName || scopeIds[0]
        : `${scopeIds.length} files`;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: userQueryText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      scopeLabel
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
          document_ids: scopeIds,
          document_id: scopeIds.length === 1 ? scopeIds[0] : '',
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
        detectedScript: data.detected_script,
        // Comparison table + bar graph data (rendered by ChatMessages)
        chartData: data.chart_data ?? data.chartData ?? null,
        comparison: Array.isArray(data.comparison) ? data.comparison : null,
        periods: Array.isArray(data.periods) ? data.periods : null
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: 'जवाब नहीं मिल सका। कृपया जाँचें कि Python backend (port 8000) और Ollama चल रहे हैं, फिर दोबारा पूछें।',
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

  // "सभी दस्तावेज़" or "2 फ़ाइलें", shown under the input box
  const liveSelected = selectedDocIds.filter((id) => documents.some((d) => d.id === id));
  const scopeText =
    liveSelected.length === 0
      ? 'सभी दस्तावेज़'
      : liveSelected.length === 1
      ? (documents.find((d) => d.id === liveSelected[0])?.fileName || '1 फ़ाइल').replace(/\.(pdf|docx?)$/i, '')
      : `${liveSelected.length} फ़ाइलें`;

  const sidebar = (
    <DocumentSelector
      documents={documents}
      selectedIds={selectedDocIds}
      onSelectionChange={setSelectedDocIds}
      activeTab={activeCategoryTab}
      onTabChange={setActiveCategoryTab}
      onUploadClick={() => {
        setIsUploadModalOpen(true);
        setUploadError(null);
        setUploadSuccessSummary(null);
      }}
      onDelete={handleDeleteDocument}
      onClose={() => setSidebarOpen(false)}
    />
  );

  return (
    // One screen tall: top bar, then sidebar | chat. Only the chat and the file list scroll.
    <div className="h-full flex flex-col">
      <Header ollamaStatus={ollamaStatus} onMenuClick={() => setSidebarOpen(true)} />

      <div className="flex-1 min-h-0 flex">
        {/* Sidebar: always visible on md+ */}
        <aside className="hidden md:flex w-72 lg:w-80 shrink-0 flex-col border-r border-slate-800 bg-slate-900/40">
          {sidebar}
        </aside>

        {/* Sidebar: drawer on small screens */}
        {sidebarOpen && (
          <div className="md:hidden fixed inset-0 z-40 flex">
            <div className="w-[85%] max-w-sm h-full bg-slate-900 border-r border-slate-800 shadow-2xl">{sidebar}</div>
            <button
              type="button"
              className="flex-1 bg-black/50"
              aria-label="बंद करें"
              onClick={() => setSidebarOpen(false)}
            />
          </div>
        )}

        {/* Chat */}
        <section className="flex-1 min-w-0 flex flex-col">
          <ChatMessages
            messages={messages}
            isQuerying={isQuerying}
            copiedId={copiedId}
            onCopy={copyToClipboard}
            onSourceClick={setSelectedChunk}
            onSuggestion={sendQuery}
            scopeLabel={scopeText}
          />
          <ChatInput
            value={inputQuery}
            onChange={setInputQuery}
            onSubmit={handleSendMessage}
            isQuerying={isQuerying}
            scopeLabel={scopeText}
          />
        </section>
      </div>

      <UploadModal
        isOpen={isUploadModalOpen}
        isUploading={isUploading}
        category={uploadCategory}
        progressStep={uploadProgressStep}
        statusText={uploadStatusText}
        error={uploadError}
        successSummary={uploadSuccessSummary}
        fileInputRef={fileInputRef}
        onStartUpload={handleStartUpload}
        onFilesSelected={handleFileUpload}
        onClose={() => {
          if (!isUploading) {
            setIsUploadModalOpen(false);
            setUploadError(null);
            setUploadSuccessSummary(null);
          }
        }}
        onDone={() => {
          setIsUploadModalOpen(false);
          setUploadSuccessSummary(null);
        }}
      />

      <SourceInspector
        chunk={selectedChunk}
        copiedId={copiedId}
        onCopy={copyToClipboard}
        onClose={() => setSelectedChunk(null)}
      />
    </div>
  );
};
