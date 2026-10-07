import express from 'express';
import path from 'path';
import http from 'http';
import { createServer as createViteServer } from 'vite';

const PORT = 3000;
const PYTHON_PORT = 8000;
const PYTHON_URL = `http://127.0.0.1:${PYTHON_PORT}`;

async function startServer() {
  const app = express();

  // -------------------------------------------------------------
  // Dedicated Handlers for Fast JSON Responses
  // -------------------------------------------------------------
  app.get('/api/documents', async (_req, res) => {
    try {
      const response = await fetch(`${PYTHON_URL}/api/documents`);
      if (!response.ok) {
        return res.status(503).json({ success: false, error: 'FastAPI backend unavailable on port 8000.', documents: [] });
      }
      const data = await response.json();
      // Support both { success, documents } format and bare array
      const docs = Array.isArray(data) ? data : ((data as any).documents || []);
      return res.json({ success: true, documents: docs });
    } catch {
      return res.status(503).json({ success: false, error: 'FastAPI backend unavailable on port 8000.', documents: [] });
    }
  });

  app.get(['/api/health', '/api/ollama/status', '/api/status'], async (_req, res) => {
    try {
      const response = await fetch(`${PYTHON_URL}/api/health`);
      if (!response.ok) {
        return res.status(503).json({ status: 'unavailable', available: false, backend: 'unavailable', ollama: 'unknown' });
      }
      const data = await response.json();
      return res.json(data);
    } catch {
      return res.status(503).json({ status: 'unavailable', available: false, backend: 'unavailable', ollama: 'unknown' });
    }
  });

  // -------------------------------------------------------------
  // Transparent Proxy for Uploads & Chat (Preserves multipart streams)
  // -------------------------------------------------------------
  app.use('/api', (req, res) => {
    const cleanedHeaders = { ...req.headers };
    cleanedHeaders.host = `127.0.0.1:${PYTHON_PORT}`;

    const options: http.RequestOptions = {
      hostname: '127.0.0.1',
      port: PYTHON_PORT,
      path: req.originalUrl, // Uses the complete /api/... path
      method: req.method,
      headers: cleanedHeaders,
    };

    const proxyReq = http.request(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 500, proxyRes.headers);
      proxyRes.pipe(res, { end: true });
    });

    proxyReq.setTimeout(300000, () => {
      proxyReq.destroy();
      if (!res.headersSent) {
        res.status(504).json({ success: false, error: 'Request timed out.' });
      }
    });

    proxyReq.on('error', (err) => {
      console.error('[API Proxy Error]', err.message);
      if (!res.headersSent) {
        res.status(502).json({
          success: false,
          error: 'FastAPI backend unavailable on port 8000.',
        });
      }
    });

    req.pipe(proxyReq, { end: true });
  });

  // -------------------------------------------------------------
  // Vite Integration
  // -------------------------------------------------------------
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Server] Rajbhasha RAG Frontend Server running on http://localhost:${PORT}`);
    console.log(`[Proxy] API requests proxying to Python FastAPI server at ${PYTHON_URL}`);
  });
}

startServer().catch((err) => {
  console.error('[Server Startup Error]', err);
});
