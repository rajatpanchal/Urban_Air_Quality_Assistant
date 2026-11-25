import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const app = express();
app.use(express.json({ limit: '1mb' }));

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured in environment secrets.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function formatGeminiError(err: unknown): { status: number; message: string } {
  const raw = err instanceof Error ? err.message : String(err);
  if (raw.includes('GEMINI_API_KEY is not configured')) {
    return {
      status: 500,
      message: 'API key is not set yet. Please configure GEMINI_API_KEY in Settings > Secrets.',
    };
  }
  if (raw.includes('PERMISSION_DENIED') || raw.includes('403') || raw.includes('API_KEY_INVALID')) {
    return {
      status: 403,
      message: 'Gemini API authentication failed. Please verify your GEMINI_API_KEY in Settings > Secrets.',
    };
  }
  if (raw.includes('RESOURCE_EXHAUSTED') || raw.includes('429')) {
    return {
      status: 429,
      message: 'API quota limit reached for now. Please wait a moment and try again.',
    };
  }
  return {
    status: 500,
    message: raw || 'Unable to generate a response right now. Please try again.',
  };
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

app.post('/api/ask', async (req, res) => {
  try {
    const { message, language, cityContext, household, history } = req.body ?? {};
    if (!message || typeof message !== 'string' || !message.trim()) {
      res.status(400).json({ error: 'Please enter a question before sending.' });
      return;
    }

    const ai = getGeminiClient();
    const langInstruction =
      language === 'Hindi'
        ? 'Respond strictly in clear, everyday conversational Hindi (Devanagari script) suitable for families in Delhi NCR.'
        : language === 'Hinglish'
          ? 'Respond in friendly, natural Hinglish (Hindi written in Roman script mixed with simple English terms like AQI, PM2.5, N95, Metro, HEPA) suitable for Delhi NCR residents.'
          : 'Respond in warm, concise, practical English tailored to Delhi NCR families.';

    const systemInstruction = `You are BreatheWise NCR, a practical, calm clean-air daily advisor for families living in Delhi NCR.
${langInstruction}

Current household & locality context:
- Household: ${household?.name || 'Family'} (${household?.adults ?? 2} adults, ${household?.children ?? 1} child, ${household?.elders ?? 1} elder${household?.hasAsthma ? ', respiratory sensitivity present' : ''})
- Selected locality: ${cityContext?.name || 'Hauz Khas · Delhi'}
- Current AQI: ${cityContext?.aqi ?? 248} US AQI (${cityContext?.category || 'Poor'})
- PM2.5: ${cityContext?.pm25 ?? 142} µg/m³ · PM10: ${cityContext?.pm10 ?? 236} µg/m³
- Best outdoor window today: ${cityContext?.bestWindow || '1:00 PM – 4:00 PM'}

Guidelines:
- Give concrete, actionable advice grounded in Delhi NCR realities (morning temperature inversion, afternoon mixing, N95 masks, wet mopping vs dry jhadu sweeping, Metro vs solo car trips, HEPA timing, school outdoor timing, elders' walk timing).
- Keep answers concise (3 to 5 short bullet points or 2 short paragraphs, under 160 words) with clear reasons ("Why").
- Never provide medical diagnoses; for acute symptoms like wheezing or chest tightness, advise consulting a doctor.`;

    const priorHistory = Array.isArray(history)
      ? history
          .slice(-6)
          .filter(
            (h): h is { role: string; text: string } =>
              typeof h === 'object' &&
              h !== null &&
              'role' in h &&
              'text' in h &&
              typeof (h as { role: unknown }).role === 'string' &&
              typeof (h as { text: unknown }).text === 'string',
          )
          .map((h) => ({
            role: h.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: h.text }],
          }))
      : [];

    const contents = [
      ...priorHistory,
      { role: 'user', parts: [{ text: message.trim() }] },
    ];

    const response = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents,
      config: {
        systemInstruction,
      },
    });

    const reply = response.text?.trim();
    if (!reply) {
      res.status(502).json({ error: 'Received an empty response from the advisor. Please try again.' });
      return;
    }

    res.json({ reply });
  } catch (err) {
    const formatted = formatGeminiError(err);
    res.status(formatted.status).json({ error: formatted.message });
  }
});

app.post('/api/plan/regenerate', async (req, res) => {
  try {
    const { cityContext, household } = req.body ?? {};
    const ai = getGeminiClient();

    const prompt = `Create a practical 4-slot daily clean-air breath plan for a household in ${cityContext?.name || 'Hauz Khas · Delhi'} where today's AQI is ${cityContext?.aqi ?? 248} (PM2.5: ${cityContext?.pm25 ?? 142} µg/m³), and the cleanest afternoon window is ${cityContext?.bestWindow || '1:00 PM – 4:00 PM'}.
Household members: ${household?.adults ?? 2} adults, ${household?.children ?? 1} children, ${household?.elders ?? 1} elders${household?.hasAsthma ? ', someone has asthma/respiratory sensitivity' : ''}.
Provide realistic NCR-specific guidance for Morning (7–10 AM), Afternoon Window (${cityContext?.bestWindow || '1–4 PM'}), Evening Commute (5–8 PM), and Night Sleep (9–11 PM).`;

    const response = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              id: { type: Type.STRING },
              timeRange: { type: Type.STRING },
              title: { type: Type.STRING },
              audience: { type: Type.STRING },
              maskAdvice: { type: Type.STRING },
              action: { type: Type.STRING },
              why: { type: Type.STRING },
            },
            required: ['id', 'timeRange', 'title', 'audience', 'maskAdvice', 'action', 'why'],
          },
        },
      },
    });

    const jsonStr = (response.text ?? '[]').trim();
    const slots = JSON.parse(jsonStr);
    res.json({ slots });
  } catch (err) {
    const formatted = formatGeminiError(err);
    res.status(formatted.status).json({ error: formatted.message });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`BreatheWise NCR running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
