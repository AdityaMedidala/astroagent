import { useCallback, useRef, useState } from 'react';
import type { BirthDetails, ChatMessage, IntentLabel, ToolEntry } from '../types';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// ── SSE event shapes from the backend ────────────────────────────────────────
interface TokenEvent   { type: 'token';      content: string; }
interface IntentEvent  { type: 'intent';     value: IntentLabel; }
interface ToolStart    { type: 'tool_start'; name: string; args: Record<string, unknown>; }
interface ToolEnd      { type: 'tool_end';   name: string; result: unknown; }
interface DoneEvent    { type: 'done'; }
type SSEEvent = TokenEvent | IntentEvent | ToolStart | ToolEnd | DoneEvent;

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function useChatStream(birthDetails: BirthDetails | null, threadId: string) {
  const [messages, setMessages]   = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (text: string) => {
    if (streaming) return;

    const userMsg: ChatMessage = { id: uid(), role: 'user', content: text };

    // Snapshot of all messages to send to the API (user + prior history)
    const history = [...messages, userMsg];

    // Append user message and a blank assistant placeholder
    const assistantId = uid();
    setMessages(prev => [
      ...prev,
      userMsg,
      { id: assistantId, role: 'assistant', content: '', streaming: true, tools: [] },
    ]);
    setStreaming(true);
    setError(null);

    // Build the request body — convert ChatMessage[] to backend format
    const apiMessages = history.map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content,
    }));

    // Include birth details if available so the backend can use natal chart context
    const body: Record<string, unknown> = { messages: apiMessages, thread_id: threadId };
    if (birthDetails) {
      body.birth_details = {
        date: birthDetails.date,
        time: birthDetails.timeKnown ? birthDetails.time : null,
        place: birthDetails.place,
      };
    }

    abortRef.current = new AbortController();

    try {
      const resp = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortRef.current.signal,
      });

      if (!resp.ok) {
        throw new Error(`Server returned ${resp.status}`);
      }

      const reader = resp.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      // Track tool name→count so we can build stable IDs
      const toolCounters: Record<string, number> = {};

      const patchAssistant = (updater: (prev: ChatMessage) => ChatMessage) => {
        setMessages(msgs =>
          msgs.map(m => (m.id === assistantId ? updater(m) : m))
        );
      };

      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;

          const payload = trimmed.slice(5).trim();
          let event: SSEEvent;
          try {
            event = JSON.parse(payload) as SSEEvent;
          } catch {
            continue;
          }

          switch (event.type) {
            case 'token':
              patchAssistant(m => ({ ...m, content: m.content + event.content }));
              break;

            case 'intent':
              patchAssistant(m => ({ ...m, intent: event.value }));
              break;

            case 'tool_start': {
              const count = (toolCounters[event.name] ?? 0) + 1;
              toolCounters[event.name] = count;
              const toolId = `${event.name}-${count}`;
              const entry: ToolEntry = {
                id: toolId,
                name: event.name,
                args: event.args,
                status: 'running',
              };
              patchAssistant(m => ({
                ...m,
                tools: [...(m.tools ?? []), entry],
              }));
              break;
            }

            case 'tool_end': {
              // Mark the most-recent running tool with this name as done
              patchAssistant(m => {
                const tools = [...(m.tools ?? [])];
                const idx = tools.map((t, i) => ({ t, i }))
                  .filter(({ t }) => t.name === event.name && t.status === 'running')
                  .at(-1)?.i;
                if (idx !== undefined) {
                  tools[idx] = { ...tools[idx], status: 'done', result: event.result };
                }
                return { ...m, tools };
              });
              break;
            }

            case 'done':
              patchAssistant(m => ({ ...m, streaming: false }));
              break outer;
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const msg = err instanceof Error ? err.message : 'Connection error';
      setError(msg);
      // Remove streaming flag from the assistant message so cursor disappears
      setMessages(msgs =>
        msgs.map(m => (m.id === assistantId ? { ...m, streaming: false } : m))
      );
    } finally {
      setStreaming(false);
    }
  }, [messages, streaming, birthDetails, threadId]);

  const retry = useCallback(() => {
    const lastUser = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUser) return;
    // Remove the failed assistant reply and re-send
    setMessages(msgs => msgs.filter(m => m.id !== msgs.at(-1)?.id));
    setError(null);
    sendMessage(lastUser.content);
  }, [messages, sendMessage]);

  const clearHistory = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setError(null);
    setStreaming(false);
  }, []);

  return { messages, streaming, error, sendMessage, retry, clearHistory, setMessages };
}
