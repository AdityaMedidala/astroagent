import { useCallback, useRef, useState } from 'react';
import type { BirthDetails, ChatMessage, IntentLabel, PendingInterrupt, ToolEntry } from '../types';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// ── SSE event shapes from the backend ────────────────────────────────────────
interface TokenEvent     { type: 'token';     content: string; }
interface IntentEvent    { type: 'intent';    value: IntentLabel; }
interface ToolStart      { type: 'tool_start'; name: string; args: Record<string, unknown>; }
interface ToolEnd        { type: 'tool_end';  name: string; result: unknown; }
interface InterruptEvent { type: 'interrupt'; reason: string; thread_id: string; }
interface DoneEvent      { type: 'done'; }
type SSEEvent = TokenEvent | IntentEvent | ToolStart | ToolEnd | InterruptEvent | DoneEvent;

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

// ── Shared SSE stream consumer ────────────────────────────────────────────────
// Module-level so both sendMessage and resume can call it without duplicating
// the parsing logic.  State setters from useState are stable references.
async function consumeSSEStream(
  resp: Response,
  assistantId: string,
  setMessages: (fn: (prev: ChatMessage[]) => ChatMessage[]) => void,
  setPendingInterrupt: (v: PendingInterrupt | null) => void,
): Promise<void> {
  const reader = resp.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';
  const toolCounters: Record<string, number> = {};

  // Patch the assistant message identified by assistantId
  const patch = (updater: (m: ChatMessage) => ChatMessage) => {
    setMessages(msgs => msgs.map(m => (m.id === assistantId ? updater(m) : m)));
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
          patch(m => ({ ...m, content: m.content + event.content }));
          break;

        case 'intent':
          patch(m => ({ ...m, intent: event.value }));
          break;

        case 'tool_start': {
          const count = (toolCounters[event.name] ?? 0) + 1;
          toolCounters[event.name] = count;
          const entry: ToolEntry = {
            id: `${event.name}-${count}`,
            name: event.name,
            args: event.args,
            status: 'running',
          };
          patch(m => ({ ...m, tools: [...(m.tools ?? []), entry] }));
          break;
        }

        case 'tool_end':
          patch(m => {
            const tools = [...(m.tools ?? [])];
            const idx = tools
              .map((t, i) => ({ t, i }))
              .filter(({ t }) => t.name === event.name && t.status === 'running')
              .at(-1)?.i;
            if (idx !== undefined) {
              tools[idx] = { ...tools[idx], status: 'done', result: event.result };
            }
            return { ...m, tools };
          });
          break;

        case 'interrupt':
          // Drop the blank placeholder and surface the ConfirmCard instead
          setMessages(msgs => msgs.filter(m => m.id !== assistantId));
          setPendingInterrupt({ reason: event.reason, threadId: event.thread_id });
          break outer;

        case 'done':
          patch(m => ({ ...m, streaming: false }));
          break outer;
      }
    }
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────
export function useChatStream(birthDetails: BirthDetails | null, threadId: string) {
  const [messages, setMessages]                   = useState<ChatMessage[]>([]);
  const [streaming, setStreaming]                 = useState(false);
  const [error, setError]                         = useState<string | null>(null);
  const [pendingInterrupt, setPendingInterrupt]   = useState<PendingInterrupt | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // ── sendMessage ─────────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (text: string) => {
    if (streaming) return;

    const userMsg: ChatMessage = { id: uid(), role: 'user', content: text };
    const history = [...messages, userMsg];
    const assistantId = uid();

    setMessages(prev => [
      ...prev,
      userMsg,
      { id: assistantId, role: 'assistant', content: '', streaming: true, tools: [] },
    ]);
    setStreaming(true);
    setError(null);

    const apiMessages = history.map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content,
    }));

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

      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      await consumeSSEStream(resp, assistantId, setMessages, setPendingInterrupt);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const msg = err instanceof Error ? err.message : 'Connection error';
      setError(msg);
      setMessages(msgs =>
        msgs.map(m => (m.id === assistantId ? { ...m, streaming: false } : m))
      );
    } finally {
      setStreaming(false);
    }
  }, [messages, streaming, birthDetails, threadId]);

  // ── resume ──────────────────────────────────────────────────────────────────
  // Called by ConfirmCard when the user clicks Continue or Not now.
  const resume = useCallback(async (decision: 'approved' | 'declined') => {
    if (!pendingInterrupt) return;
    const resumeThreadId = pendingInterrupt.threadId;

    // Clear the card immediately so a second click can't double-fire
    setPendingInterrupt(null);

    const assistantId = uid();
    setMessages(prev => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', streaming: true, tools: [] },
    ]);
    setStreaming(true);
    setError(null);

    abortRef.current = new AbortController();

    try {
      const resp = await fetch(`${API_BASE}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: resumeThreadId, decision }),
        signal: abortRef.current.signal,
      });

      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      await consumeSSEStream(resp, assistantId, setMessages, setPendingInterrupt);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const msg = err instanceof Error ? err.message : 'Connection error';
      setError(msg);
      setMessages(msgs =>
        msgs.map(m => (m.id === assistantId ? { ...m, streaming: false } : m))
      );
    } finally {
      setStreaming(false);
    }
  }, [pendingInterrupt]);

  // ── retry / clearHistory ────────────────────────────────────────────────────
  const retry = useCallback(() => {
    const lastUser = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUser) return;
    setMessages(msgs => msgs.filter(m => m.id !== msgs.at(-1)?.id));
    setError(null);
    sendMessage(lastUser.content);
  }, [messages, sendMessage]);

  const clearHistory = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setError(null);
    setStreaming(false);
    setPendingInterrupt(null);
  }, []);

  return {
    messages,
    streaming,
    error,
    pendingInterrupt,
    sendMessage,
    resume,
    retry,
    clearHistory,
    setMessages,
  };
}
