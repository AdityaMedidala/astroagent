import { useEffect, useRef, type KeyboardEvent, type ChangeEvent } from 'react';
import type { ChatMessage } from '../types';
import { Message } from './Message';

const STARTER_PROMPTS = [
  'What does it mean to have Sun in Scorpio?',
  "What's happening in the sky today?",
  'Compute my chart: 14 March 1879, 11:30 AM, Ulm Germany',
];

interface ChatViewProps {
  messages: ChatMessage[];
  streaming: boolean;
  error: string | null;
  onSend: (text: string) => void;
  onRetry: () => void;
  draft: string;
  onDraftChange: (v: string) => void;
}

export function ChatView({
  messages,
  streaming,
  error,
  onSend,
  onRetry,
  draft,
  onDraftChange,
}: ChatViewProps) {
  const bottomRef  = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-grow textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [draft]);

  function submit() {
    const text = draft.trim();
    if (!text || streaming) return;
    onDraftChange('');
    onSend(text);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function handleChange(e: ChangeEvent<HTMLTextAreaElement>) {
    onDraftChange(e.target.value);
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-view">
      {isEmpty ? (
        <div className="empty-state">
          <div className="empty-state__symbol">✦</div>
          <h1 className="empty-state__title">Aradhana</h1>
          <p className="empty-state__body">
            A daily companion for self-reflection through astrology.
            Ask about your natal chart, today's planetary sky, or the symbolism
            behind signs and houses.
          </p>
          <div className="empty-state__prompts">
            {STARTER_PROMPTS.map(p => (
              <button
                key={p}
                className="prompt-chip"
                onClick={() => { onDraftChange(p); textareaRef.current?.focus(); }}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="message-list">
          {messages.map(m => <Message key={m.id} message={m} />)}
          <div ref={bottomRef} />
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="error-banner">
          <span>⚠ {error}</span>
          <button className="error-banner__retry" onClick={onRetry}>
            Retry
          </button>
        </div>
      )}

      {/* Input area */}
      <div className="input-area">
        <div className="input-area__row">
          <textarea
            ref={textareaRef}
            className="input-area__textarea"
            rows={1}
            placeholder="Ask about your chart, today's sky, or any astrological question…"
            value={draft}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={streaming}
          />
          <button
            className="input-area__send"
            onClick={submit}
            disabled={!draft.trim() || streaming}
            aria-label="Send message"
          >
            {streaming ? '⏸' : '↑'}
          </button>
        </div>
        <p className="input-area__hint">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}
