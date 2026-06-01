import { useCallback, useEffect, useState } from 'react';
import './App.css';
import type { BirthDetails, ChatMessage } from './types';
import { useChatStream } from './hooks/useChatStream';
import { ChatView } from './components/ChatView';
import { BirthDetailsForm } from './components/BirthDetailsForm';

const STORAGE_KEYS = {
  messages:     'aradhana:messages',
  birthDetails: 'aradhana:birthDetails',
  threadId:     'aradhana:threadId',
} as const;

function loadFromStorage<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function saveToStorage(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // quota exceeded — silently skip
  }
}

export default function App() {
  // ── Persisted birth details ───────────────────────────────────────────────
  const [birthDetails, setBirthDetails] = useState<BirthDetails | null>(
    () => loadFromStorage<BirthDetails>(STORAGE_KEYS.birthDetails)
  );

  // ── Stable thread ID — persisted across reloads, reset on clear ───────────
  const [threadId, setThreadId] = useState<string>(() => {
    const stored = loadFromStorage<string>(STORAGE_KEYS.threadId);
    if (stored) return stored;
    const fresh = crypto.randomUUID();
    saveToStorage(STORAGE_KEYS.threadId, fresh);
    return fresh;
  });

  // ── Modal visibility ──────────────────────────────────────────────────────
  const [showForm, setShowForm] = useState(false);

  // ── Chat draft (controlled at App level so clearing works across renders) ─
  const [draft, setDraft] = useState('');

  // ── Chat stream hook ──────────────────────────────────────────────────────
  const { messages, streaming, error, pendingInterrupt, sendMessage, resume, retry, clearHistory, setMessages } =
    useChatStream(birthDetails, threadId);

  // ── Re-hydrate message history from localStorage on first mount ───────────
  useEffect(() => {
    const saved = loadFromStorage<ChatMessage[]>(STORAGE_KEYS.messages);
    if (saved && saved.length > 0) {
      // Clear any lingering streaming flags from a previous interrupted session
      setMessages(saved.map(m => ({ ...m, streaming: false })));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Persist messages whenever they change ─────────────────────────────────
  useEffect(() => {
    if (messages.length > 0) {
      saveToStorage(STORAGE_KEYS.messages, messages);
    }
  }, [messages]);

  // ── Persist birth details ─────────────────────────────────────────────────
  useEffect(() => {
    saveToStorage(STORAGE_KEYS.birthDetails, birthDetails);
  }, [birthDetails]);

  const handleSaveBirthDetails = useCallback((details: BirthDetails) => {
    setBirthDetails(details);
    setShowForm(false);
  }, []);

  const handleClearBirthDetails = useCallback(() => {
    setBirthDetails(null);
    localStorage.removeItem(STORAGE_KEYS.birthDetails);
  }, []);

  const handleClear = useCallback(() => {
    clearHistory();
    localStorage.removeItem(STORAGE_KEYS.messages);
    const fresh = crypto.randomUUID();
    saveToStorage(STORAGE_KEYS.threadId, fresh);
    setThreadId(fresh);
  }, [clearHistory]);

  // ── Birth details summary string for the chip ─────────────────────────────
  const birthSummary = birthDetails
    ? `${birthDetails.date} · ${birthDetails.place}`
    : null;

  return (
    <div className="app">
      <div className="app-background"></div>
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header__brand">
          <span className="header__title">Aradhana</span>
          <span className="header__subtitle">A daily spiritual companion</span>
        </div>

        <div className="header__actions">
          {birthSummary ? (
            <>
              <button
                className="birth-chip"
                onClick={() => setShowForm(true)}
                title="Edit birth details"
              >
                ✦ {birthSummary}
              </button>
              <button
                className="btn-icon btn-icon--small"
                onClick={handleClearBirthDetails}
                title="Clear birth details"
                aria-label="Clear birth details"
              >
                ✕
              </button>
            </>
          ) : (
            <button
              className="birth-chip"
              onClick={() => setShowForm(true)}
              title="Add birth details"
            >
              + Add birth details
            </button>
          )}

          <button
            className="btn-icon"
            onClick={handleClear}
            title="Clear conversation"
            aria-label="Clear conversation"
          >
            ⟳
          </button>
        </div>
      </header>

      {/* ── Chat ──────────────────────────────────────────────────────────── */}
      <ChatView
        messages={messages}
        streaming={streaming}
        error={error}
        pendingInterrupt={pendingInterrupt}
        onSend={sendMessage}
        onRetry={retry}
        onResume={resume}
        draft={draft}
        onDraftChange={setDraft}
      />

      {/* ── Birth details modal ────────────────────────────────────────────── */}
      {showForm && (
        <BirthDetailsForm
          initial={birthDetails}
          onSave={handleSaveBirthDetails}
          onClose={() => setShowForm(false)}
        />
      )}
    </div>
  );
}
