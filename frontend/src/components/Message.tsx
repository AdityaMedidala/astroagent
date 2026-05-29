import type { ChatMessage } from '../types';
import { ToolTrace } from './ToolTrace';

const INTENT_LABELS: Record<string, string> = {
  chart_request:   '✦ Natal chart',
  daily_horoscope: '☽ Daily transits',
  freeform:        '✧ Astrology query',
};

interface MessageProps {
  message: ChatMessage;
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`message message--${isUser ? 'user' : 'assistant'}`}>
      {/* Intent badge — assistant only, for labelled intents */}
      {!isUser && message.intent && INTENT_LABELS[message.intent] && (
        <span className="intent-badge">{INTENT_LABELS[message.intent]}</span>
      )}

      {/* Tool trace — assistant only */}
      {!isUser && message.tools && message.tools.length > 0 && (
        <ToolTrace tools={message.tools} />
      )}

      {/* Text bubble — always render, even if empty while streaming starts */}
      {(message.content || message.streaming) && (
        <div className={`message__bubble${message.streaming ? ' message__bubble--streaming' : ''}`}>
          {message.content}
        </div>
      )}
    </div>
  );
}
