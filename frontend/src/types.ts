export type IntentLabel =
  | 'chart_request'
  | 'daily_horoscope'
  | 'freeform'
  | 'offtopic'
  | 'adversarial';

export type ToolStatus = 'running' | 'done';

export interface ToolEntry {
  id: string;        // unique per message (tool name + index)
  name: string;
  args: Record<string, unknown>;
  status: ToolStatus;
  result?: unknown;  // present when status === 'done'
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  streaming?: boolean;
  intent?: IntentLabel;
  tools?: ToolEntry[];
}

export interface BirthDetails {
  date: string;       // YYYY-MM-DD
  time: string;       // HH:MM or '' when unknown
  timeKnown: boolean;
  place: string;
}

export interface PendingInterrupt {
  reason: string;    // warm one-sentence framing from the backend classifier
  threadId: string;  // needed to POST /resume
}
