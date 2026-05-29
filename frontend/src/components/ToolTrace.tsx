import { useState } from 'react';
import type { ToolEntry } from '../types';

interface ToolRowProps {
  entry: ToolEntry;
}

function ToolRow({ entry }: ToolRowProps) {
  const [open, setOpen] = useState(false);

  const isDone    = entry.status === 'done';
  const isRunning = entry.status === 'running';

  const resultObj = entry.result as Record<string, unknown> | undefined;
  const truncated = resultObj?.truncated === true;

  return (
    <div className="tool-row">
      <div
        className="tool-row__header"
        onClick={() => { if (isDone) setOpen(o => !o); }}
        role={isDone ? 'button' : undefined}
        aria-expanded={isDone ? open : undefined}
      >
        <span className="tool-row__status">
          {isRunning && <span className="spinner" />}
          {isDone    && <span className="status-icon--done">✓</span>}
        </span>

        <span className="tool-row__name">{entry.name}</span>

        {isDone && (
          <span className={`tool-row__chevron${open ? ' tool-row__chevron--open' : ''}`}>
            ▼
          </span>
        )}
      </div>

      {isDone && open && (
        <div className="tool-row__body">
          <div>
            <div className="tool-row__section-label">Args</div>
            <pre className="tool-row__json">
              {JSON.stringify(entry.args, null, 2)}
            </pre>
          </div>

          <div>
            <div className="tool-row__section-label">Result</div>
            <pre className="tool-row__json">
              {truncated
                ? (resultObj?.preview as string) ?? ''
                : JSON.stringify(entry.result, null, 2)}
            </pre>
          </div>

          {truncated && (
            <p className="tool-row__truncated-note">
              Result truncated ({resultObj?.size_chars as number} chars total)
            </p>
          )}
        </div>
      )}
    </div>
  );
}

interface ToolTraceProps {
  tools: ToolEntry[];
}

export function ToolTrace({ tools }: ToolTraceProps) {
  if (tools.length === 0) return null;

  return (
    <div className="tool-trace">
      {tools.map(t => (
        <ToolRow key={t.id} entry={t} />
      ))}
    </div>
  );
}
