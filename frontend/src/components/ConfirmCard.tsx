import type { PendingInterrupt } from '../types';

interface ConfirmCardProps {
  interrupt: PendingInterrupt;
  onResume: (decision: 'approved' | 'declined') => void;
}

export function ConfirmCard({ interrupt, onResume }: ConfirmCardProps) {
  return (
    <div className="confirm-card">
      <span className="confirm-card__symbol">✦</span>
      <div className="confirm-card__content">
        <p className="confirm-card__reason">{interrupt.reason}</p>
        <div className="confirm-card__actions">
          <button
            className="confirm-card__approve"
            onClick={() => onResume('approved')}
          >
            Continue
          </button>
          <button
            className="confirm-card__decline"
            onClick={() => onResume('declined')}
          >
            Not now
          </button>
        </div>
      </div>
    </div>
  );
}
