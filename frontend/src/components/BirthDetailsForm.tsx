import { useState, type FormEvent } from 'react';
import type { BirthDetails } from '../types';

interface BirthDetailsFormProps {
  initial: BirthDetails | null;
  onSave: (details: BirthDetails) => void;
  onClose: () => void;
}

interface FormErrors {
  date?: string;
  time?: string;
  place?: string;
}

export function BirthDetailsForm({ initial, onSave, onClose }: BirthDetailsFormProps) {
  const [date, setDate]           = useState(initial?.date ?? '');
  const [time, setTime]           = useState(initial?.time ?? '');
  const [timeKnown, setTimeKnown] = useState(initial?.timeKnown ?? true);
  const [place, setPlace]         = useState(initial?.place ?? '');
  const [errors, setErrors]       = useState<FormErrors>({});

  function validate(): FormErrors {
    const errs: FormErrors = {};
    if (!date) {
      errs.date = 'Birth date is required.';
    } else if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      errs.date = 'Use YYYY-MM-DD format (e.g. 1990-06-15).';
    }
    if (timeKnown && time && !/^\d{2}:\d{2}$/.test(time)) {
      errs.time = 'Use HH:MM format (e.g. 14:30).';
    }
    if (!place.trim()) {
      errs.place = 'Birth place is required.';
    }
    return errs;
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      return;
    }
    onSave({
      date,
      time: timeKnown ? time : '',
      timeKnown,
      place: place.trim(),
    });
  }

  // Close on overlay click
  function handleOverlayClick(e: React.MouseEvent<HTMLDivElement>) {
    if (e.target === e.currentTarget) onClose();
  }

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <div className="modal__header">
          <h2 className="modal__title" id="modal-title">Birth Details</h2>
          <p className="modal__subtitle">
            Your birth data lets Aradhana compute a personalised natal chart and
            overlay transits with your houses.
          </p>
        </div>

        <form onSubmit={handleSubmit} noValidate>
          {/* Date */}
          <div className="form-group">
            <label className="form-label" htmlFor="bd-date">
              Date of birth
            </label>
            <input
              id="bd-date"
              type="date"
              className={`form-input${errors.date ? ' form-input--error' : ''}`}
              value={date}
              onChange={e => { setDate(e.target.value); setErrors(p => ({ ...p, date: undefined })); }}
              placeholder="YYYY-MM-DD"
            />
            {errors.date && <p className="form-error">{errors.date}</p>}
          </div>

          {/* Time */}
          <div className="form-group">
            <label className="form-label" htmlFor="bd-time">
              Time of birth
              <span className="form-label__optional">(optional)</span>
            </label>
            <input
              id="bd-time"
              type="time"
              className={`form-input${errors.time ? ' form-input--error' : ''}`}
              value={time}
              disabled={!timeKnown}
              onChange={e => { setTime(e.target.value); setErrors(p => ({ ...p, time: undefined })); }}
              placeholder="HH:MM"
            />
            {errors.time && <p className="form-error">{errors.time}</p>}
            <div className="form-checkbox-row">
              <input
                id="bd-time-unknown"
                type="checkbox"
                checked={!timeKnown}
                onChange={e => {
                  setTimeKnown(!e.target.checked);
                  if (e.target.checked) setTime('');
                }}
              />
              <label htmlFor="bd-time-unknown">Birth time unknown</label>
            </div>
          </div>

          {/* Place */}
          <div className="form-group">
            <label className="form-label" htmlFor="bd-place">
              Birth place
            </label>
            <input
              id="bd-place"
              type="text"
              className={`form-input${errors.place ? ' form-input--error' : ''}`}
              value={place}
              onChange={e => { setPlace(e.target.value); setErrors(p => ({ ...p, place: undefined })); }}
              placeholder="e.g. Paris, France"
              autoComplete="off"
            />
            {errors.place && <p className="form-error">{errors.place}</p>}
          </div>

          <div className="modal__actions">
            <button type="button" className="btn btn--ghost" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn--primary">
              Save details
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
