'use client';

import { Check, Copy } from 'lucide-react';
import { useState } from 'react';

/** Terminal-style command with a copy button. */
export function CopyCommand({ command, label }: { command: string; label: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="px-command" aria-label={label}>
      <div className="px-command-bar">
        <span className="px-window-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span>bash</span>
        <button type="button" onClick={copy} aria-live="polite">
          {copied ? (
            <Check aria-hidden="true" size={14} strokeWidth={2} />
          ) : (
            <Copy aria-hidden="true" size={14} strokeWidth={1.8} />
          )}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre>
        <code>
          <span className="px-command-prompt">$ </span>
          {command}
        </code>
      </pre>
    </div>
  );
}
