'use client';

import { Check, ChevronDown, Copy, ExternalLink } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

export type PageActionLabels = {
  copy: string;
  copied: string;
  open: string;
  markdown: string;
  markdownHint: string;
  chatgpt: string;
  chatgptHint: string;
  claude: string;
  claudeHint: string;
  edit: string;
  editHint: string;
};

/** "Copy Markdown" plus an "Open" menu that hands the page to Markdown viewers and assistants. */
export function DocsPageActions({
  markdownUrl,
  editUrl,
  labels,
}: {
  markdownUrl: string;
  editUrl: string;
  labels: PageActionLabels;
}) {
  const [copied, setCopied] = useState(false);
  const [absoluteUrl, setAbsoluteUrl] = useState(markdownUrl);
  const menuRef = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    setAbsoluteUrl(new URL(markdownUrl, window.location.origin).href);
  }, [markdownUrl]);

  useEffect(() => {
    const onPointer = (event: PointerEvent) => {
      const menu = menuRef.current;
      if (menu?.open && !menu.contains(event.target as Node)) menu.open = false;
    };
    document.addEventListener('pointerdown', onPointer);
    return () => document.removeEventListener('pointerdown', onPointer);
  }, []);

  async function copyMarkdown() {
    try {
      const response = await fetch(markdownUrl);
      if (!response.ok) throw new Error(String(response.status));
      await navigator.clipboard.writeText(await response.text());
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  const prompt = encodeURIComponent(`Read ${absoluteUrl}, I want to ask questions about it.`);
  const items = [
    { label: labels.markdown, hint: labels.markdownHint, href: markdownUrl },
    {
      label: labels.chatgpt,
      hint: labels.chatgptHint,
      href: `https://chatgpt.com/?hints=search&q=${prompt}`,
    },
    { label: labels.claude, hint: labels.claudeHint, href: `https://claude.ai/new?q=${prompt}` },
    { label: labels.edit, hint: labels.editHint, href: editUrl },
  ];

  return (
    <div className="px-page-actions">
      <button type="button" className="px-action-btn" onClick={copyMarkdown} aria-live="polite">
        {copied ? (
          <Check aria-hidden="true" size={15} strokeWidth={2} />
        ) : (
          <Copy aria-hidden="true" size={15} strokeWidth={1.8} />
        )}
        <span>{copied ? labels.copied : labels.copy}</span>
      </button>
      <details className="px-action-menu" ref={menuRef}>
        <summary className="px-action-btn">
          <span>{labels.open}</span>
          <ChevronDown aria-hidden="true" size={15} strokeWidth={1.8} />
        </summary>
        <div className="px-action-menu-panel" role="menu">
          {items.map((item) => (
            <a
              key={item.label}
              role="menuitem"
              href={item.href}
              target="_blank"
              rel="noreferrer"
              className="px-action-menu-item"
              onClick={() => {
                if (menuRef.current) menuRef.current.open = false;
              }}
            >
              <span className="px-action-menu-text">
                <strong>{item.label}</strong>
                <small>{item.hint}</small>
              </span>
              <ExternalLink aria-hidden="true" size={13} strokeWidth={1.8} />
            </a>
          ))}
        </div>
      </details>
    </div>
  );
}
