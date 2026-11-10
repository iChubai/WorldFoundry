'use client';

import { AnchorProvider, ScrollProvider, TOCItem, useActiveAnchors } from 'fumadocs-core/toc';
import type { TOCItemType } from 'fumadocs-core/toc';
import { ChevronDown, TextAlignStart } from 'lucide-react';
import { type ReactNode, useLayoutEffect, useRef, useState } from 'react';

function depthClass(depth: number, minDepth: number) {
  const level = Math.min(Math.max(depth - minDepth, 0), 3);
  return `px-toc-link px-toc-depth-${level}`;
}

function TocLinks({ items }: { items: TOCItemType[] }) {
  const listRef = useRef<HTMLDivElement>(null);
  const active = useActiveAnchors();
  const [thumb, setThumb] = useState<{ top: number; height: number } | null>(null);
  const minDepth = items.reduce((min, item) => Math.min(min, item.depth), 6);

  useLayoutEffect(() => {
    const list = listRef.current;
    if (!list || active.length === 0) {
      setThumb(null);
      return;
    }
    let top = Infinity;
    let bottom = 0;
    for (const id of active) {
      const link = list.querySelector<HTMLElement>(`a[href="#${CSS.escape(id)}"]`);
      if (!link) continue;
      top = Math.min(top, link.offsetTop);
      bottom = Math.max(bottom, link.offsetTop + link.offsetHeight);
    }
    setThumb(Number.isFinite(top) ? { top, height: bottom - top } : null);
  }, [active]);

  return (
    <div className="px-toc-list" ref={listRef}>
      {thumb ? (
        <span
          className="px-toc-thumb"
          aria-hidden="true"
          style={{ top: thumb.top, height: thumb.height }}
        />
      ) : null}
      {items.map((item) => (
        <TOCItem key={item.url} href={item.url} className={depthClass(item.depth, minDepth)}>
          {item.title}
        </TOCItem>
      ))}
    </div>
  );
}

/** "On this page" rail with scroll spy and a spectrum position thumb. */
export function DocsToc({
  items,
  title,
  children,
}: {
  items: TOCItemType[];
  title: string;
  children?: ReactNode;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  return (
    <AnchorProvider toc={items}>
      <nav className="px-toc" aria-label={title}>
        {items.length > 0 ? (
          <>
            <p className="px-toc-title">
              <TextAlignStart aria-hidden="true" size={15} strokeWidth={1.9} />
              {title}
            </p>
            <div className="px-toc-scroll" ref={scrollRef}>
              <ScrollProvider containerRef={scrollRef}>
                <TocLinks items={items} />
              </ScrollProvider>
            </div>
          </>
        ) : null}
        {children}
      </nav>
    </AnchorProvider>
  );
}

/** Collapsible TOC shown inside the sheet when the right rail is hidden. */
export function DocsInlineToc({ items, title }: { items: TOCItemType[]; title: string }) {
  if (items.length === 0) return null;

  return (
    <AnchorProvider toc={items}>
      <details className="px-inline-toc">
        <summary>
          <TextAlignStart aria-hidden="true" size={15} strokeWidth={1.9} />
          <span>{title}</span>
          <ChevronDown aria-hidden="true" size={15} strokeWidth={1.9} className="px-inline-toc-chevron" />
        </summary>
        <TocLinks items={items} />
      </details>
    </AnchorProvider>
  );
}
