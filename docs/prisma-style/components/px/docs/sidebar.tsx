'use client';

import { ChevronRight } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { type ReactNode, useEffect, useRef, useState } from 'react';

import { DocsSidebarIcon } from '@/components/docs-sidebar-icon';
import { useDocsNav } from '@/components/px/docs/nav-context';
import type { SidebarNavEntry, SidebarNavGroup, SidebarNavPage } from '@/lib/docs-sidebar';

export type DocsSidebarLabels = {
  sidebar: string;
  navGroups: Record<string, string>;
  expandApiList: string;
  collapseApiList: string;
  expandMetricsList: string;
  collapseMetricsList: string;
  expandArchitectureList: string;
  collapseArchitectureList: string;
};

type TreeEntry = Extract<SidebarNavEntry, { hub: SidebarNavPage }>;

/** Catalog hubs stay marked while one of their detail pages is open. */
const ANCESTOR_HUBS = ['/evaluation/benchmark-hub', '/guides/supported-models'];

function itemState(item: SidebarNavPage, currentUrl: string) {
  if (item.link.url === currentUrl) return 'active';
  if (
    ANCESTOR_HUBS.some((hub) => item.link.url.endsWith(hub)) &&
    currentUrl.startsWith(`${item.link.url}/`)
  ) {
    return 'ancestor';
  }
  return undefined;
}

function SideLink({ item, currentUrl }: { item: SidebarNavPage; currentUrl: string }) {
  const state = itemState(item, currentUrl);

  return (
    <Link
      href={item.link.url}
      className="px-side-item"
      data-depth={item.depth}
      data-active={state === 'active' ? 'true' : undefined}
      data-ancestor={state === 'ancestor' ? 'true' : undefined}
      aria-current={state === 'active' ? 'page' : undefined}
    >
      {item.depth === 0 ? <DocsSidebarIcon url={item.link.url} active={state === 'active'} /> : null}
      <span className="px-side-label">{item.link.label}</span>
    </Link>
  );
}

function treeLabels(entry: TreeEntry, labels: DocsSidebarLabels) {
  if (entry.type === 'api-tree') return [labels.expandApiList, labels.collapseApiList];
  if (entry.type === 'metrics-tree') return [labels.expandMetricsList, labels.collapseMetricsList];
  return [labels.expandArchitectureList, labels.collapseArchitectureList];
}

function SideTree({
  entry,
  currentUrl,
  labels,
}: {
  entry: TreeEntry;
  currentUrl: string;
  labels: DocsSidebarLabels;
}) {
  const childActive = entry.items.some(
    (child) => child.type === 'page' && child.link.url === currentUrl,
  );
  const hubActive = entry.hub.link.url === currentUrl;
  // The API overview stays collapsed on its own page; other hubs open with their page.
  const [open, setOpen] = useState(childActive || (hubActive && entry.type !== 'api-tree'));
  const [expandLabel, collapseLabel] = treeLabels(entry, labels);

  useEffect(() => {
    if (childActive) setOpen(true);
  }, [childActive]);

  return (
    <div className="px-side-tree" data-open={open ? 'true' : 'false'}>
      <div className="px-side-tree-row">
        <SideLink item={entry.hub} currentUrl={currentUrl} />
        <button
          type="button"
          className="px-side-toggle"
          aria-expanded={open}
          aria-label={open ? collapseLabel : expandLabel}
          onClick={() => setOpen(!open)}
        >
          <ChevronRight aria-hidden="true" size={15} strokeWidth={1.9} />
        </button>
      </div>
      {open ? (
        <ul className="px-side-sublist">
          {entry.items.map((child) =>
            child.type === 'divider' ? (
              <li className="px-side-subheading" key={`divider-${child.label}`}>
                {child.label}
              </li>
            ) : (
              <li key={child.link.url}>
                <SideLink item={child} currentUrl={currentUrl} />
              </li>
            ),
          )}
        </ul>
      ) : null}
    </div>
  );
}

function entryKey(entry: SidebarNavEntry) {
  if (entry.type === 'page') return entry.link.url;
  if (entry.type === 'divider') return `divider-${entry.label}`;
  return `${entry.type}-${entry.hub.link.url}`;
}

/** Docs navigation: sticky column on desktop, drawer on small screens. */
export function DocsSidebar({
  groups,
  currentUrl,
  labels,
  drawerExtras,
}: {
  groups: SidebarNavGroup[];
  currentUrl: string;
  labels: DocsSidebarLabels;
  drawerExtras?: ReactNode;
}) {
  const { open, setOpen } = useDocsNav();
  const pathname = usePathname();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setOpen(false);
  }, [pathname, setOpen]);

  useEffect(() => {
    document.documentElement.toggleAttribute('data-px-nav-open', open);
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, setOpen]);

  // Keep the current page in view when the list is longer than the column.
  useEffect(() => {
    const box = scrollRef.current;
    const active = box?.querySelector<HTMLElement>('[data-active="true"]');
    if (!box || !active) return;
    const top = active.offsetTop;
    if (top < box.scrollTop || top > box.scrollTop + box.clientHeight - 48) {
      box.scrollTop = Math.max(0, top - box.clientHeight / 3);
    }
  }, [currentUrl]);

  return (
    <>
      <div
        className="px-sidebar-backdrop"
        data-open={open ? 'true' : 'false'}
        aria-hidden="true"
        onClick={() => setOpen(false)}
      />
      <aside
        id="px-sidebar"
        className="px-sidebar"
        data-open={open ? 'true' : 'false'}
        aria-label={labels.sidebar}
      >
        <div className="px-sidebar-scroll" ref={scrollRef}>
          <nav className="px-side-nav">
            {groups.map((group) => (
              <div className="px-side-group" key={group.id}>
                <p className="px-side-heading">{labels.navGroups[group.id] ?? group.id}</p>
                <ul className="px-side-list">
                  {group.items.map((entry) => (
                    <li key={entryKey(entry)}>
                      {entry.type === 'page' ? (
                        <SideLink item={entry} currentUrl={currentUrl} />
                      ) : entry.type === 'divider' ? (
                        <span className="px-side-subheading">{entry.label}</span>
                      ) : (
                        <SideTree entry={entry} currentUrl={currentUrl} labels={labels} />
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </nav>
          {drawerExtras ? <div className="px-sidebar-drawer-extras">{drawerExtras}</div> : null}
        </div>
      </aside>
    </>
  );
}
