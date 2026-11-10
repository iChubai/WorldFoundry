import type { ReactNode } from 'react';

import { cx } from '@/components/px/cx';

/** Icon in a rounded tile with the cyan / yellow / red underglow. */
export function IconTile({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span aria-hidden="true" className={cx('px-icon-tile', className)}>
      <span className="px-icon-tile-glow" />
      <span className="px-icon-tile-icon">{children}</span>
    </span>
  );
}

/** Small colored dot + label above a card title. */
export function Kicker({
  children,
  tone = 'cyan',
}: {
  children: ReactNode;
  tone?: 'cyan' | 'yellow' | 'red';
}) {
  return (
    <p className="px-kicker" data-tone={tone}>
      <span className="px-kicker-dot" aria-hidden="true" />
      {children}
    </p>
  );
}

/** Pill connector between stacked product cards. */
export function Connector({ label, caption }: { label: string; caption?: string }) {
  return (
    <div className="px-connector" aria-hidden="true">
      <span className="px-connector-line" />
      <span className="px-connector-pill">{label}</span>
      {caption ? <span className="px-connector-caption">{caption}</span> : null}
      <span className="px-connector-line" />
    </div>
  );
}
