import Link from 'next/link';
import type { ReactNode } from 'react';

import { cx } from '@/components/px/cx';

type ButtonProps = {
  href: string;
  children: ReactNode;
  className?: string;
  size?: 'md' | 'lg';
  /** Open in a new tab; external URLs are detected automatically. */
  external?: boolean;
};

function isExternal(href: string, external?: boolean) {
  return external ?? /^https?:\/\//.test(href);
}

function Anchor({
  href,
  external,
  className,
  children,
}: {
  href: string;
  external?: boolean;
  className: string;
  children: ReactNode;
}) {
  if (isExternal(href, external)) {
    return (
      <a href={href} className={className} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  }
  return (
    <Link href={href} className={className}>
      {children}
    </Link>
  );
}

/** Primary CTA: ink pill over a drifting spectrum glow; the spectrum bursts through on hover. */
export function PrismButton({ href, children, className, size = 'md', external }: ButtonProps) {
  return (
    <Anchor
      href={href}
      external={external}
      className={cx('px-btn-prism', size === 'lg' && 'px-btn-lg', className)}
    >
      <span className="px-btn-glow" aria-hidden="true">
        <span />
      </span>
      <span className="px-btn-body">
        <span className="px-btn-label">{children}</span>
      </span>
    </Anchor>
  );
}

/** Secondary CTA: neutral outline that turns into a moving spectrum ring on hover. */
export function OutlineButton({ href, children, className, size = 'md', external }: ButtonProps) {
  return (
    <Anchor
      href={href}
      external={external}
      className={cx('px-btn-outline', 'px-spectrum-ring', size === 'lg' && 'px-btn-lg', className)}
    >
      <span className="px-btn-label">{children}</span>
    </Anchor>
  );
}

/** Compact ink pill for navigation bars. */
export function InkPill({ href, children, className, external }: Omit<ButtonProps, 'size'>) {
  return (
    <Anchor href={href} external={external} className={cx('px-ink-pill', className)}>
      {children}
    </Anchor>
  );
}

/** Text link with an arrow that picks up the spectrum on hover. */
export function ArrowLink({
  href,
  children,
  className,
  external,
}: Omit<ButtonProps, 'size'>) {
  return (
    <Anchor href={href} external={external} className={cx('px-arrow-link', className)}>
      <span>{children}</span>
      <svg aria-hidden="true" viewBox="0 0 16 16" width="14" height="14">
        <path
          d="M3 8h9.5M8.5 3.5 13 8l-4.5 4.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </Anchor>
  );
}
