import Link from 'next/link';

import { cx } from '@/components/px/cx';

export type LanguageLink = {
  href: string;
  label: string;
  current?: boolean;
};

/** Segmented EN / 中文 switch. */
export function LanguageSwitch({
  links,
  label = 'Language',
  className,
}: {
  links: LanguageLink[];
  label?: string;
  className?: string;
}) {
  return (
    <nav className={cx('px-lang', className)} aria-label={label}>
      {links.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          aria-current={item.current ? 'true' : undefined}
          data-active={item.current ? 'true' : undefined}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}
