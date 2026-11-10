'use client';

import { useSearchContext } from 'fumadocs-ui/contexts/search';
import { Search } from 'lucide-react';
import { useEffect, useState } from 'react';

import { cx } from '@/components/px/cx';

/** Opens the shared static search dialog. */
export function SearchButton({
  variant = 'pill',
  label = 'Search',
  className,
}: {
  variant?: 'pill' | 'icon';
  label?: string;
  className?: string;
}) {
  const { enabled, setOpenSearch } = useSearchContext();
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    setIsMac(/Mac|iPhone|iPad/.test(navigator.platform));
  }, []);

  if (!enabled) return null;

  if (variant === 'icon') {
    return (
      <button
        type="button"
        className={cx('px-icon-btn', className)}
        onClick={() => setOpenSearch(true)}
        aria-label={label}
      >
        <Search aria-hidden="true" size={18} strokeWidth={1.8} />
      </button>
    );
  }

  return (
    <button
      type="button"
      className={cx('px-search-pill', className)}
      onClick={() => setOpenSearch(true)}
      aria-label={label}
    >
      <Search aria-hidden="true" size={15} strokeWidth={1.9} />
      <span className="px-search-pill-label">{label}</span>
      <span className="px-kbds" aria-hidden="true">
        <kbd>{isMac ? '⌘' : 'Ctrl'}</kbd>
        <kbd>K</kbd>
      </span>
    </button>
  );
}
