'use client';

import { Monitor, Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

import { cx } from '@/components/px/cx';

const OPTIONS = [
  { value: 'light', Icon: Sun, label: { en: 'Light', zh: '浅色' } },
  { value: 'dark', Icon: Moon, label: { en: 'Dark', zh: '深色' } },
  { value: 'system', Icon: Monitor, label: { en: 'System', zh: '跟随系统' } },
] as const;

/** Three-way light / dark / system switch. */
export function ThemeToggle({
  className,
  locale = 'en',
}: {
  className?: string;
  locale?: 'en' | 'zh';
}) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  const current = mounted ? (theme ?? 'system') : undefined;

  return (
    <div
      className={cx('px-theme', className)}
      role="radiogroup"
      aria-label={locale === 'zh' ? '主题' : 'Theme'}
    >
      {OPTIONS.map(({ value, Icon, label }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={current === value}
          aria-label={label[locale]}
          title={label[locale]}
          data-active={current === value ? 'true' : undefined}
          onClick={() => setTheme(value)}
        >
          <Icon aria-hidden="true" size={14} strokeWidth={1.9} />
        </button>
      ))}
    </div>
  );
}
