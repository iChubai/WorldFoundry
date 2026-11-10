'use client';

import { X } from 'lucide-react';
import { useServerInsertedHTML } from 'next/navigation';
import { useRef, type ReactNode } from 'react';

import { cx } from '@/components/px/cx';
import { ANNOUNCEMENT_KEY } from '@/lib/px-site';

/**
 * Runs before the bar paints so a dismissed announcement never flashes.
 * The flag lives on <html>, which also drives the sticky offsets in CSS.
 */
const ANNOUNCEMENT_BOOTSTRAP = `(function(){try{if(localStorage.getItem('${ANNOUNCEMENT_KEY}')==='1'){document.documentElement.dataset.pxAnnouncement='hidden'}}catch(e){}})();`;

/**
 * Injects the announcement bootstrap into `<head>` during SSR only, like
 * ThemeBootstrap: an in-tree `<script>` is created again on client navigation,
 * never executes there, and trips React 19's script-tag warning.
 */
export function AnnouncementBootstrap() {
  const flushed = useRef(false);

  useServerInsertedHTML(() => {
    if (flushed.current) {
      return null;
    }
    flushed.current = true;
    return (
      <script
        id="px-announcement-bootstrap"
        dangerouslySetInnerHTML={{ __html: ANNOUNCEMENT_BOOTSTRAP }}
      />
    );
  });

  return null;
}

/** Full-width announcement strip with a spectrum hairline; dismissible. */
export function AnnouncementBar({
  tone = 'light',
  children,
  closeLabel = 'Dismiss',
}: {
  tone?: 'light' | 'dark';
  children: ReactNode;
  closeLabel?: string;
}) {
  return (
    <div className={cx('px-announcement', tone === 'dark' && 'px-announcement-dark')}>
      <div className="px-announcement-content">{children}</div>
      <button
        type="button"
        className="px-announcement-close"
        aria-label={closeLabel}
        onClick={() => {
          try {
            localStorage.setItem(ANNOUNCEMENT_KEY, '1');
          } catch {
            // Storage can be unavailable; hiding still applies to this page view.
          }
          document.documentElement.dataset.pxAnnouncement = 'hidden';
        }}
      >
        <X aria-hidden="true" size={15} strokeWidth={1.8} />
      </button>
    </div>
  );
}
