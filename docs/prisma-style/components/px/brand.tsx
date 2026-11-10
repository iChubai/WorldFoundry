import Link from 'next/link';

import { cx } from '@/components/px/cx';

/** The WorldFoundry mark: an ink globe crossed by the three spectrum bands. */
export function BrandMark({ className }: { className?: string }) {
  return <span aria-hidden="true" className={cx('px-mark', className)} />;
}

/** Mark plus wordmark, linking home. */
export function BrandLogo({
  href = '/',
  className,
  label = 'WorldFoundry home',
}: {
  href?: string;
  className?: string;
  label?: string;
}) {
  return (
    <Link href={href} className={cx('px-logo', className)} aria-label={label}>
      <BrandMark />
      <span className="px-logo-word">WorldFoundry</span>
    </Link>
  );
}
