import type { CSSProperties } from 'react';

import { cx } from '@/components/px/cx';

type PrismRayProps = {
  className?: string;
  /** Degrees; the brand angle rises to the right. */
  angle?: number;
  intensity?: 'whisper' | 'structural' | 'hero';
  /** Which ends fade out. */
  mask?: 'both' | 'start' | 'end' | 'none';
  style?: CSSProperties;
};

/** Decorative triple-band ray (cyan / yellow / red) crossing a surface. */
export function PrismRay({
  className,
  angle = -16,
  intensity = 'structural',
  mask = 'both',
  style,
}: PrismRayProps) {
  return (
    <div
      aria-hidden="true"
      className={cx('px-ray', `px-ray-${intensity}`, mask !== 'none' && `px-ray-mask-${mask}`, className)}
      style={{ rotate: `${angle}deg`, ...style }}
    />
  );
}

/** Paper grain overlay; fills its positioned parent. */
export function Grain({ className }: { className?: string }) {
  return <div aria-hidden="true" className={cx('px-grain', className)} />;
}
