import type { ReactNode } from 'react';

import { Grain, PrismRay } from '@/components/px/prism-ray';

/** Framed hero for secondary pages: label, title, description, optional media. */
export function SubpageHero({
  label,
  title,
  children,
  media,
}: {
  label: string;
  title: string;
  children?: ReactNode;
  media?: ReactNode;
}) {
  return (
    <section className="px-hero px-subhero">
      <div className="px-hero-frame">
        <div className="px-hero-bloom" aria-hidden="true" />
        <Grain />
        <PrismRay intensity="structural" mask="start" className="px-subhero-ray" />
        <div className="px-subhero-inner">
          {media ? <div className="px-subhero-media">{media}</div> : null}
          <div className="px-subhero-copy">
            <p className="px-kicker">
              <span className="px-kicker-dot" aria-hidden="true" />
              {label}
            </p>
            <h1>{title}</h1>
            {children}
          </div>
        </div>
      </div>
    </section>
  );
}

/** Card announcing a section with nothing published yet. */
export function ComingSoon({ title = 'Coming Soon', children }: { title?: string; children: ReactNode }) {
  return (
    <section className="px-section" aria-labelledby="px-coming-soon">
      <div className="px-container">
        <div className="px-soon">
          <div className="px-soon-art" aria-hidden="true">
            <span className="px-step-beam" style={{ '--px-step-index': 0 } as React.CSSProperties}>
              <span />
            </span>
          </div>
          <div className="px-soon-body">
            <h2 id="px-coming-soon">{title}</h2>
            <p>{children}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
