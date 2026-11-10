import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

import { AnnouncementBar } from '@/components/px/announcement-bar';
import { SiteFooter } from '@/components/px/site-footer';
import { SiteHeader } from '@/components/px/site-header';
import { catalogCoverage } from '@/lib/catalog-coverage';

/** Marketing pages: announcement strip, morphing navbar, footer. */
export default function SiteLayout({ children }: LayoutProps<'/'>) {
  return (
    <div className="px-site">
      <AnnouncementBar tone="dark">
        <strong>
          {catalogCoverage.modelsTotal} model entries · {catalogCoverage.benchmarksTotal} browsable
          benchmarks
        </strong>
        <span className="px-announcement-muted">with explicit readiness signals.</span>
        <Link className="px-announcement-cta" href="/docs/guides/supported-models">
          <span className="px-announcement-cta-label">Explore model recipes</span>
          <ArrowRight aria-hidden="true" size={13} strokeWidth={2} />
        </Link>
      </AnnouncementBar>
      <SiteHeader />
      {children}
      <SiteFooter />
    </div>
  );
}
