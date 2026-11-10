import type { Metadata } from 'next';

import { ComingSoon, SubpageHero } from '@/components/px/subpage';

export const metadata: Metadata = {
  title: 'Blog',
  description: 'WorldFoundry updates, technical notes, release notes, and integration writeups.',
};

export default function BlogPage() {
  return (
    <main className="px-home">
      <SubpageHero label="Project notes" title="Blog">
        <p className="px-lead">
          Updates, technical notes, and release highlights from the WorldFoundry team.
        </p>
      </SubpageHero>
      <ComingSoon>Updates, technical notes, and release highlights will be posted here soon.</ComingSoon>
    </main>
  );
}
