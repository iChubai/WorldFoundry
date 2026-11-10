import type { Metadata } from 'next';

import { ComingSoon, SubpageHero } from '@/components/px/subpage';

export const metadata: Metadata = {
  title: 'Events',
  description: 'WorldFoundry meetings, demos, benchmark sprints, and release milestones.',
};

export default function EventsPage() {
  return (
    <main className="px-home">
      <SubpageHero label="Project calendar" title="Events">
        <p className="px-lead">Community meetups, demos, and project milestones.</p>
      </SubpageHero>
      <ComingSoon>
        Meetings, demos, benchmark sprints, and release milestones will be posted here soon.
      </ComingSoon>
    </main>
  );
}
