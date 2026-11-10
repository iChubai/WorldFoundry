import {
  OPENENVISION_AWESOME_WORLD_MODELING,
  OPENENVISION_BLOGXIV_SITE,
  OPENENVISION_ORG,
  WORLDFOUNDRY_GITHUB_DISCUSSIONS,
  WORLDFOUNDRY_GITHUB_ISSUES,
  WORLDFOUNDRY_GITHUB_REPO,
  WORLDFOUNDRY_SLACK_INVITE,
} from '@/lib/site-links';

export type SiteNavId = 'home' | 'docs' | 'blog' | 'events' | 'community' | 'openenvision';

export type SiteLink = {
  id?: SiteNavId;
  label: string;
  href: string;
  external?: boolean;
};

/** Top-level navigation; the logo covers Home. */
export const siteNav: SiteLink[] = [
  { id: 'docs', label: 'Docs', href: '/docs' },
  { id: 'blog', label: 'Blog', href: '/blog' },
  { id: 'events', label: 'Events', href: '/events' },
  { id: 'community', label: 'Community', href: WORLDFOUNDRY_SLACK_INVITE, external: true },
  { id: 'openenvision', label: 'OpenEnvision', href: '/openenvision' },
];

export type FooterColumn = {
  title: string;
  links: SiteLink[];
};

export const footerColumns: FooterColumn[] = [
  {
    title: 'Documentation',
    links: [
      { label: 'Introduction', href: '/docs' },
      { label: 'Quickstart', href: '/docs/quickstart' },
      { label: 'Model catalog', href: '/docs/guides/supported-models' },
      { label: 'Benchmark Hub', href: '/docs/evaluation/benchmark-hub' },
      { label: 'API reference', href: '/docs/api-reference' },
    ],
  },
  {
    title: 'Product',
    links: [
      { label: 'Studio', href: '/docs/guides/studio' },
      { label: 'CLI', href: '/docs/reference/cli' },
      { label: 'TUI', href: '/docs/guides/tui' },
      { label: 'Run inference', href: '/docs/guides/inference' },
      { label: 'Evaluation', href: '/docs/evaluation' },
    ],
  },
  {
    title: 'Community',
    links: [
      { label: 'Slack', href: WORLDFOUNDRY_SLACK_INVITE, external: true },
      { label: 'GitHub Issues', href: WORLDFOUNDRY_GITHUB_ISSUES, external: true },
      { label: 'GitHub Discussions', href: WORLDFOUNDRY_GITHUB_DISCUSSIONS, external: true },
      { label: 'Contributing', href: '/docs/maintainers/contributing' },
    ],
  },
  {
    title: 'Ecosystem',
    links: [
      { label: 'OpenEnvision', href: '/openenvision' },
      { label: 'Blog', href: '/blog' },
      { label: 'Events', href: '/events' },
      { label: 'Awesome World Modeling', href: OPENENVISION_AWESOME_WORLD_MODELING, external: true },
      { label: 'BlogrXiv', href: OPENENVISION_BLOGXIV_SITE, external: true },
    ],
  },
];

export const siteLinks = {
  github: WORLDFOUNDRY_GITHUB_REPO,
  slack: WORLDFOUNDRY_SLACK_INVITE,
  organization: OPENENVISION_ORG,
};

/** Dismissal key for the announcement bar; bump it when the message changes. */
export const ANNOUNCEMENT_KEY = 'wf-px-announcement-catalog';
