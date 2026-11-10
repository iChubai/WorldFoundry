import { AnnouncementBootstrap } from '@/components/px/announcement-bar';
import { SiteProvider } from '@/components/px/site-provider';
import { ThemeBootstrap } from '@/components/theme-bootstrap';
import { withBasePath } from '@/lib/site-path';
import type { Metadata } from 'next';
import './globals.css';

const iconPath = withBasePath('/favicon.svg') ?? '/favicon.svg';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3002'),
  title: {
    default: 'WorldFoundry',
    template: '%s | WorldFoundry',
  },
  description:
    'Open-source infrastructure for world intelligence: discover, run, inspect, and evaluate video, 3D/4D, interactive-world, and embodied models under shared contracts.',
  icons: {
    icon: [{ url: iconPath, type: 'image/svg+xml' }],
    shortcut: iconPath,
  },
};

export default function RootLayout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <ThemeBootstrap />
        <AnnouncementBootstrap />
        <SiteProvider>{children}</SiteProvider>
      </body>
    </html>
  );
}
