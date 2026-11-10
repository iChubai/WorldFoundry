'use client';

import SearchDialog from '@/components/search';
import { RootProvider } from 'fumadocs-ui/provider/next';
import type { ReactNode } from 'react';

// next-themes renders an inline script; the blocking theme bootstrap lives in
// the server layout, so this one is marked as a data block.
const THEME_SCRIPT_PROPS = { type: 'application/json' } as const;

/** Fumadocs root provider with the shared static search dialog. */
export function SiteProvider({ children }: { children: ReactNode }) {
  return (
    <RootProvider
      search={{ SearchDialog, preload: false }}
      theme={{ scriptProps: THEME_SCRIPT_PROPS }}
    >
      {children}
    </RootProvider>
  );
}
