'use client';

import { createContext, type ReactNode, use, useMemo, useState } from 'react';

type DocsNavState = {
  open: boolean;
  setOpen: (open: boolean) => void;
};

const DocsNavContext = createContext<DocsNavState>({ open: false, setOpen: () => {} });

/** Shares the mobile sidebar drawer state between the header button and the sidebar. */
export function DocsNavProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const value = useMemo(() => ({ open, setOpen }), [open]);
  return <DocsNavContext value={value}>{children}</DocsNavContext>;
}

export function useDocsNav() {
  return use(DocsNavContext);
}
