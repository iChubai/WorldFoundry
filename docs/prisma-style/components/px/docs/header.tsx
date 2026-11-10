'use client';

import { Menu, X } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';

import { BrandLogo } from '@/components/px/brand';
import { InkPill } from '@/components/px/buttons';
import { useDocsNav } from '@/components/px/docs/nav-context';
import { GitHubIcon } from '@/components/px/icons';
import { LanguageSwitch, type LanguageLink } from '@/components/px/language-switch';
import { SearchButton } from '@/components/px/search-button';
import { ThemeToggle } from '@/components/px/theme-toggle';
import { siteLinks } from '@/lib/px-site';

export type DocsHeaderLabels = {
  docs: string;
  search: string;
  community: string;
  language: string;
  openMenu: string;
  closeMenu: string;
  nav: string;
};

export type DocsHeaderNavLink = {
  label: string;
  href: string;
  external?: boolean;
};

/**
 * Docs navbar. Docked it is a translucent strip on the paper ground; after the
 * first scroll it morphs into a floating glass pill without changing height.
 */
export function DocsHeader({
  locale,
  docsHref,
  section,
  labels,
  languageLinks,
  navLinks,
}: {
  locale: 'en' | 'zh';
  docsHref: string;
  section?: string;
  labels: DocsHeaderLabels;
  languageLinks: LanguageLink[];
  navLinks: DocsHeaderNavLink[];
}) {
  const [floating, setFloating] = useState(false);
  const { open, setOpen } = useDocsNav();

  useEffect(() => {
    const onScroll = () => setFloating(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header className="px-docs-header" data-floating={floating ? 'true' : 'false'}>
      <div className="px-docs-header-panel">
        <div className="px-docs-header-start">
          <BrandLogo href={locale === 'zh' ? '/zh/docs' : '/'} />
          <span className="px-crumb-sep" aria-hidden="true">
            /
          </span>
          <Link className="px-crumb" href={docsHref}>
            {labels.docs}
          </Link>
          {section ? (
            <>
              <span className="px-crumb-sep px-hide-sm" aria-hidden="true">
                /
              </span>
              <span className="px-crumb px-crumb-section px-hide-sm">{section}</span>
            </>
          ) : null}
        </div>

        <div className="px-docs-header-end">
          <nav className="px-docs-header-nav" aria-label={labels.nav}>
            {navLinks.map((item) =>
              item.external ? (
                <a key={item.href} href={item.href} target="_blank" rel="noreferrer">
                  {item.label}
                </a>
              ) : (
                <Link key={item.href} href={item.href}>
                  {item.label}
                </Link>
              ),
            )}
          </nav>
          <SearchButton variant="pill" label={labels.search} className="px-hide-md" />
          <a
            className="px-icon-btn px-hide-md"
            href={siteLinks.github}
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub"
          >
            <GitHubIcon />
          </a>
          <LanguageSwitch links={languageLinks} label={labels.language} className="px-hide-md" />
          <ThemeToggle locale={locale} className="px-hide-md" />
          <InkPill href={siteLinks.slack} className="px-hide-lg">
            {labels.community}
          </InkPill>
          <SearchButton variant="icon" label={labels.search} className="px-show-md" />
          <button
            type="button"
            className="px-icon-btn px-show-md"
            aria-label={open ? labels.closeMenu : labels.openMenu}
            aria-expanded={open}
            aria-controls="px-sidebar"
            onClick={() => setOpen(!open)}
          >
            {open ? (
              <X aria-hidden="true" size={19} strokeWidth={1.8} />
            ) : (
              <Menu aria-hidden="true" size={19} strokeWidth={1.8} />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
