'use client';

import { Menu, X } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

import { BrandLogo } from '@/components/px/brand';
import { InkPill } from '@/components/px/buttons';
import { cx } from '@/components/px/cx';
import { GitHubIcon } from '@/components/px/icons';
import { SearchButton } from '@/components/px/search-button';
import { siteLinks, siteNav, type SiteNavId } from '@/lib/px-site';

function activeId(pathname: string): SiteNavId | undefined {
  if (pathname === '/') return 'home';
  return siteNav.find((item) => item.href !== '/' && pathname.startsWith(item.href))?.id;
}

/**
 * Marketing navbar. Docked it spans the hero frame; once the page scrolls it
 * morphs into a centred floating pill. The strip itself has zero height so it
 * overlays the first section.
 */
export function SiteHeader() {
  const pathname = usePathname();
  const [floating, setFloating] = useState(false);
  const [open, setOpen] = useState(false);
  const current = activeId(pathname);

  useEffect(() => {
    const onScroll = () => setFloating(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => setOpen(false), [pathname]);

  return (
    <header className="px-site-header" data-floating={floating ? 'true' : 'false'}>
      <div className="px-site-header-bar">
        <BrandLogo />
        <nav className="px-site-nav" aria-label="Main navigation">
          {siteNav.map((item) =>
            item.external ? (
              <a key={item.href} href={item.href} target="_blank" rel="noreferrer">
                {item.label}
              </a>
            ) : (
              <Link
                key={item.href}
                href={item.href}
                aria-current={current === item.id ? 'page' : undefined}
              >
                {item.label}
              </Link>
            ),
          )}
        </nav>
        <div className="px-site-header-tools">
          <SearchButton variant="icon" />
          <Link className="px-ghost-link" href="/zh/docs" lang="zh-CN">
            中文
          </Link>
          <a
            className="px-icon-btn"
            href={siteLinks.github}
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub"
          >
            <GitHubIcon />
          </a>
          <InkPill href="/docs/quickstart">Get started</InkPill>
        </div>
        <button
          type="button"
          className="px-icon-btn px-site-menu-btn"
          aria-label={open ? 'Close menu' : 'Open menu'}
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          {open ? (
            <X aria-hidden="true" size={19} strokeWidth={1.8} />
          ) : (
            <Menu aria-hidden="true" size={19} strokeWidth={1.8} />
          )}
        </button>
      </div>
      <div className={cx('px-site-sheet', open && 'is-open')} aria-hidden={!open}>
        <nav aria-label="Mobile navigation">
          {siteNav.map((item) =>
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
          <Link href="/zh/docs" lang="zh-CN">
            中文文档
          </Link>
          <a href={siteLinks.github} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
        <InkPill href="/docs/quickstart" className="px-site-sheet-cta">
          Get started
        </InkPill>
      </div>
    </header>
  );
}
