import Link from 'next/link';

import { BrandLogo } from '@/components/px/brand';
import { GitHubIcon, SlackIcon } from '@/components/px/icons';
import { ThemeToggle } from '@/components/px/theme-toggle';
import { footerColumns, siteLinks } from '@/lib/px-site';

/** Footer: brand block, link columns, and the giant striped wordmark. */
export function SiteFooter() {
  const year = new Date().getFullYear();

  return (
    <footer className="px-footer" aria-label="Site footer">
      <div className="px-footer-inner">
        <div className="px-footer-grid">
          <div className="px-footer-brand">
            <BrandLogo />
            <p className="px-footer-tagline">
              Unified world model inference and evaluation infrastructure.
            </p>
            <div className="px-footer-contacts">
              <a href={siteLinks.github} target="_blank" rel="noreferrer">
                <GitHubIcon size={16} />
                OpenEnvision/WorldFoundry
              </a>
              <a href={siteLinks.slack} target="_blank" rel="noreferrer">
                <SlackIcon size={15} />
                Join the Slack community
              </a>
            </div>
            <p className="px-footer-copy">
              Copyright © 2024–{year}{' '}
              <a href={siteLinks.organization} target="_blank" rel="noreferrer">
                OpenEnvision
              </a>
              . All rights reserved.
            </p>
            <ThemeToggle className="px-footer-theme" />
          </div>

          <div className="px-footer-columns">
            {footerColumns.map((column) => (
              <div key={column.title}>
                <h2>{column.title}</h2>
                <ul>
                  {column.links.map((link) => (
                    <li key={link.href}>
                      {link.external ? (
                        <a href={link.href} target="_blank" rel="noreferrer">
                          {link.label}
                        </a>
                      ) : (
                        <Link href={link.href}>{link.label}</Link>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="px-giant" aria-hidden="true">
          <span className="px-giant-ink">WorldFoundry</span>
          <span className="px-giant-color">WorldFoundry</span>
        </div>
      </div>
    </footer>
  );
}
