import fs from 'node:fs';
import path from 'node:path';

import { FUMADOCS_ROOT } from '@/lib/fumadocs-root';
import type { Locale } from '@/lib/i18n';

function resolveContentPath(pagePath: string) {
  const baseDir = path.join(FUMADOCS_ROOT, 'content/docs');
  const candidates = [
    path.join(baseDir, pagePath),
    path.join(baseDir, `${pagePath}.mdx`),
    path.join(baseDir, `${pagePath}.md`),
  ];

  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}

/** Last-modified date of a docs page source, preferring the localized file. */
export function getDocsLastUpdated(pagePath: string, locale: Locale) {
  const localizedSuffix = locale === 'en' ? '' : `.${locale}`;
  const localizedCandidates = [
    pagePath.replace(/\.mdx$/, `${localizedSuffix}.mdx`),
    pagePath.replace(/\.md$/, `${localizedSuffix}.md`),
    `${pagePath}${localizedSuffix}`,
  ];

  const filePath =
    localizedCandidates.map(resolveContentPath).find(Boolean) ?? resolveContentPath(pagePath);

  if (!filePath) return null;

  try {
    const date = new Date(fs.statSync(filePath).mtimeMs);

    return {
      date,
      iso: date.toISOString(),
      formatted: new Intl.DateTimeFormat(locale === 'zh' ? 'zh-CN' : 'en-US', {
        dateStyle: 'medium',
      }).format(date),
    };
  } catch {
    return null;
  }
}
