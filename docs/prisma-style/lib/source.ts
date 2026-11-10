import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { docs } from 'collections/server';
import { loader } from 'fumadocs-core/source';

import { FUMADOCS_ROOT } from '@/lib/fumadocs-root';
import { defaultLocale, i18n, isDefaultLocale, isLocale } from '@/lib/i18n';
import { docsImageRoute, docsRoute } from '@/lib/shared';
import { withBasePath } from '@/lib/site-path';

/** Docs source for this site; the MDX collection points at the shared content directory. */
export const source = loader({
  baseUrl: docsRoute,
  i18n,
  source: docs.toFumadocsSource(),
  plugins: [],
});

type DocsPage = (typeof source)['$inferPage'];

function getLocalizedSegments(page: DocsPage, leaf: string) {
  const segments = [...page.slugs, leaf];

  if (!isDefaultLocale(page.locale)) {
    segments.unshift(page.locale ?? defaultLocale);
  }

  return segments;
}

export function getPageImage(page: DocsPage) {
  const segments = getLocalizedSegments(page, 'image.png');
  const url = `${docsImageRoute}/${segments.join('/')}`;

  return { segments, url: withBasePath(url) ?? url };
}

/** Segments of the public markdown path `${page.url}.md`, e.g. `zh/docs/quickstart.md`. */
export function getPageMarkdownSlugs(page: DocsPage) {
  const segments = page.url.split('/').filter(Boolean);
  segments[segments.length - 1] += '.md';
  return segments;
}

export function getPageMarkdownUrl(page: DocsPage) {
  const url = `${page.url}.md`;
  return { url: withBasePath(url) ?? url };
}

/** Inverse of `getPageMarkdownSlugs`; `.mdx` leaves are accepted as aliases. */
export function resolveMarkdownSlug(slug: readonly string[]) {
  const [first, ...rest] = slug;
  const locale = isLocale(first) ? first : defaultLocale;
  const segments = isLocale(first) ? rest : [...slug];
  const last = segments.at(-1);
  if (!last || !/\.mdx?$/i.test(last)) return undefined;

  segments[segments.length - 1] = last.replace(/\.mdx?$/i, '');
  const [root, ...pageSlugs] = segments;
  return `/${root}` === docsRoute ? { locale, pageSlugs } : undefined;
}

export async function getPageSourceMarkdown(page: DocsPage) {
  const filePath = path.join(FUMADOCS_ROOT, 'content/docs', page.path);

  try {
    return await readFile(filePath, 'utf8');
  } catch {
    return getLLMText(page);
  }
}

export async function getLLMText(page: DocsPage) {
  const processed = await page.data.getText('processed');

  return `# ${page.data.title} (${page.url})

${processed}`;
}
