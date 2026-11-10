import { ArrowLeft, ArrowRight, ChevronRight } from 'lucide-react';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { createRelativeLink } from 'fumadocs-ui/mdx';

import { BenchmarkIdentityMark } from '@/components/benchmark-identity-mark';
import { DocsWelcomeHero } from '@/components/docs-welcome-hero';
import { getMDXComponents } from '@/components/mdx';
import { MetricQuickNav } from '@/components/metric-quick-nav';
import { ModelIdentityMark } from '@/components/model-identity-mark';
import { ModelPaperThumb } from '@/components/model-paper-thumb';
import { AnnouncementBar } from '@/components/px/announcement-bar';
import { cx } from '@/components/px/cx';
import { DocsHeader } from '@/components/px/docs/header';
import { DocsNavProvider } from '@/components/px/docs/nav-context';
import { DocsPageActions } from '@/components/px/docs/page-actions';
import { DocsSidebar } from '@/components/px/docs/sidebar';
import { DocsInlineToc, DocsToc } from '@/components/px/docs/toc';
import { LanguageSwitch } from '@/components/px/language-switch';
import { Grain, PrismRay } from '@/components/px/prism-ray';
import { ThemeToggle } from '@/components/px/theme-toggle';
import { getBenchmarkCatalogEntry } from '@/lib/benchmark-catalog';
import { enrichApiReferenceToc, isApiReferenceDocsPage } from '@/lib/docs-api-toc';
import { enrichBenchmarkToc } from '@/lib/docs-benchmark-toc';
import { getDocsBreadcrumbs } from '@/lib/docs-breadcrumb';
import { getDocsLastUpdated } from '@/lib/docs-last-updated';
import { docsLabels } from '@/lib/docs-navigation';
import { getDocsPagination } from '@/lib/docs-pagination';
import { getDocsRelatedLinks } from '@/lib/docs-related-links';
import { getDocsSidebarGroups, isSidebarGroupActive } from '@/lib/docs-sidebar';
import { defaultLocale, i18n, isLocale, localeNames, type Locale } from '@/lib/i18n';
import { resolveMetricQuickNavFromSlugs } from '@/lib/metric-quick-nav';
import { getModelPaperThumbSrc } from '@/lib/model-arch-diagrams';
import { getModelRecipeIndexEntry } from '@/lib/model-recipe-index';
import { pxDocsLabels } from '@/lib/px-labels';
import { siteLinks } from '@/lib/px-site';
import { getGithubContentUrls } from '@/lib/shared';
import { getPageMarkdownUrl, source } from '@/lib/source';

function normalizeLocale(locale: string): Locale {
  if (!isLocale(locale)) notFound();
  return locale;
}

function getPageUrl(slugs: string[], locale: Locale) {
  const page = source.getPage(slugs, locale);
  if (page) return page.url;

  const path = slugs.length > 0 ? `/${slugs.join('/')}` : '';
  return locale === defaultLocale ? `/docs${path}` : `/${locale}/docs${path}`;
}

function localizedHref(href: string, locale: Locale) {
  return locale === defaultLocale ? href : `/${locale}${href}`;
}

/** A docs page in the reading-sheet shell: paper ground, floating navbar, sidebar, sheet, TOC rail. */
export async function PxDocsPage({ slug, locale }: { slug: string[] | undefined; locale: string }) {
  const normalized = normalizeLocale(locale);
  const page = source.getPage(slug, normalized);
  if (!page) notFound();

  const t = docsLabels[normalized];
  const px = pxDocsLabels[normalized];
  const MDX = page.data.body;
  const markdownUrl = getPageMarkdownUrl(page).url;
  const github = getGithubContentUrls(page.path);
  const docsHref = getPageUrl([], normalized);
  const groups = getDocsSidebarGroups(normalized);
  // Detail pages (one model, one benchmark) belong to the group of their
  // closest hub, i.e. the longest sidebar URL that prefixes this page.
  const activeGroup =
    groups.find((group) => isSidebarGroupActive(group, page.url)) ??
    groups
      .flatMap((group) =>
        group.items.flatMap((item) =>
          item.type === 'page' && page.url.startsWith(`${item.link.url}/`)
            ? [{ group, length: item.link.url.length }]
            : [],
        ),
      )
      .sort((a, b) => b.length - a.length)[0]?.group;
  const apiHubPage = isApiReferenceDocsPage(page.slugs);

  const toc = enrichBenchmarkToc(
    page.slugs,
    enrichApiReferenceToc(page.slugs, page.data.toc ?? []),
    normalized,
  );
  const metricNav = resolveMetricQuickNavFromSlugs(page.slugs);
  const pagination = getDocsPagination(page.slugs, normalized);
  const relatedLinks = getDocsRelatedLinks(page.slugs, normalized);
  const breadcrumbs = getDocsBreadcrumbs(page.slugs, normalized, page.data.title);
  const benchmarkId =
    page.slugs[0] === 'evaluation' && page.slugs[1] === 'benchmark-hub' && page.slugs.length === 3
      ? page.slugs[2]
      : undefined;
  const benchmarkEntry = benchmarkId ? getBenchmarkCatalogEntry(benchmarkId) : undefined;
  const modelId =
    page.slugs[0] === 'guides' && page.slugs[1] === 'supported-models' && page.slugs.length === 3
      ? page.slugs[2]
      : undefined;
  const modelEntry = modelId ? getModelRecipeIndexEntry(modelId) : undefined;
  const paperThumb = modelId ? getModelPaperThumbSrc(modelId) : undefined;
  const lastUpdated = getDocsLastUpdated(page.path, normalized);
  const isWelcomePage = page.slugs.length === 0;
  const isModelRecipePage = Boolean(modelEntry);
  const showToc = !isWelcomePage && !isModelRecipePage && toc.length > 0;
  const showRail = !isModelRecipePage && !isWelcomePage && (showToc || Boolean(metricNav));

  const mdxComponents = getMDXComponents({ a: createRelativeLink(source, page) });

  const languageLinks = i18n.languages.map((item) => ({
    href: getPageUrl(page.slugs, item),
    label: item === defaultLocale ? 'EN' : localeNames[item],
    current: item === normalized,
  }));
  const navLinks = [
    { label: t.home, href: normalized === defaultLocale ? '/' : docsHref },
    { label: t.blog, href: '/blog' },
    { label: t.events, href: '/events' },
    { label: t.openEnvision, href: '/openenvision' },
  ];

  const actions = (
    <DocsPageActions
      markdownUrl={markdownUrl}
      editUrl={github.edit}
      labels={{
        copy: px.copy,
        copied: t.askAiCopied,
        open: px.open,
        markdown: px.markdown,
        markdownHint: px.markdownHint,
        chatgpt: px.chatgpt,
        chatgptHint: px.chatgptHint,
        claude: px.claude,
        claudeHint: px.claudeHint,
        edit: px.edit,
        editHint: px.editHint,
      }}
    />
  );

  const pager =
    pagination.prev || pagination.next ? (
      <nav className="px-pager" aria-label={`${t.previousPage} / ${t.nextPage}`}>
        {pagination.prev ? (
          <Link className="px-pager-card px-spectrum-ring" href={pagination.prev.url}>
            <span className="px-pager-dir">
              <ArrowLeft aria-hidden="true" size={14} strokeWidth={1.9} />
              {t.previousPage}
            </span>
            <span className="px-pager-title">{pagination.prev.label}</span>
          </Link>
        ) : (
          <span />
        )}
        {pagination.next ? (
          <Link className="px-pager-card px-pager-next px-spectrum-ring" href={pagination.next.url}>
            <span className="px-pager-dir">
              {t.nextPage}
              <ArrowRight aria-hidden="true" size={14} strokeWidth={1.9} />
            </span>
            <span className="px-pager-title">{pagination.next.label}</span>
          </Link>
        ) : null}
      </nav>
    ) : null;

  return (
    <DocsNavProvider>
      <div
        className={cx(
          'px-docs',
          isWelcomePage && 'px-docs-welcome pi-doc-shell-welcome',
          benchmarkId && 'pi-doc-shell-benchmark-detail',
          isModelRecipePage && 'px-docs-model pi-doc-shell-model-detail',
          apiHubPage && 'pi-doc-shell-api-hub',
          metricNav && 'pi-doc-shell-has-metric-nav',
          !showRail && 'px-docs-no-rail',
        )}
        lang={normalized}
      >
        <div className="px-docs-ground" aria-hidden="true">
          <Grain />
          <PrismRay intensity="whisper" className="px-docs-ray" />
        </div>

        <div className="px-docs-top">
          <AnnouncementBar closeLabel={px.dismiss}>
            <strong>{px.announcementTitle}</strong>
            <span className="px-announcement-muted">{px.announcementBody}</span>
            <Link
              className="px-announcement-cta"
              href={localizedHref('/docs/guides/supported-models', normalized)}
            >
              <span className="px-announcement-cta-label">{px.announcementCta}</span>
              <ArrowRight aria-hidden="true" size={13} strokeWidth={2} />
            </Link>
          </AnnouncementBar>
          <DocsHeader
            locale={normalized}
            docsHref={docsHref}
            section={activeGroup ? t.navGroups[activeGroup.id] : undefined}
            languageLinks={languageLinks}
            navLinks={navLinks}
            labels={{
              docs: px.docsCrumb,
              search: t.search,
              community: t.community,
              language: t.language,
              openMenu: t.openMenu,
              closeMenu: t.closeMenu,
              nav: t.nav,
            }}
          />
        </div>

        <div className="px-docs-body">
          <DocsSidebar
            groups={groups}
            currentUrl={page.url}
            labels={{
              sidebar: t.sidebar,
              navGroups: t.navGroups,
              expandApiList: t.expandApiList,
              collapseApiList: t.collapseApiList,
              expandMetricsList: t.expandMetricsList,
              collapseMetricsList: t.collapseMetricsList,
              expandArchitectureList: t.expandArchitectureList,
              collapseArchitectureList: t.collapseArchitectureList,
            }}
            drawerExtras={
              <>
                <nav className="px-drawer-links" aria-label={t.nav}>
                  {navLinks.map((item) => (
                    <Link key={item.href} href={item.href}>
                      {item.label}
                    </Link>
                  ))}
                  <a href={siteLinks.slack} target="_blank" rel="noreferrer">
                    {t.community}
                  </a>
                  <a href={siteLinks.github} target="_blank" rel="noreferrer">
                    GitHub
                  </a>
                </nav>
                <div className="px-drawer-row">
                  <LanguageSwitch links={languageLinks} label={t.language} />
                  <ThemeToggle locale={normalized} />
                </div>
              </>
            }
          />

          <main className="px-sheet" id="px-main">
            <article className={cx('px-article', benchmarkId && 'wf-benchmark-detail')}>
              {isWelcomePage ? (
                <>
                  <div className="px-welcome-actions">{actions}</div>
                  <DocsWelcomeHero description={page.data.description ?? ''} locale={normalized} />
                  <div className="px-prose pi-doc-content">
                    <MDX components={mdxComponents} />
                  </div>
                  {pager}
                </>
              ) : (
                <>
                  {breadcrumbs.length > 1 ? (
                    <nav className="px-breadcrumb" aria-label="Breadcrumb">
                      {breadcrumbs.map((item, index) => (
                        <span className="px-breadcrumb-item" key={item.href ?? item.label}>
                          {index > 0 ? (
                            <ChevronRight aria-hidden="true" size={13} strokeWidth={1.8} />
                          ) : null}
                          {item.href && index < breadcrumbs.length - 1 ? (
                            <Link href={item.href}>{item.label}</Link>
                          ) : (
                            <span aria-current="page">{item.label}</span>
                          )}
                        </span>
                      ))}
                    </nav>
                  ) : null}

                  <header className="px-article-head">
                    <div className="px-title-row">
                      <div className="px-title">
                        {benchmarkId && benchmarkEntry ? (
                          <BenchmarkIdentityMark
                            id={benchmarkId}
                            name={benchmarkEntry.name}
                            category={benchmarkEntry.category}
                            logoKey={benchmarkEntry.logoKey}
                            size="large"
                          />
                        ) : modelEntry ? (
                          <ModelIdentityMark
                            id={modelEntry.id}
                            name={modelEntry.name}
                            provider={modelEntry.provider}
                            category={modelEntry.category}
                            size="large"
                          />
                        ) : null}
                        <h1>{page.data.title}</h1>
                      </div>
                      {actions}
                    </div>
                    {(benchmarkId ? null : page.data.description) || (modelId && paperThumb) ? (
                      <div className={cx('px-article-intro', paperThumb && 'has-thumb')}>
                        {benchmarkId || !page.data.description ? null : (
                          <p className="px-description">{page.data.description}</p>
                        )}
                        {modelId && paperThumb ? (
                          <ModelPaperThumb modelId={modelId} locale={normalized} />
                        ) : null}
                      </div>
                    ) : null}
                  </header>

                  {showToc ? <DocsInlineToc items={toc} title={t.onThisPage} /> : null}

                  <div className="px-prose pi-doc-content">
                    <MDX components={mdxComponents} />
                  </div>

                  {relatedLinks.length > 0 ? (
                    <section className="px-related" aria-label={t.relatedPages}>
                      <p className="px-related-title">{t.relatedPages}</p>
                      <div className="px-related-links">
                        {relatedLinks.map((link) => (
                          <Link key={link.url} href={link.url}>
                            {link.label}
                            <ArrowRight aria-hidden="true" size={13} strokeWidth={1.9} />
                          </Link>
                        ))}
                      </div>
                    </section>
                  ) : null}

                  {pager}
                </>
              )}

              <footer className="px-article-foot">
                {lastUpdated ? (
                  <span>
                    {t.lastUpdated}{' '}
                    <time dateTime={lastUpdated.iso}>{lastUpdated.formatted}</time>
                  </span>
                ) : (
                  <span />
                )}
                <a href={github.edit} target="_blank" rel="noreferrer">
                  {t.editPage}
                </a>
              </footer>
            </article>
          </main>

          {showRail ? (
            <aside className="px-rail">
              <DocsToc items={showToc ? toc : []} title={t.onThisPage}>
                {metricNav ? (
                  <div className="px-rail-extra">
                    <MetricQuickNav
                      locale={normalized}
                      page={metricNav.page}
                      variant={metricNav.variant}
                      placement="rail"
                    />
                  </div>
                ) : null}
              </DocsToc>
            </aside>
          ) : null}
        </div>
      </div>
    </DocsNavProvider>
  );
}
