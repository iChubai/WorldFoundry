import type { Locale } from '@/lib/i18n';
import { defaultLocale } from '@/lib/i18n';

type LocalizedLabel = Record<Locale, string>;

export type MetricQuickNavItem = {
  id: LocalizedLabel;
  label: LocalizedLabel;
};

export type MetricQuickNavGroup = {
  id: string;
  page: MetricQuickNavPageId;
  label: LocalizedLabel;
  items: MetricQuickNavItem[];
};

export type MetricQuickNavPageId = 'scorers' | 'distribution';

export const metricQuickNavLabels: Record<
  Locale,
  { title: string; summary: string; hubTitle: string }
> = {
  en: {
    title: 'Jump to a metric',
    summary: '{count} ids · click to expand',
    hubTitle: 'Metric pages',
  },
  zh: {
    title: '跳转到指标',
    summary: '{count} 个 id · 点击展开',
    hubTitle: '指标子页面',
  },
};

export const metricQuickNavPageMeta: Record<
  MetricQuickNavPageId,
  { slug: string; label: LocalizedLabel; description: LocalizedLabel }
> = {
  scorers: {
    slug: '/evaluation/metrics/scorers',
    label: { en: 'Multimodal scorers', zh: '多模态 scorer' },
    description: {
      en: 'CLIPScore, VQAScore, ITMScore',
      zh: 'CLIPScore、VQAScore、ITMScore',
    },
  },
  distribution: {
    slug: '/evaluation/metrics/distribution',
    label: { en: 'Distribution', zh: '分布指标' },
    description: {
      en: 'FID, Inception Score, KID, FVD, and JEDi',
      zh: 'FID、Inception Score、KID、FVD 与 JEDi',
    },
  },
};

/** Anchor ids match fumadocs slug output for `### \`metric\` (↑/↓)` headings. */
export const metricQuickNavGroups: MetricQuickNavGroup[] = [
  {
    id: 'multimodal-scorers',
    page: 'scorers',
    label: { en: 'Multimodal scorers', zh: '多模态 scorer' },
    items: [
      { id: { en: 'clip_score-', zh: 'clip_score-' }, label: { en: 'clip_score', zh: 'clip_score' } },
      { id: { en: 'vqa_score-', zh: 'vqa_score-' }, label: { en: 'vqa_score', zh: 'vqa_score' } },
      { id: { en: 'itm_score-', zh: 'itm_score-' }, label: { en: 'itm_score', zh: 'itm_score' } },
    ],
  },
  {
    id: 'image-distribution',
    page: 'distribution',
    label: { en: 'Image distribution', zh: '图像分布' },
    items: [
      { id: { en: 'fid-', zh: 'fid-' }, label: { en: 'fid', zh: 'fid' } },
      { id: { en: 'scene_fid-', zh: 'scene_fid-' }, label: { en: 'scene_fid', zh: 'scene_fid' } },
      { id: { en: 'inception_score-', zh: 'inception_score-' }, label: { en: 'inception_score', zh: 'inception_score' } },
      { id: { en: 'kid-', zh: 'kid-' }, label: { en: 'kid', zh: 'kid' } },
    ],
  },
  {
    id: 'video-distribution',
    page: 'distribution',
    label: { en: 'Video distribution', zh: '视频分布' },
    items: [
      { id: { en: 'fvd-', zh: 'fvd-' }, label: { en: 'fvd', zh: 'fvd' } },
      { id: { en: 'jedi-', zh: 'jedi-' }, label: { en: 'jedi', zh: 'jedi' } },
    ],
  },
];

export function metricDocsPath(locale: Locale, slug: string) {
  return locale === defaultLocale ? `/docs${slug}` : `/${locale}/docs${slug}`;
}

export function metricQuickNavGroupsForPage(page: MetricQuickNavPageId) {
  return metricQuickNavGroups.filter((group) => group.page === page);
}

export type MetricQuickNavResolution = {
  variant: 'hub' | 'page';
  page?: MetricQuickNavPageId;
};

export function resolveMetricQuickNavFromSlugs(
  slugs: readonly string[],
): MetricQuickNavResolution | null {
  if (slugs[0] !== 'evaluation' || slugs[1] !== 'metrics') {
    return null;
  }
  if (slugs.length === 2 || slugs[2] === 'reference') {
    return { variant: 'hub' };
  }
  if (slugs[2] === 'scorers' || slugs[2] === 'distribution') {
    return { variant: 'page', page: slugs[2] };
  }
  return null;
}
