import { BenchmarkLeaderboard } from '@/components/benchmark-leaderboard';
import { BenchmarkPaperFigures } from '@/components/benchmark-paper-figures';
import { KvCatalog, type KvCatalogRow } from '@/components/kv-catalog';
import { getBenchmarkHeroFigures } from '@/lib/benchmark-paper-figures';
import {
  getBenchmarkPageEntry,
  localizedDataSummary,
  localizedDimensionDescription,
  localizedDimensionName,
  localizedMetricDescription,
  localizedSizeLabel,
} from '@/lib/benchmark-page-data';
import type { Locale } from '@/lib/i18n';

const copy = {
  en: {
    dimensions: 'Dimensions',
    data: 'Data composition',
    metrics: 'Metrics',
    leaderboard: 'Leaderboard',
    kind: 'Kind',
    direction: 'Direction',
    higher: 'Higher is better',
    lower: 'Lower is better',
    primary: 'Primary',
    size: 'Size',
    sources: 'Sources',
    splits: 'Splits',
    modalities: 'Modalities',
    license: 'License',
    notes: 'Notes',
    emptyDimensions: 'No recorded axes, suites, or splits in the catalog.',
    emptyData: 'No recorded dataset size or sources in the catalog.',
    emptyMetrics: 'No metrics recorded in the catalog.',
    notRecorded: 'Not recorded',
    kinds: {
      axis: 'Axis',
      split: 'Split',
      track: 'Track',
      category: 'Category',
      task: 'Task',
      suite: 'Suite',
    },
  },
  zh: {
    dimensions: '评测维度',
    data: '数据组成',
    metrics: '指标',
    leaderboard: '排行榜',
    kind: '类型',
    direction: '方向',
    higher: '越高越好',
    lower: '越低越好',
    primary: '主指标',
    size: '规模',
    sources: '来源',
    splits: '划分',
    modalities: '模态',
    license: '许可证',
    notes: '说明',
    emptyDimensions: '目录未记录评测轴、套件或划分。',
    emptyData: '目录未记录数据集规模或来源。',
    emptyMetrics: '目录未记录指标。',
    notRecorded: '未记录',
    kinds: {
      axis: '轴',
      split: '划分',
      track: '赛道',
      category: '类别',
      task: '任务',
      suite: '套件',
    },
  },
} as const;

const MODALITY_LABELS: Record<string, { en: string; zh: string }> = {
  '3d': { en: '3D', zh: '3D' },
  '4d': { en: '4D', zh: '4D' },
  action: { en: 'Action', zh: '动作' },
  camera: { en: 'Camera', zh: '相机' },
  'camera-trajectory': { en: 'Camera trajectory', zh: '相机轨迹' },
  depth: { en: 'Depth', zh: '深度' },
  frames: { en: 'Frames', zh: '帧序列' },
  'generated-video': { en: 'Generated video', zh: '生成视频' },
  image: { en: 'Image', zh: '图像' },
  language: { en: 'Language', zh: '语言' },
  observation: { en: 'Observation', zh: '观测' },
  pointcloud: { en: 'Point cloud', zh: '点云' },
  proprioception: { en: 'Proprioception', zh: '本体感知' },
  rgb: { en: 'RGB', zh: 'RGB' },
  state: { en: 'State', zh: '状态' },
  text: { en: 'Text', zh: '文本' },
  video: { en: 'Video', zh: '视频' },
  vision: { en: 'Vision', zh: '视觉' },
};

function modalityLabel(value: string, locale: Locale) {
  return MODALITY_LABELS[value]?.[locale] ?? value;
}

function kindLabel(kind: string, locale: Locale) {
  const labels = copy[locale].kinds;
  return labels[kind as keyof typeof labels] ?? kind;
}

function hostLabel(value: string) {
  try {
    return new URL(value).hostname.replace(/^www\./, '');
  } catch {
    return value;
  }
}

export function BenchmarkPageSections({
  id,
  locale = 'en',
}: {
  id: string;
  locale?: Locale;
}) {
  const entry = getBenchmarkPageEntry(id);
  if (!entry) return null;
  const t = copy[locale];

  const dimensionRows: KvCatalogRow[] = entry.dimensions.map((item) => ({
    name: localizedDimensionName(item, locale),
    hintLabel: t.kind,
    hint: kindLabel(item.kind, locale),
    description: localizedDimensionDescription(item, locale) || undefined,
  }));

  const dataRows: KvCatalogRow[] = [];
  const size = localizedSizeLabel(entry, locale);
  const summary = localizedDataSummary(entry, locale);
  if (size || summary) {
    dataRows.push({
      name: t.size,
      description: summary || size || t.notRecorded,
    });
  }
  if (entry.data.sources.length > 0) {
    dataRows.push({
      name: t.sources,
      description: entry.data.sources
        .map((source) => (source.kind === 'huggingface' ? source.id : hostLabel(source.id)))
        .join(' · '),
    });
  }
  if (entry.data.splits.length > 0) {
    dataRows.push({
      name: t.splits,
      description: entry.data.splits.join(' · '),
    });
  }
  if (entry.data.modalities.length > 0) {
    dataRows.push({
      name: t.modalities,
      description: entry.data.modalities.map((item) => modalityLabel(item, locale)).join(' · '),
    });
  }
  if (entry.data.license) {
    dataRows.push({
      name: t.license,
      description: entry.data.license,
    });
  }
  const notes = locale === 'zh' ? entry.data.notesZh : entry.data.notes;
  if (notes.length > 0) {
    dataRows.push({
      name: t.notes,
      description: notes.join(' '),
    });
  }

  const metricRows: KvCatalogRow[] = entry.metrics.map((metric) => {
    const rawDesc = localizedMetricDescription(metric, locale);
    const fallbackName = metric.name !== metric.id ? metric.name : undefined;
    const description = rawDesc || fallbackName;
    const showPrimaryRead = metric.primary && Boolean(rawDesc);
    return {
      name: metric.id,
      hintLabel: t.direction,
      hint: `${metric.higherIsBetter ? t.higher : t.lower}${metric.primary && !showPrimaryRead ? ` · ${t.primary}` : ''}`,
      description,
      reads: showPrimaryRead ? [{ label: t.primary, value: metric.name }] : undefined,
    };
  });

  const heroFigures = getBenchmarkHeroFigures(id, locale);

  return (
    <div className="wf-benchmark-page-sections not-prose">
      {heroFigures.length > 0 ? (
        <BenchmarkPaperFigures benchmarkId={id} locale={locale} figures={heroFigures} />
      ) : null}

      <section aria-labelledby="benchmark-dimensions">
        <h2 id="benchmark-dimensions">{t.dimensions}</h2>
        {dimensionRows.length > 0 ? (
          <KvCatalog rows={dimensionRows} />
        ) : (
          <p className="wf-benchmark-section-empty">{t.emptyDimensions}</p>
        )}
      </section>

      <section aria-labelledby="benchmark-data">
        <h2 id="benchmark-data">{t.data}</h2>
        {dataRows.length > 0 ? (
          <KvCatalog rows={dataRows} />
        ) : (
          <p className="wf-benchmark-section-empty">{t.emptyData}</p>
        )}
      </section>

      <section aria-labelledby="benchmark-metrics">
        <h2 id="benchmark-metrics">{t.metrics}</h2>
        {metricRows.length > 0 ? (
          <KvCatalog heading={locale === 'zh' ? '指标' : 'Metric'} rows={metricRows} />
        ) : (
          <p className="wf-benchmark-section-empty">{t.emptyMetrics}</p>
        )}
      </section>

      <section aria-labelledby="benchmark-leaderboard">
        <h2 id="benchmark-leaderboard">{t.leaderboard}</h2>
        <BenchmarkLeaderboard id={id} locale={locale} fallbackUrl={entry.leaderboard.officialUrl} />
      </section>
    </div>
  );
}
