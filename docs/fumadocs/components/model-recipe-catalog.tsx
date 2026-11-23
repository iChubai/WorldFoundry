'use client';

import Link from 'next/link';
import { ArrowRight, Search } from 'lucide-react';
import { type CSSProperties, useDeferredValue, useMemo, useState } from 'react';

import { ModelIdentityMark } from '@/components/model-identity-mark';
import { modelRecipeIndex } from '@/lib/model-recipe-index';
import { formatCudaLabel } from '@/lib/runtime-labels';
import {
  catalogReadinessLabel,
  type ModelRecipeIndexEntry,
  type ModelRecipeStatusGroup,
} from '@/lib/model-recipe-types';

type Locale = 'en' | 'zh';
type RuntimeFilter = 'all' | 'dedicated' | 'unified' | 'unrecorded';
type SortMode = 'readiness' | 'name';

const PAGE_SIZE = 60;
const STATUS_RANK: Record<ModelRecipeStatusGroup, number> = {
  verified: 0,
  integrated: 1,
  runtime_ported: 2,
  profile: 3,
  planned: 4,
  blocked: 5,
};

const copy = {
  en: {
    modelsCount: 'models',
    coverage: 'runtime data from repository manifests',
    search: 'Search by model, task, provider, or alias…',
    all: 'All',
    results: 'recipes',
    status: 'Status',
    allStatuses: 'All statuses',
    runtime: 'Runtime',
    allRuntimes: 'All runtimes',
    dedicated: 'Dedicated environment',
    unified: 'Unified environment',
    dedicatedShort: 'Dedicated env',
    unifiedShort: 'Unified env',
    unrecorded: 'Runtime not recorded',
    sort: 'Sort',
    readiness: 'Readiness',
    name: 'Name A–Z',
    model: 'Model',
    tasks: 'Tasks',
    python: 'Python',
    cuda: 'CUDA',
    none: 'No recipes matched these filters.',
    more: 'Show more recipes',
  },
  zh: {
    modelsCount: '个模型',
    coverage: '运行时数据来自仓库 manifest',
    search: '按模型、任务、提供方或别名搜索…',
    all: '全部',
    results: '个配方',
    status: '状态',
    allStatuses: '全部状态',
    runtime: '运行时',
    allRuntimes: '全部运行时',
    dedicated: '独立环境',
    unified: '统一环境',
    dedicatedShort: '独立环境',
    unifiedShort: '统一环境',
    unrecorded: '未记录运行时',
    sort: '排序',
    readiness: '按就绪度',
    name: '按名称 A–Z',
    model: '模型',
    tasks: '任务',
    python: 'Python',
    cuda: 'CUDA',
    none: '没有符合当前筛选条件的模型配方。',
    more: '显示更多模型配方',
  },
} as const;

function includesQuery(recipe: ModelRecipeIndexEntry, query: string) {
  if (!query) return true;
  const haystack = [
    recipe.id,
    recipe.name,
    recipe.provider,
    recipe.providerZh ?? '',
    recipe.summary,
    recipe.summaryZh ?? '',
    ...recipe.aliases,
    ...recipe.tasks,
    ...(recipe.tasksZh ?? []),
  ]
    .join(' ')
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function recipeHref(recipe: ModelRecipeIndexEntry, locale: Locale) {
  const prefix = locale === 'zh' ? '/zh' : '';
  return `${prefix}/docs/guides/supported-models/${recipe.id}`;
}

const GENERIC_RAW_TASKS = new Set(['world', 'video', 'multimodal', 'three_dimension']);

const TASK_ACRONYMS: Record<string, string> = {
  '3d': '3D',
  '4d': '4D',
  '3dgs': '3DGS',
  api: 'API',
  jepa: 'JEPA',
  rgb: 'RGB',
  umi: 'UMI',
  va: 'VA',
  vam: 'VAM',
  vla: 'VLA',
  vlm: 'VLM',
  wam: 'WAM',
};

function taskFactLabel(task: string) {
  const words = task
    .replace(/[._-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean)
    .map((word) => TASK_ACRONYMS[word.toLowerCase()] ?? word);
  if (!words.length) return task;
  const joined = words.join(' ');
  return joined.charAt(0).toUpperCase() + joined.slice(1);
}

function primaryTaskFact(recipe: ModelRecipeIndexEntry, locale: Locale) {
  const rawTasks = recipe.tasks ?? [];
  if (!rawTasks.length) return '';
  const specificIndex = rawTasks.findIndex((item) => !GENERIC_RAW_TASKS.has(item.toLowerCase()));
  const index = specificIndex >= 0 ? specificIndex : 0;
  if (locale === 'zh' && recipe.tasksZh?.[index]) {
    return recipe.tasksZh[index];
  }
  return taskFactLabel(rawTasks[index]);
}

function recipeFacts(recipe: ModelRecipeIndexEntry, t: (typeof copy)[Locale], locale: Locale) {
  const envFact =
    recipe.runtime.environmentKind === 'dedicated'
      ? t.dedicatedShort
      : recipe.runtime.environmentKind === 'unified'
        ? t.unifiedShort
        : '';
  const stackFact = [
    envFact,
    recipe.runtime.python ? `${t.python} ${recipe.runtime.python}` : '',
    formatCudaLabel(recipe.runtime.cudaLabel),
  ]
    .filter(Boolean)
    .join(' · ');
  return [
    catalogReadinessLabel(recipe.status.group, locale),
    primaryTaskFact(recipe, locale),
    stackFact,
  ].filter(Boolean);
}

export function ModelRecipeCatalog({ locale = 'en' }: { locale?: Locale }) {
  const t = copy[locale];
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [status, setStatus] = useState<'all' | ModelRecipeStatusGroup>('all');
  const [runtime, setRuntime] = useState<RuntimeFilter>('all');
  const [sort, setSort] = useState<SortMode>('readiness');
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const deferredQuery = useDeferredValue(query.trim());

  const results = useMemo(() => {
    const filtered = modelRecipeIndex.recipes.filter((recipe) => {
      if (category !== 'all' && recipe.category !== category) return false;
      if (status !== 'all' && recipe.status.group !== status) return false;
      if (runtime !== 'all' && recipe.runtime.environmentKind !== runtime) return false;
      return includesQuery(recipe, deferredQuery);
    });

    return filtered.sort((left, right) => {
      if (sort === 'name') return left.name.localeCompare(right.name);
      const statusDifference = STATUS_RANK[left.status.group] - STATUS_RANK[right.status.group];
      return statusDifference || left.name.localeCompare(right.name);
    });
  }, [category, deferredQuery, runtime, sort, status]);

  const visibleResults = results.slice(0, visibleCount);
  const resetVisible = () => setVisibleCount(PAGE_SIZE);

  return (
    <div className="wf-recipe-catalog">
      <div className="wf-recipe-catalog-coverage">
        <span>
          {modelRecipeIndex.total} {t.modelsCount}
        </span>
        <span aria-hidden="true">·</span>
        <span>{t.coverage}</span>
      </div>

      <label className="wf-recipe-search">
        <Search aria-hidden="true" size={20} strokeWidth={1.6} />
        <span className="sr-only">{t.search}</span>
        <input
          type="search"
          value={query}
          placeholder={t.search}
          autoComplete="off"
          onChange={(event) => {
            setQuery(event.target.value);
            resetVisible();
          }}
        />
      </label>

      <div className="wf-recipe-family-tabs" role="tablist" aria-label="Model families">
        <button
          type="button"
          role="tab"
          aria-selected={category === 'all'}
          className={category === 'all' ? 'is-active' : undefined}
          onClick={() => {
            setCategory('all');
            resetVisible();
          }}
        >
          {t.all}
          <span>{modelRecipeIndex.total}</span>
        </button>
        {modelRecipeIndex.categories.map((item) => (
          <button
            type="button"
            role="tab"
            aria-selected={category === item.id}
            className={category === item.id ? 'is-active' : undefined}
            key={item.id}
            onClick={() => {
              setCategory(item.id);
              resetVisible();
            }}
          >
            {locale === 'zh' ? item.label_zh : item.label}
            <span>{item.count}</span>
          </button>
        ))}
      </div>

      <div className="wf-recipe-catalog-toolbar">
        <p aria-live="polite">
          <strong>{results.length}</strong> {t.results}
        </p>
        <div>
          <label className={status !== 'all' ? 'is-set' : undefined}>
            <span>{t.status}</span>
            <select
              value={status}
              onChange={(event) => {
                setStatus(event.target.value as 'all' | ModelRecipeStatusGroup);
                resetVisible();
              }}
            >
              <option value="all">{t.allStatuses}</option>
              <option value="verified">{catalogReadinessLabel('verified', locale)}</option>
              <option value="integrated">{catalogReadinessLabel('integrated', locale)}</option>
              <option value="runtime_ported">{catalogReadinessLabel('runtime_ported', locale)}</option>
              <option value="profile">{catalogReadinessLabel('profile', locale)}</option>
              <option value="planned">{catalogReadinessLabel('planned', locale)}</option>
              <option value="blocked">{catalogReadinessLabel('blocked', locale)}</option>
            </select>
          </label>
          <label className={runtime !== 'all' ? 'is-set' : undefined}>
            <span>{t.runtime}</span>
            <select
              value={runtime}
              onChange={(event) => {
                setRuntime(event.target.value as RuntimeFilter);
                resetVisible();
              }}
            >
              <option value="all">{t.allRuntimes}</option>
              <option value="dedicated">{t.dedicated}</option>
              <option value="unified">{t.unified}</option>
              <option value="unrecorded">{t.unrecorded}</option>
            </select>
          </label>
          <label className={sort !== 'readiness' ? 'is-set' : undefined}>
            <span>{t.sort}</span>
            <select value={sort} onChange={(event) => setSort(event.target.value as SortMode)}>
              <option value="readiness">{t.readiness}</option>
              <option value="name">{t.name}</option>
            </select>
          </label>
        </div>
      </div>

      <div className="wf-recipe-table wf-model-table wf-recipe-list" role="list" aria-label="Model recipes">
        {visibleResults.length > 0 ? (
          visibleResults.map((recipe, index) => {
            const facts = recipeFacts(recipe, t, locale);
            const providerLabel =
              (locale === 'zh' ? recipe.providerZh || recipe.provider : recipe.provider) ||
              (locale === 'zh' ? recipe.categoryLabelZh : recipe.categoryLabel);
            const summaryText = locale === 'zh' ? recipe.summaryZh || recipe.summary : recipe.summary;
            return (
            <Link
              className="wf-recipe-row"
              href={recipeHref(recipe, locale)}
              role="listitem"
              key={recipe.id}
              style={{ '--wf-row-index': Math.min(index, 12) } as CSSProperties}
            >
              <span className="wf-recipe-row-model">
                <ModelIdentityMark
                  id={recipe.id}
                  name={recipe.name}
                  provider={recipe.provider}
                  category={recipe.category}
                  size="medium"
                />
                <span className="wf-recipe-row-identity">
                  <span>
                    <strong>{recipe.name}</strong>
                    {providerLabel ? <em>{providerLabel}</em> : null}
                  </span>
                  <small>{summaryText}</small>
                  {facts.length > 0 ? (
                    <span className="wf-recipe-row-facts">
                      {facts.map((fact) => (
                        <span key={fact}>{fact}</span>
                      ))}
                    </span>
                  ) : null}
                </span>
              </span>
              <ArrowRight aria-hidden="true" size={17} strokeWidth={1.5} />
            </Link>
            );
          })
        ) : (
          <p className="wf-recipe-empty">{t.none}</p>
        )}
      </div>

      {visibleCount < results.length ? (
        <button
          className="wf-recipe-show-more"
          type="button"
          onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}
        >
          {t.more}
          <span>{Math.min(PAGE_SIZE, results.length - visibleCount)}</span>
        </button>
      ) : null}
    </div>
  );
}
