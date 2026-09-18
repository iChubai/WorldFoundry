import catalogCoverageData from '@/lib/catalog-coverage-data.json';
import { isBenchmarkHiddenFromDocs } from '@/lib/benchmark-docs-visibility';

export type CatalogEntry = {
  id: string;
  name: string;
  status?: string | null;
  /** Variant / catalog ids collapsed into this display row. */
  aliases?: string[];
  /** Human-readable version labels for a grouped model family. */
  versions?: string[];
};

export type CatalogFamily = {
  id: string;
  label: string;
  blurb: string;
  count: number;
  entries: CatalogEntry[];
};

export type CatalogBenchmarkGroup = {
  id: string;
  label: string;
  count: number;
  entries: CatalogEntry[];
};

export type CatalogCoverageData = {
  modelsTotal: number;
  modelsListed?: number;
  benchmarksTotal: number;
  modelFamilies: CatalogFamily[];
  benchmarkGroups: CatalogBenchmarkGroup[];
};

const rawCoverage = catalogCoverageData as CatalogCoverageData;

export const catalogCoverage: CatalogCoverageData = {
  ...rawCoverage,
  benchmarkGroups: rawCoverage.benchmarkGroups.map((group) => {
    const entries = group.entries.filter((entry) => !isBenchmarkHiddenFromDocs(entry.id));
    return { ...group, entries, count: entries.length };
  }),
  benchmarksTotal: rawCoverage.benchmarkGroups.reduce((total, group) => {
    return (
      total + group.entries.filter((entry) => !isBenchmarkHiddenFromDocs(entry.id)).length
    );
  }, 0),
};

/** Benchmark Hub pages currently missing from docs (catalog still lists them). */
export const benchmarkHubMissingIds = new Set(['worldreasonbench', 'wrbench']);
