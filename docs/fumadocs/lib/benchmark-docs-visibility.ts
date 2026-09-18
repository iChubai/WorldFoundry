/** Benchmarks that exist in-repo but should not appear in public docs yet. */
export const hiddenBenchmarkDocsIds = new Set(['worldatlas-arena']);

export function isBenchmarkHiddenFromDocs(id: string) {
  return hiddenBenchmarkDocsIds.has(id);
}
