import 'server-only';

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { FUMADOCS_ROOT } from '@/lib/fumadocs-root';

const cache = new Map<string, unknown>();

/** Read generated JSON from the main docs app at runtime instead of bundling multi-MB catalogs. */
export function readSiteJson<T>(relativePath: string): T {
  const cached = cache.get(relativePath);
  if (cached) {
    return cached as T;
  }

  const value = JSON.parse(readFileSync(join(FUMADOCS_ROOT, relativePath), 'utf8')) as T;
  cache.set(relativePath, value);
  return value;
}
