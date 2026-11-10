import path from 'node:path';

/**
 * Root of the main docs app. This site reads its content, generated catalogs,
 * and shared widgets from there instead of keeping copies.
 */
export const FUMADOCS_ROOT = path.resolve(
  process.env.WF_FUMADOCS_ROOT ?? path.join(process.cwd(), '..', 'fumadocs'),
);
