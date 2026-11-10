// Copy the main docs app's widget styles into app/shared/.
// Their @import lines are dropped: this app imports Tailwind, Fumadocs, KaTeX,
// and fonts itself, and bare imports cannot resolve from ../fumadocs.
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const appRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const fumadocsRoot = process.env.WF_FUMADOCS_ROOT ?? join(appRoot, '..', 'fumadocs');
const outDir = join(appRoot, 'app', 'shared');

const files = [
  ['app/styles/tokens.css', 'tokens.css'],
  ['app/global.css', 'global.css'],
  ['app/fd-skin.css', 'fd-skin.css'],
  ['app/arch-diagram.css', 'arch-diagram.css'],
];

mkdirSync(outDir, { recursive: true });

let changed = 0;
for (const [from, to] of files) {
  const source = readFileSync(join(fumadocsRoot, from), 'utf8');
  const body = source
    .split('\n')
    .filter((line) => !/^@import\s/.test(line))
    .join('\n');
  const output = `/* Generated from docs/fumadocs/${from} by scripts/sync-shared-styles.mjs. */\n${body}`;
  const target = join(outDir, to);
  if (existsSync(target) && readFileSync(target, 'utf8') === output) continue;
  writeFileSync(target, output);
  changed += 1;
}

console.log(`prisma-style: shared styles ${changed ? `updated (${changed})` : 'up to date'}`);
