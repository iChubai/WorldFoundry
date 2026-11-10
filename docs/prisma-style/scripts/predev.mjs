// Prepare generated inputs before `next dev` / `next build`.
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const appRoot = join(dirname(fileURLToPath(import.meta.url)), '..');

function runNode(args, label) {
  const result = spawnSync(process.execPath, args, { cwd: appRoot, stdio: 'inherit' });
  if (result.error || result.status !== 0) {
    console.error(result.error ?? `${label} failed`);
    process.exit(result.status ?? 1);
  }
}

runNode([join(appRoot, 'scripts', 'sync-shared-styles.mjs')], 'shared style sync');

const mdxCli = join(appRoot, 'node_modules', 'fumadocs-mdx', 'bin.js');
if (!existsSync(mdxCli)) {
  console.error('prisma-style: fumadocs-mdx is not installed; run npm ci first');
  process.exit(1);
}
runNode([mdxCli], 'MDX generation');
