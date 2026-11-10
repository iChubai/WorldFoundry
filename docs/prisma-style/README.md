# WorldFoundry docs — Prisma-style site

A second front end for the WorldFoundry documentation. It renders the same
content, catalogs, and widgets as `docs/fumadocs/` with a visual language
modeled on [prisma.io](https://www.prisma.io/): a paper ground with floating
reading sheets, a morphing pill navbar, Sora and Inter type, and the
cyan / yellow / red spectrum accents. The implementation is independent; no
Prisma source code or brand assets are used.

## Run

Docs development runs from a local-SSD mirror, the same way `npm run dev:local`
works for `docs/fumadocs/`:

```bash
cd docs/prisma-style
npm run dev:local          # syncs both sites to /tmp, installs, starts next dev on :3002
npm run sync:local         # push checkout edits into the running mirror (hot reload)
```

`PORT` changes the port. `WF_DOCS_DEV_ORIGINS` adds hostnames allowed to reach
dev resources; it defaults to this machine's IP so hot reload also works when
the site is opened by IP.

## How it fits together

| Piece | Source |
| --- | --- |
| MDX content | `../fumadocs/content/docs` via `source.config.ts` |
| Widgets, data, i18n, search dialog | `../fumadocs/{components,lib}` through the `@/*` path fallback |
| Static assets | `public` → `../fumadocs/public` (symlink) |
| Widget styles | `app/shared/*.css`, generated from `../fumadocs/app` by `scripts/sync-shared-styles.mjs` |
| Theme and chrome | `app/styles/*.css`, `components/px/` |

`tsconfig.json` maps `@/*` to `./*` first and `../fumadocs/*` second, so a file
here overrides its counterpart there and everything else is shared. The
overrides are the modules that resolve files relative to the working directory:
`lib/source.ts`, `lib/read-site-json.ts`, and `lib/docs-last-updated.ts`.

Shared files have no `node_modules` of their own. Webpack falls back to this
site's packages, and the mirror links `fumadocs/node_modules` to this site's
install so `tsc` resolves them too.

`app/styles/tokens.css` defines the `--px-*` palette and maps the shared
`--pi-*` and `--color-fd-*` variables onto it, so shared widgets follow the
theme in light and dark mode.

## Layout

- `app/(home)/` — homepage, Blog, Events, OpenEnvision, with the shared navbar and footer
- `app/docs/`, `app/[lang]/docs/` — docs pages (English and Chinese)
- `app/api/search`, `app/llms*.txt`, `app/llms.mdx`, `app/og` — search index, LLM text, Markdown export, social cards
- `components/px/docs/` — docs shell: navbar, sidebar, TOC rail, page actions
- `components/px/home/` — homepage blocks

Fonts: Sora, Inter, and Geist Mono (SIL OFL 1.1) from Fontsource; Noto Sans SC
covers Chinese.
