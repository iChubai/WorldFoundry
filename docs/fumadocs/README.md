# WorldFoundry Documentation

[English](#local-development) · [中文](#中文) · [Live site](https://openenvision.github.io/WorldFoundry/)

The WorldFoundry documentation site, built with **Next.js 16**, **React 19**, **Fumadocs**, and **Tailwind CSS 4**. The homepage, model catalog, benchmark hub, guides, and Python API reference share this app.

- [English documentation](https://openenvision.github.io/WorldFoundry/docs)
- [中文文档](https://openenvision.github.io/WorldFoundry/zh/docs)

## Local development

Requirements:

- **Node.js 22** (see [`.nvmrc`](.nvmrc)) and npm.
- **Python 3.10+** for API metadata and catalog generators. Use the repository's [unified development environment](../../CONTRIBUTING.md#development-environment) for Python-backed tooling.

From a fresh checkout:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenEnvision/WorldFoundry.git
cd WorldFoundry/docs/fumadocs

# If using nvm:
nvm use

npm ci
npm run dev
```

Open <http://localhost:3000>; documentation is available at `/docs` and `/zh/docs`. Demo videos are served from GitHub CDN, so docs development does not need `git lfs pull`.

Python scripts select `WF_DOCS_PYTHON`, then `PYTHON`, then `python3` / `python`. Set `WF_DOCS_PYTHON` to your environment's Python executable when needed.

### Fast iteration

Run these commands from `docs/fumadocs/`:

| Command | Use |
| --- | --- |
| `npm run dev:fast` | UI/CSS iteration with existing MDX and API metadata. |
| `npm run dev:ssd` | Move hot caches to local SSD while keeping sources in the checkout. |
| `npm run dev:local` | Mirror the site to local SSD when dependency I/O is also slow. |
| `npm run build:fast` | Preview static export without the full validation gates. |

After adding pages or changing navigation/frontmatter, stop the dev server and run `npm run mdx:generate` before restarting a fast dev session. See the [maintainer guide](content/docs/maintainers/contributing.mdx#fast-site-iteration) for cache configuration and refresh options.

## Directory layout

```text
app/              Next.js routes, layouts, and styles
components/       Shared site and MDX components
content/docs/     English and Chinese documentation pages and navigation
content/blog/     Blog articles
mdx/partials/     Reusable MDX content
lib/              Site helpers and generated catalog data
public/           Images, fonts, and other static assets
scripts/          Code generation, checks, and local development helpers
source.config.ts  MDX collections, schemas, and plugins
next.config.mjs   Static export, base path, and cache configuration
```

`.source/`, `generated/`, `tmp/`, `.next/`, and `out/` contain generated metadata, caches, or build output.

## Editing documentation

- Keep English and Chinese pages paired: `page.mdx` and `page.zh.mdx`.
- Update the owning directory's `meta.json` and `meta.zh.json` navigation when needed.
- Use `components/` for reusable UI and `mdx/partials/` for shared prose.
- Set `pageSource: authored` in a model page's frontmatter before hand-editing it; model generation preserves authored pages.
- Regenerate catalog data from its source rather than editing generated JSON or generated model pages directly.

Run generator commands from `docs/fumadocs/`:

| Source changed | Generate | Verify |
| --- | --- | --- |
| Python API sources | `npm run api:generate` | `npm run api:check` |
| Model manifests or homepage recipes | `npm run models:generate` | `npm run models:check` |
| Benchmark manifests or page data | `npm run benchmarks:generate` | `npm run benchmarks:check` |
| Catalog coverage | `npm run coverage:generate` | `npm run coverage:check` |

`npm run models:homes:check` validates model homepage content. `npm run types:check` generates MDX/types and checks TypeScript.

## Build and preview

From the **repository root**:

```bash
make docs-check
bash scripts/docs/build.sh
npm --prefix docs/fumadocs start
```

The build script installs dependencies, checks types and generated data, and exports the site to `docs/fumadocs/out/`. Use `bash scripts/docs/build.sh --skip-install` when dependencies are already installed. `start` serves the static export at <http://localhost:3000>.

## Deployment

[Deploy Docs](../../.github/workflows/deploy-docs.yml) builds and publishes `out/` to GitHub Pages on matching changes to `main`, or through a manual workflow run.

| Variable | Purpose | Official deployment |
| --- | --- | --- |
| `NEXT_PUBLIC_BASE_PATH` | URL prefix; empty for local development. | `/WorldFoundry` |
| `NEXT_PUBLIC_SITE_URL` | Public site origin. | `https://openenvision.github.io` |
| `NEXT_PUBLIC_DEMO_ASSET_BASE_URL` | CDN base URL for demo assets. | GitHub media CDN for `docs/fumadocs/public` |

Set deployment variables before building. Demo videos remain on the CDN rather than in the Pages artifact.

Contribution and DCO rules live in [CONTRIBUTING.md](../../CONTRIBUTING.md). Detailed workflows live in the [maintainer guide](content/docs/maintainers/contributing.mdx). First-party content follows the repository's [license](../../LICENSE) and [third-party notices](../../THIRD-PARTY-NOTICES).

## 中文

这是 WorldFoundry 的文档站源码，使用 **Next.js 16、React 19、Fumadocs 和 Tailwind CSS 4**。首页、模型目录、Benchmark Hub、使用指南和 Python API 参考都在同一个应用中。

### 本地启动

准备 Node.js 22、npm 和 Python 3.10+；Python 文档工具使用仓库的[统一开发环境](../../CONTRIBUTING.md#development-environment)。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenEnvision/WorldFoundry.git
cd WorldFoundry/docs/fumadocs

# 使用 nvm 时执行：
nvm use

npm ci
npm run dev
```

访问 <http://localhost:3000>，英文文档位于 `/docs`，中文文档位于 `/zh/docs`。演示视频通过 GitHub CDN 加载，不需要执行 `git lfs pull`。需要指定 Python 解释器时，设置 `WF_DOCS_PYTHON`。

日常 UI/CSS 修改可使用 `npm run dev:fast`；共享文件系统上可使用 `npm run dev:ssd` 将缓存放到本地 SSD，或使用 `npm run dev:local` 镜像整个站点。`npm run build:fast` 用于快速预览静态导出，不替代完整构建检查。

### 目录与内容维护

- `app/`：页面路由、布局和样式；`components/`：共享 UI 与 MDX 组件。
- `content/docs/`：文档及导航；`content/blog/`：博客文章；`mdx/partials/`：共享 MDX 内容。
- `lib/`：站点工具与生成的目录数据；`public/`：静态资源；`scripts/`：生成、检查和开发脚本。
- `source.config.ts`：MDX 配置；`next.config.mjs`：静态导出、路径前缀和缓存配置。
- `.source/`、`generated/`、`tmp/`、`.next/`、`out/`：生成数据、缓存或构建产物。

文档使用 `page.mdx` / `page.zh.mdx` 成对维护，导航对应 `meta.json` / `meta.zh.json`。手工维护模型页面前，将 frontmatter 的 `pageSource` 设为 `authored`。生成数据从源文件更新，不直接修改生成的 JSON 或模型页面。

新增页面或修改导航、frontmatter 后，停止开发服务器，执行 `npm run mdx:generate`，再启动快速开发模式。API、模型、Benchmark 和覆盖率数据分别通过 `api:*`、`models:*`、`benchmarks:*`、`coverage:*` 的 `generate` / `check` 命令维护，见上方命令表。

### 构建与发布

在**仓库根目录**执行：

```bash
make docs-check
bash scripts/docs/build.sh
npm --prefix docs/fumadocs start
```

构建脚本安装依赖、检查类型与生成数据，并将静态站点导出至 `docs/fumadocs/out/`。依赖已安装时可添加 `--skip-install`；预览地址为 <http://localhost:3000>。

GitHub Pages 由 [Deploy Docs](../../.github/workflows/deploy-docs.yml) 工作流发布。部署前设置 `NEXT_PUBLIC_BASE_PATH`、`NEXT_PUBLIC_SITE_URL` 和 `NEXT_PUBLIC_DEMO_ASSET_BASE_URL`，含义见上方配置表。

贡献与 DCO 约定见 [CONTRIBUTING.md](../../CONTRIBUTING.md)，详细开发流程见[中文维护指南](content/docs/maintainers/contributing.zh.mdx)。许可见 [LICENSE](../../LICENSE) 和 [THIRD-PARTY-NOTICES](../../THIRD-PARTY-NOTICES)。
