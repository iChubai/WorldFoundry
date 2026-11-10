import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMDX } from 'fumadocs-mdx/next';
import { docsMovedRedirects } from '../fumadocs/lib/docs-moved-redirects.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const configPath = fileURLToPath(import.meta.url);
const withMDX = createMDX();

// Development output stays inside this app: Next resolves server bundles from
// their real location, so the dist dir must sit next to this app's node_modules.
const distDir = process.env.NODE_ENV === 'production' ? '.next' : 'tmp/next';
const webpackCacheDir =
  process.env.WF_DOCS_WEBPACK_CACHE_DIR ?? path.join(__dirname, 'tmp', 'webpack');
const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const basePath =
  configuredBasePath && configuredBasePath !== '/'
    ? `/${configuredBasePath.replace(/^\/+|\/+$/g, '')}`
    : '';
const devOrigins = (process.env.WF_DOCS_DEV_ORIGINS ?? '')
  .split(',')
  .map((item) => item.trim())
  .filter(Boolean);

/** @type {import('next').NextConfig} */
const config = {
  allowedDevOrigins: ['127.0.0.1', 'localhost', ...devOrigins],
  distDir,
  ...(basePath ? { assetPrefix: basePath, basePath } : {}),
  ...(process.env.NODE_ENV === 'production'
    ? { output: 'export' }
    : {
        async redirects() {
          return docsMovedRedirects.map(([source, destination]) => ({
            source,
            destination,
            permanent: true,
          }));
        },
        async rewrites() {
          return [
            { source: '/docs.md', destination: '/llms.mdx/docs.md' },
            { source: '/docs.mdx', destination: '/llms.mdx/docs.md' },
            { source: '/zh/docs.md', destination: '/llms.mdx/zh/docs.md' },
            { source: '/zh/docs.mdx', destination: '/llms.mdx/zh/docs.md' },
            { source: '/docs/:path*.md', destination: '/llms.mdx/docs/:path*.md' },
            { source: '/docs/:path*.mdx', destination: '/llms.mdx/docs/:path*.md' },
            { source: '/zh/docs/:path*.md', destination: '/llms.mdx/zh/docs/:path*.md' },
            { source: '/zh/docs/:path*.mdx', destination: '/llms.mdx/zh/docs/:path*.md' },
          ];
        },
      }),
  outputFileTracingRoot: path.resolve(__dirname, '..'),
  reactStrictMode: true,
  images: {
    unoptimized: true,
  },
  experimental: {
    // Shared components, data, and MDX content live in ../fumadocs.
    externalDir: true,
    optimizePackageImports: ['lucide-react', 'fumadocs-ui', 'fumadocs-core'],
  },
  webpack(config, { dev }) {
    // Files under ../fumadocs have no node_modules of their own; bare imports
    // from there fall back to this app's packages after the normal lookup.
    config.resolve.modules = [
      ...(config.resolve.modules ?? ['node_modules']),
      path.join(__dirname, 'node_modules'),
    ];
    if (dev) {
      config.cache = {
        type: 'filesystem',
        cacheDirectory: webpackCacheDir,
        buildDependencies: {
          config: [configPath],
        },
      };
      config.watchOptions = {
        ...config.watchOptions,
        ignored: [
          '**/.next/**',
          '**/node_modules/**',
          path.join(__dirname, 'tmp', '**'),
          path.join(__dirname, '..', 'fumadocs', 'tmp', '**'),
          '**/out/**',
          '**/generated/**',
          '**/public/**',
        ],
      };
    }
    return config;
  },
};

export default withMDX(config);
