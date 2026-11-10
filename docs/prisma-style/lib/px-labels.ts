import { catalogCoverage } from '@/lib/catalog-coverage';
import type { Locale } from '@/lib/i18n';

/** Chrome strings this site adds on top of the shared docsLabels. */
export const pxDocsLabels = {
  en: {
    docsCrumb: 'docs',
    copy: 'Copy Markdown',
    open: 'Open',
    markdown: 'View as Markdown',
    markdownHint: 'Plain-text source for LLMs and tools',
    chatgpt: 'Open in ChatGPT',
    chatgptHint: 'Ask questions about this page',
    claude: 'Open in Claude',
    claudeHint: 'Ask questions about this page',
    edit: 'Edit on GitHub',
    editHint: 'Suggest a change to this page',
    theme: 'Theme',
    dismiss: 'Dismiss',
    announcementTitle: `${catalogCoverage.modelsTotal} model entries · ${catalogCoverage.benchmarksTotal} browsable benchmarks`,
    announcementBody: 'with explicit readiness signals.',
    announcementCta: 'Browse the catalog',
  },
  zh: {
    docsCrumb: '文档',
    copy: '复制 Markdown',
    open: '打开',
    markdown: '查看 Markdown',
    markdownHint: '供 LLM 与工具读取的纯文本源',
    chatgpt: '在 ChatGPT 中打开',
    chatgptHint: '就本页内容提问',
    claude: '在 Claude 中打开',
    claudeHint: '就本页内容提问',
    edit: '在 GitHub 上编辑',
    editHint: '为本页提交修改建议',
    theme: '主题',
    dismiss: '关闭',
    announcementTitle: `${catalogCoverage.modelsTotal} 个模型条目 · ${catalogCoverage.benchmarksTotal} 个可浏览的 benchmark`,
    announcementBody: 'readiness 信号明确可见。',
    announcementCta: '浏览模型目录',
  },
} as const satisfies Record<Locale, Record<string, string>>;
