import { ArrowUpRight } from 'lucide-react';
import type { Metadata } from 'next';

import { CopyCommand } from '@/components/px/copy-command';
import { SubpageHero } from '@/components/px/subpage';
import { OPENENVISION_AWESOME_WORLD_MODELING, OPENENVISION_BLOGXIV_SITE } from '@/lib/site-links';
import { withBasePath } from '@/lib/site-path';

const projects = [
  {
    name: 'ScholarTube',
    category: 'Research videos',
    description: 'A curated, source-linked video knowledge index for AI researchers.',
    descriptionZh: '面向 AI 研究者的视频知识索引，精选研究视频并保留来源链接。',
    repo: 'https://github.com/OpenEnvision/ScholarTube',
    site: 'https://openenvision.github.io/ScholarTube/',
    tone: 'cyan',
  },
  {
    name: 'Awesome Multimodal Modeling',
    category: 'Multimodal research',
    description:
      'Explore research on multimodal large language models, unified multimodal modeling, and native multimodal modeling.',
    descriptionZh: '多模态研究资源合集，覆盖多模态大语言模型、统一多模态建模与原生多模态建模。',
    repo: 'https://github.com/OpenEnvision/Awesome-Multimodal-Modeling',
    site: 'https://openenvision.github.io/Awesome-Multimodal-Modeling/',
    tone: 'yellow',
  },
  {
    name: 'BlogrXiv',
    category: 'Research blogs',
    description: 'Discover AI research blogs and technical writing beyond the paper.',
    descriptionZh: '发现 AI 研究博客与技术文章，了解论文之外的研究思路和实践经验。',
    repo: 'https://github.com/OpenEnvision/BlogrXiv',
    site: OPENENVISION_BLOGXIV_SITE,
    tone: 'red',
  },
  {
    name: 'Awesome World Modeling',
    category: 'World models',
    description:
      'A curated collection of representational, generative, and agentic world-model research.',
    descriptionZh: '世界模型研究资源合集，覆盖表征式、生成式与智能体世界模型。',
    repo: OPENENVISION_AWESOME_WORLD_MODELING,
    site: 'https://openenvision.github.io/Awesome-World-Modeling/',
    tone: 'cyan',
  },
] as const;

const projectLinks = [
  { label: 'Organization', value: 'https://github.com/OpenEnvision', href: 'https://github.com/OpenEnvision' },
  {
    label: 'WorldFoundry repo',
    value: 'https://github.com/OpenEnvision/WorldFoundry',
    href: 'https://github.com/OpenEnvision/WorldFoundry',
  },
  { label: 'Clone URL', value: 'https://github.com/OpenEnvision/WorldFoundry.git' },
];

export const metadata: Metadata = {
  title: 'OpenEnvision',
  description:
    'Explore OpenEnvision research tools and open resources: ScholarTube, Awesome Multimodal Modeling, BlogrXiv, Awesome World Modeling, and WorldFoundry.',
};

export default function OpenEnvisionPage() {
  const logo = withBasePath('/openenvision-logo.png') ?? '/openenvision-logo.png';

  return (
    <main className="px-home">
      <SubpageHero
        label="GitHub organization"
        title="OpenEnvision"
        media={<img src={logo} alt="OpenEnvision logo" width={132} height={132} />}
      >
        <p className="px-lead">
          OpenEnvision Lab is a joint research lab advancing open vision intelligence through
          academia-industry collaboration.
        </p>
        <p className="px-lead" lang="zh-CN">
          OpenEnvision Lab 是一个通过产学协作推进 open vision intelligence 的联合研究实验室。
        </p>
      </SubpageHero>

      <section className="px-section" aria-labelledby="px-projects-title">
        <div className="px-container">
          <header className="px-section-intro" data-align="start">
            <h2 id="px-projects-title">Explore our work</h2>
            <p>
              Open tools and shared resources for discovering, understanding, and building AI.
              <span lang="zh-CN"> 从研究视频、技术博客到模型资源，一起探索和构建 AI。</span>
            </p>
          </header>
          <div className="px-project-grid">
            {projects.map((project) => (
              <article key={project.name} className="px-project" data-tone={project.tone}>
                <p className="px-kicker" data-tone={project.tone}>
                  <span className="px-kicker-dot" aria-hidden="true" />
                  {project.category}
                </p>
                <h3>{project.name}</h3>
                <p>{project.description}</p>
                <p lang="zh-CN" className="px-project-zh">
                  {project.descriptionZh}
                </p>
                <div className="px-project-links">
                  <a
                    href={project.site}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`Explore ${project.name}`}
                  >
                    Explore project
                    <ArrowUpRight aria-hidden="true" size={14} strokeWidth={1.9} />
                  </a>
                  <a
                    href={project.repo}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`${project.name} on GitHub`}
                  >
                    GitHub
                    <ArrowUpRight aria-hidden="true" size={14} strokeWidth={1.9} />
                  </a>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="px-section px-section-tight" aria-labelledby="px-links-title">
        <div className="px-container px-links-grid">
          <div>
            <h2 id="px-links-title" className="px-h3">
              Project Links
            </h2>
            <dl className="px-link-table">
              {projectLinks.map((item) => (
                <div key={item.label}>
                  <dt>{item.label}</dt>
                  <dd>
                    {item.href ? (
                      <a href={item.href} target="_blank" rel="noreferrer">
                        {item.value}
                      </a>
                    ) : (
                      <code>{item.value}</code>
                    )}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
          <div>
            <h2 className="px-h3">Clone</h2>
            <CopyCommand
              label="WorldFoundry OpenEnvision clone command"
              command="git clone https://github.com/OpenEnvision/WorldFoundry.git"
            />
          </div>
        </div>
      </section>
    </main>
  );
}
