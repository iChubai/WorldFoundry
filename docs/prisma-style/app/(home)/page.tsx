import {
  ArrowRight,
  ArrowUpRight,
  BookMarked,
  CircleDot,
  ClipboardCheck,
  Eye,
  Layers3,
  MessageCircle,
  MessagesSquare,
  Network,
  ScrollText,
  UserPlus,
} from 'lucide-react';
import Link from 'next/link';

import { CatalogCoverage } from '@/components/catalog-coverage';
import { HomeRunConfigurator, type HomeRecipeOption } from '@/components/home-run-configurator';
import { ArrowLink, OutlineButton, PrismButton } from '@/components/px/buttons';
import { Connector, IconTile, Kicker } from '@/components/px/home/blocks';
import { HeroShowcase } from '@/components/px/home/hero-showcase';
import { LogoCloud } from '@/components/px/home/logo-cloud';
import { CheckBold } from '@/components/px/icons';
import { Grain, PrismRay } from '@/components/px/prism-ray';
import { WorldModelProgression } from '@/components/world-model-progression';
import { WorldFoundryWorkflow } from '@/components/worldfoundry-system-map';
import { catalogCoverage } from '@/lib/catalog-coverage';
import { homeHeroVideo } from '@/lib/home-hero-video';
import { modelRecipeIndex } from '@/lib/model-recipe-index';
import { catalogReadinessLabel } from '@/lib/model-recipe-types';
import {
  OPENENVISION_AWESOME_WORLD_MODELING,
  OPENENVISION_BLOGXIV_SITE,
  WORLDFOUNDRY_GITHUB_ISSUES,
  WORLDFOUNDRY_GITHUB_REPO,
  WORLDFOUNDRY_SLACK_INVITE,
  WORLDFOUNDRY_WECHAT_QR,
} from '@/lib/site-links';
import { withBasePath } from '@/lib/site-path';

const pillars = [
  {
    title: 'Cataloged',
    description:
      'Declarative model and benchmark manifests, plus explicit runtime and asset management, make heterogeneous systems discoverable before compute is spent.',
    Icon: BookMarked,
    tone: 'cyan',
  },
  {
    title: 'Artifact-centric',
    description:
      'Durable request and result contracts decouple generation from scoring. Videos, geometry, actions, and traces can be inspected or rescored without reloading the model.',
    Icon: Layers3,
    tone: 'yellow',
  },
  {
    title: 'Evidence-aware',
    description:
      'Scorecards keep declared support, operational readiness, and validated performance separate, so a catalog entry or a demo is never mistaken for a benchmark claim.',
    Icon: ClipboardCheck,
    tone: 'red',
  },
] as const;

const capabilities = [
  {
    index: '01',
    title: 'Know what exists',
    description:
      'Manifests expose stable IDs, sources, capabilities, assets, runtime bindings, readiness, and blockers before compute is allocated.',
    href: '/docs/overview/design',
    link: 'See what is included',
    Icon: ScrollText,
  },
  {
    index: '02',
    title: 'Run through shared boundaries',
    description:
      'Pipelines preserve model-native I/O while TUI, CLI, Studio, Python, and MCP reuse the same execution contracts.',
    href: '/docs/overview/design',
    link: 'Understand the design',
    Icon: Network,
  },
  {
    index: '03',
    title: 'Inspect before scoring',
    description:
      'Videos, geometry, actions, trajectories, and traces stay visible on disk and in Studio instead of disappearing inside scripts.',
    href: '/docs/guides/studio',
    link: 'Explore Studio',
    Icon: Eye,
  },
  {
    index: '04',
    title: 'Evidence, not headlines',
    description:
      'Benchmark runners preserve per-sample outcomes, coverage, provenance, blockers, reports, and scorecards.',
    href: '/docs/evaluation',
    link: 'Read about evaluation',
    Icon: ClipboardCheck,
  },
];

const featuredModelIds = [
  'wan2.2',
  'ltx-video',
  'cosmos-predict-2.5',
  'hunyuanvideo-1.5',
  'longcat-video',
  'lingbot-video',
  'bernini',
  'matrix-game-2',
];

const featuredModels: HomeRecipeOption[] = [
  ...featuredModelIds
    .map((id) => modelRecipeIndex.recipes.find((recipe) => recipe.id === id))
    .filter((recipe): recipe is NonNullable<typeof recipe> => Boolean(recipe)),
  ...modelRecipeIndex.recipes.filter(
    (recipe) => recipe.status.group === 'verified' && !featuredModelIds.includes(recipe.id),
  ),
]
  .slice(0, 8)
  .map((recipe) => ({
    id: recipe.id,
    name: recipe.name,
    provider: recipe.provider,
    category: recipe.category,
    tasks: recipe.tasks,
    status: catalogReadinessLabel(recipe.status.group),
    environment: recipe.runtime.environmentName,
    python: recipe.runtime.python,
    cuda: recipe.runtime.cudaLabel,
  }));

const communityLinks = [
  {
    title: 'Join Slack',
    description: 'Real-time help & discussions',
    href: WORLDFOUNDRY_SLACK_INVITE,
    Icon: MessagesSquare,
  },
  {
    title: 'GitHub Issues',
    description: 'Bug reports & feature requests',
    href: WORLDFOUNDRY_GITHUB_ISSUES,
    Icon: CircleDot,
  },
  {
    title: 'GitHub',
    description: 'Source, stars, and contributions',
    href: WORLDFOUNDRY_GITHUB_REPO,
    Icon: MessageCircle,
  },
];

const resourceLinks = [
  { label: 'Blog', text: 'Updates, technical notes, and release highlights.', href: '/blog' },
  { label: 'Events', text: 'Meetups, demos, benchmark sprints, and milestones.', href: '/events' },
  { label: 'OpenEnvision', text: 'The lab organization behind WorldFoundry.', href: '/openenvision' },
  {
    label: 'Awesome World Modeling',
    text: 'Curated papers, models, and resources for world modeling.',
    href: OPENENVISION_AWESOME_WORLD_MODELING,
  },
  {
    label: 'BlogrXiv',
    text: 'Curated index for technical AI research blogs and writing.',
    href: OPENENVISION_BLOGXIV_SITE,
  },
  {
    label: 'WorldFoundry on GitHub',
    text: 'Source, issues, discussions, and releases.',
    href: WORLDFOUNDRY_GITHUB_REPO,
  },
];

const workflowRail = ['Discover', 'Prepare', 'Run', 'Inspect', 'Evaluate'] as const;

function SectionIntro({
  title,
  children,
  align = 'center',
}: {
  title: React.ReactNode;
  children?: React.ReactNode;
  align?: 'center' | 'start';
}) {
  return (
    <header className="px-section-intro" data-align={align}>
      <h2>{title}</h2>
      {children ? <p>{children}</p> : null}
    </header>
  );
}

export default function HomePage() {
  return (
    <main className="px-home">
      {/* Hero: framed prismatic panel with the product window */}
      <section className="px-hero" aria-labelledby="px-hero-title">
        <div className="px-hero-frame">
          <div className="px-hero-bloom" aria-hidden="true" />
          <Grain />
          <div className="px-hero-copy">
            <p className="px-hero-eyebrow">
              <span className="px-mark" aria-hidden="true" />
              WorldFoundry
            </p>
            <h1 id="px-hero-title">Toward a general infrastructure for world intelligence</h1>
            <p className="px-hero-lead">
              Discover, run, inspect, and evaluate world models under one operational definition —
              without flattening them into a single interface.
            </p>
            <ul className="px-hero-checks">
              {pillars.map((pillar) => (
                <li key={pillar.title} data-tone={pillar.tone}>
                  <CheckBold />
                  {pillar.title}
                </li>
              ))}
            </ul>
            <div className="px-hero-actions">
              <PrismButton href="/docs/quickstart">
                Get started
                <ArrowRight aria-hidden="true" size={16} strokeWidth={2} />
              </PrismButton>
              <OutlineButton href="/docs/guides/supported-models">Explore model recipes</OutlineButton>
            </div>
            <p className="px-hero-proof">
              <Link href="/docs/guides/supported-models">
                <strong>{catalogCoverage.modelsTotal}</strong> model entries
              </Link>
              <span aria-hidden="true">·</span>
              <Link href="/docs/evaluation/benchmark-hub">
                <strong>{catalogCoverage.benchmarksTotal}</strong> browsable benchmarks
              </Link>
            </p>
          </div>
          <div className="px-hero-product">
            <PrismRay intensity="hero" className="px-hero-ray" />
            <div className="px-hero-halo" aria-hidden="true" />
            <HeroShowcase video={homeHeroVideo} />
          </div>
        </div>
      </section>

      <LogoCloud title="Integrates and adapts upstream projects from" />

      {/* Configure a run */}
      <section className="px-section" aria-labelledby="px-configure-title">
        <div className="px-container px-split">
          <div className="px-split-copy">
            <Kicker>Try it out</Kicker>
            <h2 id="px-configure-title" className="px-h2">
              Configure a run
            </h2>
            <p className="px-lead">
              Choose a real catalog entry. Runtime and version facts come from the repository
              manifests, and the command stays copyable.
            </p>
            <ArrowLink href="/docs/guides/supported-models">Explore model recipes</ArrowLink>
          </div>
          <div className="px-split-panel">
            <div className="px-panel-wash" aria-hidden="true" />
            <div className="px-configurator">
              <HomeRunConfigurator models={featuredModels} />
            </div>
          </div>
        </div>
      </section>

      {/* Why WorldFoundry: three pillars */}
      <section className="px-section" aria-labelledby="px-pillars-title">
        <div className="px-container">
          <SectionIntro
            align="start"
            title={<span id="px-pillars-title">Commensurability without homogenization.</span>}
          >
            Shared catalogs, durable artifacts, and evidence-aware scorecards — while native inputs,
            execution semantics, and evaluation protocols stay model-specific.
          </SectionIntro>
          <div className="px-steps">
            {pillars.map((pillar, index) => (
              <article
                key={pillar.title}
                className="px-step-card"
                data-tone={pillar.tone}
                style={{ '--px-step-index': index } as React.CSSProperties}
              >
                <div className="px-step-art">
                  <span className="px-step-beam" aria-hidden="true">
                    <span />
                  </span>
                  <IconTile className="px-icon-tile-lg">
                    <pillar.Icon size={26} strokeWidth={1.6} />
                  </IconTile>
                </div>
                <div className="px-step-body">
                  <h3>
                    <span className="px-step-num">{index + 1}</span>
                    {pillar.title}
                  </h3>
                  <p>{pillar.description}</p>
                </div>
              </article>
            ))}
          </div>
          <div className="px-section-action">
            <OutlineButton href="/docs">Read the introduction</OutlineButton>
          </div>
        </div>
      </section>

      {/* Catalog, progression, workflow: the framed stack */}
      <section className="px-section px-section-framed" aria-labelledby="px-stack-title">
        <div className="px-frame">
          <Grain />
          <div className="px-frame-inner">
            <SectionIntro
              title={
                <span id="px-stack-title">
                  {catalogCoverage.modelsTotal} model entries, one operational vocabulary.
                </span>
              }
            >
              Video, 3D/4D, interactive worlds, and embodied systems share manifests and readiness
              signals. Catalog inclusion is not uniform runtime maturity.
            </SectionIntro>

            <article className="px-stack-card">
              <div className="px-stack-head">
                <Kicker>Catalog</Kicker>
              </div>
              <CatalogCoverage />
            </article>

            <Connector label="manifests" />

            <article className="px-stack-card">
              <div className="px-stack-head">
                <Kicker tone="yellow">Capability progression</Kicker>
                <h3>From generation to world intelligence.</h3>
                <p>
                  Visual plausibility is not world modeling. A system exhibits stronger competence as
                  representations become persistent, transitions become intervention-responsive, and
                  predictions become useful in a closed loop.
                </p>
              </div>
              <WorldModelProgression locale="en" />
              <ArrowLink href="/docs#operational-definition">
                Read the operational definition
              </ArrowLink>
            </article>

            <Connector label="artifacts" />

            <article className="px-stack-card">
              <div className="px-stack-head">
                <Kicker tone="red">Five-stage pipeline</Kicker>
                <h3>How it works</h3>
                <p>
                  Every surface follows the same sequence. The artifact is the handoff between model
                  execution and benchmark evaluation.
                </p>
                <ol className="px-rail-steps" aria-label="Pipeline stages">
                  {workflowRail.map((label, index) => (
                    <li key={label}>
                      <span>{String(index + 1).padStart(2, '0')}</span>
                      {label}
                    </li>
                  ))}
                </ol>
              </div>
              <WorldFoundryWorkflow variant="home" />
            </article>
          </div>
        </div>
      </section>

      {/* End to end: four capabilities */}
      <section className="px-section" aria-labelledby="px-capabilities-title">
        <div className="px-container px-capabilities">
          <div className="px-capabilities-copy">
            <Kicker>End to end</Kicker>
            <h2 id="px-capabilities-title" className="px-h2">
              Operate with shared contracts.
            </h2>
            <p className="px-lead">
              Catalog, runtime, workspace, and evaluation share identities and durable outputs.
            </p>
          </div>
          <div className="px-capability-grid">
            {capabilities.map((capability) => (
              <article key={capability.title} className="px-capability">
                <div className="px-capability-top">
                  <IconTile>
                    <capability.Icon size={20} strokeWidth={1.7} />
                  </IconTile>
                  <span className="px-capability-index">{capability.index}</span>
                </div>
                <h3>{capability.title}</h3>
                <p>{capability.description}</p>
                <ArrowLink href={capability.href}>{capability.link}</ArrowLink>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* Community */}
      <section className="px-section" aria-labelledby="px-community-title">
        <div className="px-container">
          <div className="px-community">
            <div className="px-community-copy">
              <Kicker tone="yellow">Everyone welcome</Kicker>
              <h2 id="px-community-title" className="px-h2">
                Got questions? <span>We&apos;re here to help.</span>
              </h2>
              <p className="px-lead">
                Whether you&apos;re just getting started or debugging a complex run, the community is
                open to everyone. No question is too basic.
              </p>
              <ul className="px-community-signals">
                <li>
                  <MessageCircle aria-hidden="true" size={16} strokeWidth={1.8} />
                  Fast &amp; friendly responses
                </li>
                <li>
                  <UserPlus aria-hidden="true" size={16} strokeWidth={1.8} />
                  Active maintainers
                </li>
              </ul>
            </div>
            <div className="px-community-actions">
              <ul className="px-community-links">
                {communityLinks.map(({ title, description, href, Icon }) => (
                  <li key={title}>
                    <a href={href} target="_blank" rel="noreferrer" className="px-spectrum-ring">
                      <IconTile>
                        <Icon size={18} strokeWidth={1.7} />
                      </IconTile>
                      <span>
                        <strong>{title}</strong>
                        <em>{description}</em>
                      </span>
                      <ArrowRight aria-hidden="true" size={16} strokeWidth={1.8} />
                    </a>
                  </li>
                ))}
              </ul>
              <figure className="px-community-qr">
                <img
                  src={withBasePath(WORLDFOUNDRY_WECHAT_QR)}
                  alt="WorldFoundry WeChat community QR code"
                  width={168}
                  height={168}
                />
                <figcaption>
                  <strong>WeChat Community</strong>
                  <span>Scan to join. The QR code is updated periodically if it expires.</span>
                </figcaption>
              </figure>
            </div>
          </div>
        </div>
      </section>

      {/* Resources */}
      <section className="px-section" aria-labelledby="px-resources-title">
        <div className="px-container">
          <SectionIntro title={<span id="px-resources-title">Resources &amp; ecosystem</span>}>
            Notes, events, and sibling projects in the OpenEnvision network.
          </SectionIntro>
          <div className="px-resource-grid">
            {resourceLinks.map((item) => {
              const external = item.href.startsWith('http');
              const body = (
                <>
                  <h3>{item.label}</h3>
                  <p>{item.text}</p>
                  <span className="px-resource-arrow" aria-hidden="true">
                    {external ? (
                      <ArrowUpRight size={16} strokeWidth={1.8} />
                    ) : (
                      <ArrowRight size={16} strokeWidth={1.8} />
                    )}
                  </span>
                </>
              );
              return external ? (
                <a
                  key={item.label}
                  className="px-resource px-spectrum-ring"
                  href={item.href}
                  target="_blank"
                  rel="noreferrer"
                >
                  {body}
                </a>
              ) : (
                <Link key={item.label} className="px-resource px-spectrum-ring" href={item.href}>
                  {body}
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      {/* Closing call to action */}
      <section className="px-section px-cta-section" aria-labelledby="px-cta-title">
        <div className="px-cta">
          <div className="px-cta-glass">
            <div className="px-cta-wash" aria-hidden="true" />
            <Grain />
            <div className="px-cta-copy">
              <h2 id="px-cta-title">World models stay heterogeneous.</h2>
              <p>Shared catalogs, artifacts, and scorecards are what make them comparable.</p>
              <ul className="px-hero-checks">
                {capabilities.slice(1).map((capability, index) => (
                  <li key={capability.title} data-tone={(['cyan', 'yellow', 'red'] as const)[index]}>
                    <CheckBold />
                    {capability.title}
                  </li>
                ))}
              </ul>
              <div className="px-hero-actions">
                <PrismButton href="/docs/quickstart">Get started</PrismButton>
                <OutlineButton href="/docs">Read the introduction</OutlineButton>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
