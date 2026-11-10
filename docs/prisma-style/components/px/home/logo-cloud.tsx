import { existsSync } from 'node:fs';
import { join } from 'node:path';

import { FUMADOCS_ROOT } from '@/lib/fumadocs-root';
import logoMap from '@/lib/model-logo-map.json';
import { withBasePath } from '@/lib/site-path';

type Org = { key: string; name: string; abbr: string; src?: string };

/** Organizations behind the most catalog entries, with a logo on disk. */
function topOrganizations(limit: number) {
  const orgs = logoMap.orgs as Record<string, Org>;
  const counts = new Map<string, number>();
  for (const key of Object.values(logoMap.modelLogos as Record<string, string>)) {
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  const seen = new Set<string>();
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([key]) => orgs[key])
    .filter((org): org is Org & { src: string } => {
      if (!org?.src || seen.has(org.src)) return false;
      if (!existsSync(join(FUMADOCS_ROOT, 'public', org.src))) return false;
      seen.add(org.src);
      return true;
    })
    .slice(0, limit);
}

function Track({ orgs, hidden = false }: { orgs: (Org & { src: string })[]; hidden?: boolean }) {
  return (
    <ul className="px-logo-track" aria-hidden={hidden || undefined}>
      {orgs.map((org) => (
        <li key={org.key} className="px-logo-chip">
          <img src={withBasePath(org.src)} alt="" loading="lazy" draggable={false} />
          <span>{org.name.replace(/\s*\(.*\)$/, '')}</span>
        </li>
      ))}
    </ul>
  );
}

/** Marquee of upstream organizations whose models the catalog integrates. */
export function LogoCloud({ title }: { title: string }) {
  const orgs = topOrganizations(18);

  return (
    <section className="px-logos" aria-label={title}>
      <p className="px-logos-title">{title}</p>
      <div className="px-logos-viewport">
        <div className="px-logos-rail">
          <Track orgs={orgs} />
          <Track orgs={orgs} hidden />
        </div>
      </div>
    </section>
  );
}
