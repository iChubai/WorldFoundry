import { withBasePath } from '@/lib/site-path';

// A regular Git blob at its published revision, served from raw.githubusercontent.com
// instead of the Pages artifact.
const HERO_VIDEO_URL =
  'https://raw.githubusercontent.com/OpenEnvision/WorldFoundry/ef8001ec617d9256ee439f2f3452b27b87e3b166/docs/fumadocs/public/cover_4x4_hero.mp4';

export const homeHeroVideo = {
  src: HERO_VIDEO_URL,
  poster: withBasePath('/cover_4x4_hero-poster.webp') ?? '/cover_4x4_hero-poster.webp',
};
