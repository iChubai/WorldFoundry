import { DocsZoomImage } from '@/components/docs-zoom-image';
import { withBasePath } from '@/lib/site-path';

const homeTeaserSrc = withBasePath('/cover_4x4_hero-poster.webp');
const docsTeaserSrc = withBasePath('/cover_4x4_hero-poster.webp');

const docsTeaserWidth = 1600;
const docsTeaserHeight = 900;

type TeaserImageProps = {
  alt: string;
  variant?: 'home' | 'docs';
};

export function TeaserImage({ alt, variant = 'docs' }: TeaserImageProps) {
  const src = variant === 'home' ? homeTeaserSrc : docsTeaserSrc;
  const image =
    variant === 'home' ? (
      <img src={src} alt={alt} className="w-full h-auto object-contain" />
    ) : (
      <DocsZoomImage
        alt={alt}
        className="wf-docs-teaser-image"
        height={docsTeaserHeight}
        src={src}
        width={docsTeaserWidth}
      />
    );

  if (variant === 'home') {
    return (
      <div className="w-full overflow-hidden border-2 border-[var(--pi-ink)] shadow-[8px_8px_0_var(--pi-ink)] bg-white mb-20">
        {image}
      </div>
    );
  }

  return image;
}
