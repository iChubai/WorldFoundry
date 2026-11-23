import { DocsZoomImage } from '@/components/docs-zoom-image';
import { getModelPaperFigures } from '@/lib/model-arch-diagrams';
import { withBasePath } from '@/lib/site-path';

type Locale = 'en' | 'zh';

const LABEL: Record<Locale, string> = {
  en: 'Paper figures',
  zh: '论文配图',
};

export function ModelPaperFigures({
  modelId,
  locale = 'en',
}: {
  modelId: string;
  locale?: Locale;
}) {
  const figures = getModelPaperFigures(modelId, locale);
  if (!figures.length) return null;

  return (
    <section className="wf-model-paper-figures" aria-label={LABEL[locale]}>
      {figures.map((figure) => (
        <figure key={`${figure.role}:${figure.src}`} className="wf-model-paper-figure">
          <div className="wf-recipe-media is-diagram">
            <DocsZoomImage src={withBasePath(figure.src)} alt={figure.alt} />
          </div>
          {figure.caption ? <figcaption>{figure.caption}</figcaption> : null}
        </figure>
      ))}
    </section>
  );
}
