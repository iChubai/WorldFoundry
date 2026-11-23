import { getModelRecipe } from '@/lib/model-recipes';
import { readerStatusLabel } from '@/lib/model-status-label';

type Locale = 'en' | 'zh';

const copy = {
  en: {
    heading: 'Variants',
    task: 'Task',
    profile: 'Runtime profile',
    binding: 'Pipeline binding',
    status: 'Status',
    empty: 'Not recorded',
  },
  zh: {
    heading: '变体',
    task: '任务',
    profile: '运行时配置',
    binding: 'Pipeline 绑定',
    status: '状态',
    empty: '未记录',
  },
} as const;

const TASK_ACRONYMS: Record<string, string> = {
  '3d': '3D',
  '4d': '4D',
  '3dgs': '3DGS',
  api: 'API',
  i2i: 'I2I',
  i2v: 'I2V',
  jepa: 'JEPA',
  rgb: 'RGB',
  t2i: 'T2I',
  t2v: 'T2V',
  ti2v: 'TI2V',
  umi: 'UMI',
  v2v: 'V2V',
  va: 'VA',
  vam: 'VAM',
  vla: 'VLA',
  vlm: 'VLM',
  wam: 'WAM',
};

function formatVariantTask(task: string, fallback: string) {
  if (!task) return fallback;
  const words = task
    .replace(/[._-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean)
    .map((word) => TASK_ACRONYMS[word.toLowerCase()] ?? word);
  if (!words.length) return fallback;
  const joined = words.join(' ');
  return joined.charAt(0).toUpperCase() + joined.slice(1);
}

export function ModelVariantCards({
  modelId,
  locale = 'en',
}: {
  modelId: string;
  locale?: Locale;
}) {
  const recipe = getModelRecipe(modelId);
  const variants = recipe?.variants ?? [];
  if (variants.length < 2) return null;

  const t = copy[locale];

  return (
    <div className="pi-kv-catalog is-vars not-prose">
      <section className="pi-kv-group">
        <h3>{t.heading}</h3>
        <ul>
          {variants.map((variant) => {
            const showBinding =
              Boolean(variant.pipelineBinding) && variant.pipelineBinding !== variant.runtimeProfile;
            return (
              <li className="pi-kv-row" key={variant.id}>
                <div className="pi-kv-head">
                  <div className="pi-kv-names">
                    <code>{variant.id}</code>
                  </div>
                  <span className="pi-kv-default">
                    <span>{t.status}</span>
                    {readerStatusLabel(variant.status, locale, t.empty)}
                  </span>
                </div>
                <p>{formatVariantTask(variant.task, t.empty)}</p>
                <p className="pi-kv-read">
                  <span>{t.profile}</span>
                  <code>{variant.runtimeProfile || t.empty}</code>
                </p>
                {showBinding ? (
                  <p className="pi-kv-read">
                    <span>{t.binding}</span>
                    <code>{variant.pipelineBinding}</code>
                  </p>
                ) : null}
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
