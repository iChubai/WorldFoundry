import { withBasePath } from '@/lib/site-path';

type StudioGalleryLocale = 'en' | 'zh';

type StudioVisualizerDemo = {
  id: string;
  title: Record<StudioGalleryLocale, string>;
  subtitle: Record<StudioGalleryLocale, string>;
  aliases: Record<StudioGalleryLocale, string>;
  demoUrl?: string;
  port: string;
  artifacts: string;
  command: string;
};

const labels = {
  en: {
    eyebrow: 'Studio visualizer demos',
    overviewCaption: 'WorldFoundry Workspace · Visualizers tab',
    overviewAlt: 'WorldFoundry Workspace visualizers overview',
    port: 'Port',
    aliases: 'Aliases',
    artifacts: 'Artifacts',
    launch: 'Launch',
    demo: 'Live demo',
  },
  zh: {
    eyebrow: 'Studio 可视化演示',
    overviewCaption: 'WorldFoundry Workspace · 可视化面板',
    overviewAlt: 'WorldFoundry Workspace 可视化入口总览',
    port: '端口',
    aliases: '别名',
    artifacts: '产物',
    launch: '启动命令',
    demo: '在线演示',
  },
} satisfies Record<StudioGalleryLocale, Record<string, string>>;

const demos = [
  {
    id: 'world',
    title: {
      en: 'World Rollout Viewer',
      zh: '世界模型分段推演查看器',
    },
    subtitle: {
      en: 'Buffered realtime target for action-conditioned roaming. Controls are sampled continuously while generated chunks are prefetched.',
      zh: '面向实时漫游的缓冲式动作条件推演。前端持续采样控制输入并预取生成片段。',
    },
    aliases: {
      en: 'interactive-world, world-model, world-rollout',
      zh: 'interactive-world, world-model, world-rollout',
    },
    port: '7868',
    artifacts: 'image, video, action_trace',
    command: 'python -m worldfoundry.studio.cli matrix-game-2 --frontend world --port 7868',
  },
  {
    id: 'viser',
    title: {
      en: 'Viser Geometry Viewer',
      zh: 'Viser 几何查看器',
    },
    subtitle: {
      en: 'Point clouds, depth-derived point sets, meshes, camera poses, and trajectories.',
      zh: '查看点云、深度转换点集、网格、相机位姿与运动轨迹。',
    },
    aliases: {
      en: 'viser, geometry, pointcloud',
      zh: 'viser, geometry, pointcloud',
    },
    port: '18590',
    artifacts: 'ply, pcd, xyz, glb, gltf, obj, npz',
    command: 'python -m worldfoundry.studio.cli pi3 --frontend points --asset /path/to/scene.ply',
  },
  {
    id: 'rerun',
    title: {
      en: 'Rerun Timeline Viewer',
      zh: 'Rerun 时间线查看器',
    },
    subtitle: {
      en: '`.rrd` timelines and recordings with synchronized cameras, frames, tracks, and 3D entities.',
      zh: '加载 `.rrd` 时间线与录制文件，同步查看相机、视频帧、轨迹与三维实体。',
    },
    aliases: {
      en: 'rrd',
      zh: 'rrd',
    },
    port: '9876',
    artifacts: 'rrd',
    command: 'python -m worldfoundry.studio.cli vggt --frontend rerun --asset /path/to/recording.rrd',
  },
  {
    id: 'spark',
    title: {
      en: 'Spark Gaussian Splat Viewer',
      zh: 'Spark 3DGS 查看器',
    },
    subtitle: {
      en: '3D Gaussian splats via the in-tree Spark frontend. Live preview uses the Spark.js Hello World butterfly demo.',
      zh: '通过内置 Spark 前端查看三维高斯泼溅（3DGS）场景，在线预览搭载 Spark.js 官方示例。',
    },
    aliases: {
      en: '3dgs, splat',
      zh: '3dgs, splat',
    },
    demoUrl: 'https://sparkjs.dev/examples/#hello-world',
    port: '8765',
    artifacts: 'splat, spz, ksplat, sog, splat-ply',
    command: 'python -m worldfoundry.studio.cli vggt --frontend spark --asset /path/to/scene.splat',
  },
] satisfies StudioVisualizerDemo[];

export function StudioVisualizerGallery({ locale = 'en' }: { locale?: StudioGalleryLocale }) {
  const t = labels[locale];

  return (
    <section className="pi-studio-viz-gallery not-prose" aria-label={t.eyebrow}>
      <figure className="pi-studio-viz-overview">
        <div className="pi-studio-viz-overview-media">
          <img
            src={withBasePath('/images/studio/visualizers/live/workspace-overview.png')}
            alt={t.overviewAlt}
            className="pi-studio-viz-overview-image"
            loading="lazy"
          />
        </div>
        <figcaption className="pi-studio-viz-overview-caption">{t.overviewCaption}</figcaption>
      </figure>

      <ul className="pi-studio-viz-list">
        {demos.map((demo) => (
          <li className="pi-studio-viz-item" key={demo.id}>
            <h3 className="pi-studio-viz-item-title">{demo.title[locale]}</h3>
            <p className="pi-studio-viz-item-desc">{demo.subtitle[locale]}</p>
            <p className="pi-studio-viz-item-facts">
              <span>
                {t.port} {demo.port}
              </span>
              <span>
                {t.artifacts} {demo.artifacts}
              </span>
            </p>
            <p className="pi-studio-viz-item-cmd">
              <span className="pi-studio-viz-card-fact-label">{t.launch}</span>
              <code>{demo.command}</code>
            </p>
            {demo.demoUrl ? (
              <p className="pi-studio-viz-item-demo">
                <a href={demo.demoUrl} rel="noreferrer" target="_blank">
                  {t.demo}
                </a>
              </p>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}
