import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { notFound } from 'next/navigation';
import { ImageResponse } from 'next/og';

import { defaultLocale, isLocale } from '@/lib/i18n';
import { getPageImage, source } from '@/lib/source';

export const revalidate = false;

const fontDir = join(process.cwd(), 'node_modules/@fontsource/noto-sans-sc/files');
let fontsPromise:
  | Promise<{ data: ArrayBuffer; name: string; style: 'normal'; weight: 400 | 700 }[]>
  | undefined;

function toArrayBuffer(buffer: Buffer): ArrayBuffer {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer;
}

function getFonts() {
  fontsPromise ??= Promise.all([
    readFile(join(fontDir, 'noto-sans-sc-chinese-simplified-400-normal.woff')),
    readFile(join(fontDir, 'noto-sans-sc-chinese-simplified-700-normal.woff')),
  ]).then(([regular, bold]) => [
    { data: toArrayBuffer(regular), name: 'Noto Sans SC', style: 'normal' as const, weight: 400 as const },
    { data: toArrayBuffer(bold), name: 'Noto Sans SC', style: 'normal' as const, weight: 700 as const },
  ]);
  return fontsPromise;
}

const BAND = { cyan: '#01d7e4', yellow: '#f3c306', red: '#f34a60' };

/** Social card: paper ground, the triple-band ray, ink title. */
export async function GET(_req: Request, { params }: RouteContext<'/og/docs/[...slug]'>) {
  const { slug } = await params;
  const slugs = slug.slice(0, -1);
  const maybeLocale = slugs[0];
  const locale = isLocale(maybeLocale) ? maybeLocale : defaultLocale;
  const page = source.getPage(isLocale(maybeLocale) ? slugs.slice(1) : slugs, locale);
  if (!page) notFound();

  return new ImageResponse(
    (
      <div
        style={{
          background: '#f9faf5',
          color: '#151515',
          display: 'flex',
          flexDirection: 'column',
          fontFamily: 'Noto Sans SC',
          height: '100%',
          overflow: 'hidden',
          padding: '64px 72px',
          position: 'relative',
          width: '100%',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            height: 132,
            left: -120,
            position: 'absolute',
            top: 470,
            transform: 'rotate(-16deg)',
            width: 1500,
          }}
        >
          <div style={{ background: BAND.cyan, height: 44, opacity: 0.85 }} />
          <div style={{ background: BAND.yellow, height: 44, opacity: 0.85 }} />
          <div style={{ background: BAND.red, height: 44, opacity: 0.85 }} />
        </div>
        <div style={{ alignItems: 'center', display: 'flex', fontSize: 34, fontWeight: 700, gap: 16 }}>
          <div
            style={{
              background: '#151515',
              borderRadius: 999,
              display: 'flex',
              height: 44,
              width: 44,
            }}
          />
          <span>WorldFoundry</span>
          <span style={{ color: '#909193', fontWeight: 400 }}>/</span>
          <span style={{ color: '#3a3b3c', fontWeight: 400 }}>{locale === 'zh' ? '文档' : 'docs'}</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 22, marginTop: 92, maxWidth: 980 }}>
          <div style={{ display: 'flex', fontSize: 72, fontWeight: 700, lineHeight: 1.08 }}>
            {page.data.title}
          </div>
          {page.data.description ? (
            <div style={{ color: '#646567', display: 'flex', fontSize: 28, lineHeight: 1.4 }}>
              {page.data.description}
            </div>
          ) : null}
        </div>
      </div>
    ),
    { fonts: await getFonts(), width: 1200, height: 630 },
  );
}

export function generateStaticParams() {
  return source.getPages().map((page) => ({ slug: getPageImage(page).segments }));
}
