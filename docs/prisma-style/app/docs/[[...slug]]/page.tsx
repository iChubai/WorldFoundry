import { PxDocsPage } from '@/components/px/docs/docs-page';
import { generateDocsMetadata, generateDocsStaticParams } from '@/lib/docs-page-server';
import type { Metadata } from 'next';

export default async function Page(props: PageProps<'/docs/[[...slug]]'>) {
  const params = await props.params;
  return <PxDocsPage slug={params.slug} locale="en" />;
}

export function generateStaticParams() {
  return generateDocsStaticParams('en');
}

export async function generateMetadata(props: PageProps<'/docs/[[...slug]]'>): Promise<Metadata> {
  const params = await props.params;
  return generateDocsMetadata(params.slug, 'en');
}
