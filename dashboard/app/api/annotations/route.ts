import { NextRequest, NextResponse } from 'next/server';
import { getAnnotations, saveAnnotation, updateAnnotation } from '@/lib/db';

export async function GET(req: NextRequest) {
  const annotations = await getAnnotations();
  return NextResponse.json(annotations, { status: 200 });
}

export async function POST(req: NextRequest) {
  const annotationData = await req.json();
  try {
    await saveAnnotation(annotationData);
    return NextResponse.json(
      { message: 'Annotation saved successfully' },
      { status: 201 }
    );
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

export function OPTIONS() {
  return NextResponse.json({ error: 'Method not allowed' }, { status: 405 });
}

export async function PUT(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const id = searchParams.get('id');
  const annotationData = await req.json();

  if (typeof id === 'string') {
    try {
      const { id: _, createdAt: __, ...rest } = annotationData;
      await updateAnnotation(id, rest);
      return NextResponse.json(
        { message: 'Annotation updated successfully' },
        { status: 200 }
      );
    } catch (error) {
      console.error('Error updating annotation:', error);
      return NextResponse.json(
        { error: (error as any).message },
        { status: 500 }
      );
    }
  } else {
    return NextResponse.json(
      { error: 'Invalid query parameter' },
      { status: 400 }
    );
  }
}
