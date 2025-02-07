import { NextRequest, NextResponse } from 'next/server';

const EXPERIMENTS_URL =
  'https://github.com/ylkhayat/cocolex-workshop/basement/raw/refs/heads/main/experiments.json';

export async function GET(req: NextRequest) {
  try {
    const response = await fetch(EXPERIMENTS_URL);
    if (!response.ok) {
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: (error as any).message },
      { status: 500 }
    );
  }
}
