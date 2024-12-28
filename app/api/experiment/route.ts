import { promises as fs } from 'fs';
import path from 'path';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const url = new URL(request.url);
  const filePath = url.searchParams.get('path');

  if (!filePath) {
    return new Response(JSON.stringify({ error: 'Path is required' }), {
      status: 400
    });
  }

  try {
    const fullPath = path.join(process.cwd(), filePath);
    const fileContent = await fs.readFile(fullPath, 'utf8');
    const records = fileContent
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line));

    return new Response(JSON.stringify(records), {
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Error reading JSONL file:', error);
    return new Response(JSON.stringify({ error: 'Error reading file' }), {
      status: 500
    });
  }
}
