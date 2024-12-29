import path from 'path';

export const dynamic = 'force-dynamic';

const MAIN_PATH =
  'https://github.com/ylkhayat/cocolex-basement/raw/refs/heads/main/';
export async function GET(request: Request) {
  const url = new URL(request.url);
  const filePath = url.searchParams.get('path');

  if (!filePath) {
    return new Response(JSON.stringify({ error: 'Path is required' }), {
      status: 400
    });
  }

  try {
    const fullPath = path.join(MAIN_PATH, filePath);
    console.log('Reading file:', fullPath);
    const response = await fetch(fullPath);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const fileContent = await response.text();
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
