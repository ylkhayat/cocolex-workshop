import path from 'path';

export const dynamic = 'force-dynamic';

const MAIN_PATH =
  'https://raw.githubusercontent.com/ylkhayat/cocolex-workshop/basement/refs/heads/main/';
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
    const response = await fetch(fullPath);
    if (!response.ok) {
      throw new Error(`Network response was not ok!`);
    }
    const fileContent = await response.text();

    if (
      fileContent.startsWith('version') ||
      fileContent.includes('old') ||
      fileContent.includes('size')
    ) {
      const versionMatch = fileContent.match(/^version (.+)$/m);
      const oidMatch = fileContent.match(/^oid sha256:(.+)$/m);
      const sizeMatch = fileContent.match(/^size (\d+)$/m);

      if (!versionMatch || !oidMatch || !sizeMatch) {
        return new Response(JSON.stringify({ error: 'Invalid file content' }), {
          status: 400
        });
      }

      const oid = oidMatch[1];
      const size = sizeMatch[1];

      const jsonResponse = {
        operation: 'download',
        transfer: ['basic'],
        objects: [
          {
            oid,
            size: parseInt(size, 10)
          }
        ]
      };
      const curlResponse = await fetch(
        `https://github.com/ylkhayat/cocolex-workshop/basement.git/info/lfs/objects/batch`,
        {
          method: 'POST',
          headers: {
            Accept: 'application/vnd.git-lfs+json',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(jsonResponse)
        }
      );

      if (!curlResponse.ok) {
        throw new Error('Failed to fetch from Git LFS');
      }

      const curlData = await curlResponse.json();
      if (!curlData.objects[0].actions.download.href) {
        throw new Error('Failed to fetch download URL from Git LFS');
      }

      const downloadUrl = curlData.objects[0].actions.download.href;
      const downloadResponse = await fetch(downloadUrl);

      if (!downloadResponse.ok) {
        throw new Error('Failed to fetch from download URL');
      }

      const downloadData = await downloadResponse.text();
      const records = downloadData
        .split('\n')
        .filter((line) => line.trim())
        .map((line) => JSON.parse(line));
      return new Response(JSON.stringify(records), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
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
