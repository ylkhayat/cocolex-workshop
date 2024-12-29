import { promises as fs } from 'fs';
import path from 'path';
import experimentsData from '@/experiments.json';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const staticResponse = true;

  if (staticResponse) {
    try {
      const fullPath = path.join(process.cwd(), 'static-response.json');
      const fileContent = await fs.readFile(fullPath, 'utf8');
      const records = JSON.parse(fileContent);

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
  const url = new URL(request.url);
  const datasetName = url.searchParams.get('dataset');
  const numAnnotations = parseInt(url.searchParams.get('number') || '0', 10);
  if (!datasetName || isNaN(numAnnotations)) {
    return new Response(
      JSON.stringify({ error: 'Dataset and number are required' }),
      {
        status: 400
      }
    );
  }

  try {
    const dataset = experimentsData.find((d) => d.name === datasetName);
    if (!dataset) {
      return new Response(JSON.stringify({ error: 'Dataset not found' }), {
        status: 404
      });
    }

    const split = dataset.splits.find((s) => s.name === 'test');
    const setup = split?.setups.find(
      (s) => s.name === 'bm25_relevant_passages_oracle_documents'
    );
    const topK =
      datasetName === 'cuad' || datasetName === 'obli_qa'
        ? setup?.topKs.find((t) => t.name === '10')
        : setup?.topKs.find((t) => t.name === '3');
    const model = topK?.models.find((m) =>
      m.name.includes('Mistral-7B-Instruct-v0.3')
    );

    const experimentNames = [
      'rag',
      'adacad',
      'knnlm-context-entropy',
      'knnlm-context-plus-entropy'
    ];
    const experiments = model?.experiments.filter((exp) =>
      experimentNames.some((name) => exp.name.startsWith(name))
    );

    const results = {
      adacad: [],
      'knnlm-context-entropy': [],
      'knnlm-context-plus-entropy': [],
      rag: []
    };

    const tests = [];

    const pathsSet = new Set();
    for (const experiment of experiments as any) {
      if (
        experiment.name.includes('knnlm-context-plus-entropy') &&
        datasetName === 'obli_qa' &&
        experiment.results.path.includes('uf-False')
      )
        continue;
      pathsSet.add(experiment.results.path);
    }
    const paths = Array.from(pathsSet);

    if (paths.length !== 4) {
      return new Response(JSON.stringify({ error: 'Expected 4 paths' }), {
        status: 400
      });
    }

    const allRecords = [];
    for (const path of paths) {
      const response = await fetch(
        `${url.origin}/api/experiment?path=${encodeURIComponent(path as any)}`
      );
      const records = await response.json();
      allRecords.push(records);
    }

    const commonDocIds = allRecords.reduce(
      (commonIds, records) => {
        const docIds = records.map((record: any) => record.meta.docid);
        return commonIds.filter((id: any) => docIds.includes(id));
      },
      allRecords[0].map((record: any) => record.meta.docid)
    );
    console.log(commonDocIds.length);
    const filteredDocIds = commonDocIds
      .sort(() => 0.5 - Math.random())
      .slice(0, numAnnotations);

    const filteredRecords = allRecords.map((records) =>
      records.filter((record: any) =>
        filteredDocIds.includes(record.meta.docid)
      )
    );

    const sortedFilteredRecords = filteredRecords.map((records) =>
      records.sort(
        (a: any, b: any) =>
          filteredDocIds.indexOf(a.meta.docid) -
          filteredDocIds.indexOf(b.meta.docid)
      )
    );

    let index = 0;
    for (const records of sortedFilteredRecords) {
      const docIds = records.map((record: any) => record.meta.docid);
      if (
        !filteredDocIds.every((id: string, idx: number) => id === docIds[idx])
      ) {
        return new Response(
          JSON.stringify({ error: 'DocIDs are not sorted consistently' }),
          {
            status: 400
          }
        );
      }
      const experimentKey = (paths[index] as string)
        .split('generations/')[1]
        .split('__')[0] as keyof typeof results;
      console.log('experimentKey', experimentKey);
      results[experimentKey].push(...(records as []));
      index++;
    }

    for (let i = 0; i < numAnnotations; i++) {
      const ragRecord = results['rag'][i] as any;
      const adacadRecord = results['adacad'][i] as any;
      const knnlmContextEntropyRecord = results['knnlm-context-entropy'][
        i
      ] as any;
      const knnlmContextPlusEntropyRecord = results[
        'knnlm-context-plus-entropy'
      ][i] as any;

      const docidInconsistency =
        ragRecord.meta.docid != adacadRecord.meta.docid ||
        ragRecord.meta.docid != knnlmContextEntropyRecord.meta.docid ||
        ragRecord.meta.docid != knnlmContextPlusEntropyRecord.meta.docid;

      const goldTextInconsistency =
        ragRecord.meta.gold_text != adacadRecord.meta.gold_text ||
        ragRecord.meta.gold_text != knnlmContextEntropyRecord.meta.gold_text ||
        ragRecord.meta.gold_text !=
          knnlmContextPlusEntropyRecord.meta.gold_text;
      const previousTextInconsistency =
        ragRecord.meta.previous_text != adacadRecord.meta.previous_text ||
        ragRecord.meta.previous_text !=
          knnlmContextEntropyRecord.meta.previous_text ||
        ragRecord.meta.previous_text !=
          knnlmContextPlusEntropyRecord.meta.previous_text;

      if (
        docidInconsistency ||
        goldTextInconsistency ||
        previousTextInconsistency
      ) {
        return new Response(
          JSON.stringify({ error: 'DocIDs are not consistent' }),
          {
            status: 400
          }
        );
      }
      tests.push({
        docid: ragRecord.meta.docid,
        gold_text: ragRecord.meta.gold_text,
        previous_text: ragRecord.meta.previous_text,
        prompt: ragRecord.meta.prompt,
        generations: {
          rag: ragRecord.gen,
          adacad: adacadRecord.gen,
          knnlm_context_entropy: knnlmContextEntropyRecord.gen,
          knnlm_context_entropy_plus: knnlmContextPlusEntropyRecord.gen
        }
      });
    }
    return new Response(JSON.stringify(tests), {
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Error processing request:', error);
    return new Response(JSON.stringify({ error: 'Error processing request' }), {
      status: 500
    });
  }
}
