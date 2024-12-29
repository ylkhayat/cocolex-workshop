import { NextResponse } from 'next/server';
import axios from 'axios';
import { asyncBufferFromUrl, parquetRead } from 'hyparquet';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const dataset = searchParams.get('dataset');
    const split = searchParams.get('split');
    const docidsArray = searchParams.get('docids');
    const docids = docidsArray ? docidsArray.split(',') : [];

    if (!dataset || !split || docids.length === 0) {
      return NextResponse.json(
        { error: 'Missing required parameters: dataset, split, docids' },
        { status: 400 }
      );
    }

    const datasetName = dataset.toUpperCase();
    const splitPrefix = split === 'train' ? 'train' : 'test';
    const baseUrl = `https://datasets-server.huggingface.co/parquet?dataset=ylkhayat/${datasetName}-generation-workshop`;
    const listResponse = await axios.get(baseUrl);
    const files = listResponse.data.parquet_files
      .filter((obj) => obj.split === splitPrefix && obj.config === 'BRPOD')
      .map((obj) => obj.url);

    if (!files || files.length === 0) {
      return NextResponse.json(
        { error: 'No files found for the specified split.' },
        { status: 404 }
      );
    }

    let columns = ['docid', 'citations'];
    if (['clerc', 'echr_qa'].includes(dataset)) {
      columns.push('top_10_passages');
    } else if (['obli_qa', 'cuad', 'oal_qa'].includes(dataset)) {
      columns.push('top_k_passages');
    }

    let cummulativeData = [];
    for (const file of files) {
      await parquetRead({
        file: await asyncBufferFromUrl({ url: file }),
        columns,
        onComplete: (data) => {
          cummulativeData = [...cummulativeData, ...data];
        }
      });
    }
    const filteredData = cummulativeData.filter((row) =>
      docids.includes(row[0])
    );
    const finalData = filteredData.map((row) => ({
      docid: row[0],
      citations: row[1],
      top_k_passages: row[2]
    }));

    if (finalData.length !== docids.length) {
      return NextResponse.json(
        { error: 'Some docids were not found in the dataset.' },
        { status: 404 }
      );
    }
    return NextResponse.json(finalData, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
