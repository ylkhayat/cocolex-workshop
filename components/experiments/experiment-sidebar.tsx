import * as React from 'react';
import { Button } from '@/components/ui/button';

export function ExperimentSidebar({ experiment, selectRecord }: any) {
  const resultsPath = experiment?.results?.path;
  const [loading, setLoading] = React.useState(true);
  const [experimentRecords, setExperimentRecords] = React.useState(null);

  React.useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch(
          `/api/experiment?path=${encodeURIComponent(resultsPath)}`
        );
        const records = await response.json();
        setExperimentRecords(records);
      } catch (error) {
        console.error('Error loading JSONL data:', error);
      } finally {
        setLoading(false);
      }
    }
    if (resultsPath) {
      fetchData();
    }
  }, [resultsPath]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!experimentRecords) {
    return <div>No data available</div>;
  }

  return (
    <div
      style={{
        display: 'flex',
        height: 'calc(100vh - 7rem)',
        flexDirection: 'column',
        padding: '1rem'
      }}
    >
      <p>Experiment Records</p>
      <div style={{ overflowY: 'auto' }}>
        <ul style={{ gap: '2px' }}>
          {(experimentRecords as any).map((record: any, index: number) => (
            <li key={index}>
              <Button variant="outline" onClick={() => selectRecord(record)}>
                {`Record ${index + 1}: ${record.meta.docid}`}
              </Button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
