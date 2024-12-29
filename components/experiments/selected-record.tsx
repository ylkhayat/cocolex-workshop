export function SelectedRecord({ record }) {
  const cellStyle = {
    overflow: 'auto',
    maxHeight: '50vh'
  };

  return (
    <>
      <p>Record: {record.meta.docid}</p>
      <div className="grid grid-cols-2 gap-4 text-xs">
        <div className="border p-4" style={cellStyle}>
          <h4 className="font-semibold">Previous Text</h4>
          <pre className="whitespace-pre-wrap">{record.meta.previous_text}</pre>
        </div>
        <div className="border p-4" style={cellStyle}>
          <h4 className="font-semibold">Prompt</h4>
          <pre className="whitespace-pre-wrap">{record.meta.prompt}</pre>
        </div>
        <div className="border p-4" style={cellStyle}>
          <h4 className="font-semibold">Gold Text</h4>
          <pre className="whitespace-pre-wrap">{record.meta.gold_text}</pre>
        </div>
        <div className="border p-4" style={cellStyle}>
          <h4 className="font-semibold">Generated Text</h4>
          <pre className="whitespace-pre-wrap">{record.gen}</pre>
        </div>
      </div>
    </>
  );
}
