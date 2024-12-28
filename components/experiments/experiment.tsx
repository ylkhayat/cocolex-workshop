import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card';

function displayValue(value: any): string {
  if (typeof value === 'number') {
    return value.toFixed(2);
  }
  return String(value);
}

function getBackgroundColor(
  value: number,
  isAlignScore: boolean = false
): string {
  const min = isAlignScore ? 0 : 0;
  const max = isAlignScore ? 100 : 1;
  const percentage = (value - min) / (max - min);
  const red = Math.round(255 * (1 - percentage));
  const green = Math.round(255 * percentage);
  return `rgba(${red}, ${green}, 0, 0.2)`;
}

function renderMetricData(metricData: any, isAlignScore: boolean = false) {
  if (typeof metricData === 'object' && !Array.isArray(metricData)) {
    return Object.entries(metricData).map(([subMetric, val]) => (
      <div
        key={subMetric}
        className="p-1 rounded"
        style={{
          backgroundColor: getBackgroundColor(val as number, isAlignScore)
        }}
      >
        <span className="font-semibold">{subMetric}:</span>{' '}
        {typeof val === 'object'
          ? renderMetricData(val, isAlignScore)
          : displayValue(val)}
      </div>
    ));
  }
  return (
    <div
      className="p-1 rounded"
      style={{
        backgroundColor: getBackgroundColor(metricData as number, isAlignScore)
      }}
    >
      {displayValue(metricData)}
    </div>
  );
}

export function Experiment({ data }: { data: any }) {
  let experimentName = data?.name.split('__')[0].toUpperCase();
  experimentName += ' — ' + data?.meta.params.dataset_percentage * 100;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{experimentName}</CardTitle>
        <CardDescription>
          <h4 className="text-sm font-semibold">Parameters</h4>
          <div className="flex flex-wrap gap-2 text-xs">
            {Object.entries(data.meta.params || {}).map(([key, value]) => (
              <div key={key}>
                <span className="font-semibold">{key}:</span> {String(value)}
              </div>
            ))}
          </div>
          {data.meta.scores && (
            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Scores</h4>
              <div className="grid gap-2 text-xs md:grid-cols-4">
                {Object.entries(data.meta.scores).map(
                  ([metricName, metricData]) => (
                    <div key={metricName} className="space-y-1">
                      <div className="font-semibold">
                        {metricName.toUpperCase()}
                      </div>
                      {renderMetricData(
                        metricData,
                        metricName.toLowerCase() === 'align_score'
                      )}
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <p>Card Content</p>
      </CardContent>
    </Card>
  );
}
