import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card';
import { Button } from '../ui/button';
import { Badge } from '@/components/ui/badge';

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
    <Card className=" w-full">
      <CardHeader>
        <CardTitle>{experimentName}</CardTitle>
        <CardDescription></CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 text-xs">
        <div className="flex flex-wrap gap-2 text-xs">
          {Object.entries(data.meta.params || {}).map(([key, value]) => (
            <Badge key={key}>
              {key}: {String(value)}
            </Badge>
          ))}
        </div>
        <div className="grid gap-2 text-xs md:grid-cols-4 margin-top-6">
          {Object.entries(data.meta.scores).map(([metricName, metricData]) => (
            <div key={metricName} className="space-y-1">
              <div className="font-semibold">{metricName.toUpperCase()}</div>
              {renderMetricData(
                metricData,
                metricName.toLowerCase() === 'align_score'
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
