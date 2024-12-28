'use client';

import { useState } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import { Modal } from '@/components/ui/modal';
import experimentsData from '@/experiments.json';
import { AppSidebar } from '@/components/experiments/app-sidebar';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator
} from '@/components/ui/breadcrumb';
import { Separator } from '@/components/ui/separator';
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger
} from '@/components/ui/sidebar';
import { useSearchParams } from 'next/navigation';
import { useMemo } from 'react';
import React from 'react';

function findModelFromPath(data: any[], path: string): any | null {
  const relevantPath = path.split('basement/')[1];
  const [dataset, split, setup, topK, model] = relevantPath.split('/');
  const datasetData = data.find((d) => d.name === dataset);
  if (datasetData) {
    const splitData = datasetData.splits.find((s) => s.name === split);
    if (splitData) {
      const setupData = splitData.setups.find((s) => s.name === setup);
      if (setupData) {
        const topKData = setupData.topKs.find((t) => t.name === topK);
        if (topKData) {
          return topKData.models.find((m) => m.name === model);
        }
      }
    }
  }
  return null;
}

function buildBreadcrumb(path: string) {
  // Split on "/" or otherwise parse to build an array of path segments for the breadcrumb.
  return path.split('/').slice(-5); // Example: dataset -> split -> setup -> topK -> model
}

export default function Page() {
  const searchParams = useSearchParams();
  const modelPath = searchParams.get('model');
  const selectedModel = useMemo(
    () => (modelPath ? findModelFromPath(experimentsData, modelPath) : null),
    [modelPath]
  );

  console.log(selectedModel);

  const breadcrumbItems = useMemo(
    () => (modelPath ? buildBreadcrumb(modelPath) : []),
    [modelPath]
  );

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b">
          <div className="flex items-center gap-2 px-3">
            <SidebarTrigger />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                {breadcrumbItems.map((item, idx) => (
                  <React.Fragment key={idx}>
                    <BreadcrumbItem>
                      <BreadcrumbLink href="#">{item}</BreadcrumbLink>
                    </BreadcrumbItem>
                    {idx < breadcrumbItems.length - 1 && (
                      <BreadcrumbSeparator />
                    )}
                  </React.Fragment>
                ))}
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 md:min-h-min">
            {selectedModel?.experiments?.length ? (
              selectedModel.experiments.map((exp) => (
                <div key={exp.name}>
                  <h3>{exp.name}</h3>
                  {/* Show any meta or results as needed */}
                </div>
              ))
            ) : (
              <p>No experiments found.</p>
            )}
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
