'use client';

import { useState } from 'react';
import { Modal } from '@/components/ui/modal';
import experimentsData from '@/experiments.json';
import { ExperimentsList } from '@/components/experiments/experiments-list';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator
} from '@/components/ui/breadcrumb';
import { Separator } from '@/components/ui/separator';

import { useSearchParams } from 'next/navigation';
import { useMemo } from 'react';
import React from 'react';
import { Experiment } from '@/components/experiments/experiment';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog';
import { ExperimentSidebar } from '@/components/experiments/experiment-sidebar';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '@/components/ui/accordion';
import { SelectedRecord } from '@/components/experiments/selected-record';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

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
  return path.split('/').slice(-5); // Example: dataset -> split -> setup -> topK -> model
}

export default function Page() {
  const searchParams = useSearchParams();
  const modelPath = searchParams.get('model');
  const selectedModel = useMemo(
    () => (modelPath ? findModelFromPath(experimentsData, modelPath) : null),
    [modelPath]
  );

  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [selectedRecord, setSelectedRecord] = useState(null);

  const onDialogChange = (isOpen) => {
    setIsDialogOpen(isOpen);
    if (!isOpen) {
      setSelectedExperiment(null);
      setSelectedRecord(null);
    }
  };
  const openDialog = (experiment) => {
    setSelectedExperiment(experiment);
    setIsDialogOpen(true);
  };

  const breadcrumbItems = useMemo(
    () => (modelPath ? buildBreadcrumb(modelPath) : []),
    [modelPath]
  );

  return (
    <>
      <div className="flex flex-1 flex-row gap-4 p-4">
        <div className="w-1/4">
          <ExperimentsList />
        </div>
        <div className="flex-1 min-h-[100vh] rounded-xl md:min-h-min">
          <div className="space-y-6">
            {selectedModel?.experiments?.length ? (
              selectedModel.experiments.map((exp) => (
                <div key={exp.name} onClick={() => openDialog(exp)}>
                  <Experiment data={exp} />
                </div>
              ))
            ) : (
              <p>No experiments found.</p>
            )}
          </div>
        </div>
      </div>
      <Dialog open={isDialogOpen} onOpenChange={onDialogChange}>
        <DialogContent className="w-4/5 p-0">
          <div
            style={{
              display: 'flex',
              height: 'calc(100vh - 4rem)',
              flexDirection: 'row',
              padding: '1rem'
            }}
          >
            <div className="w-1/4">
              <ExperimentSidebar
                experiment={selectedExperiment}
                selectRecord={setSelectedRecord}
              />
            </div>
            <div
              style={{
                padding: '1rem',
                overflowY: 'auto'
              }}
            >
              {selectedExperiment && (
                <div>
                  <DialogHeader>
                    <DialogTitle>{selectedExperiment.name}</DialogTitle>
                  </DialogHeader>
                  <Accordion type="single" collapsible>
                    <AccordionItem value="item-1">
                      <AccordionTrigger>Show Meta?</AccordionTrigger>
                      <AccordionContent>
                        <Experiment data={selectedExperiment} />
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </div>
              )}
              {selectedRecord && <SelectedRecord record={selectedRecord} />}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
