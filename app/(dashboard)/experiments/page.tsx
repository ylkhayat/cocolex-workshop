'use client';

import { useState } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Modal } from '@/components/ui/modal';
import { Sidebar } from '@/components/ui/sidebar';
import experimentsData from '@/experiments.json';

export default function ExperimentsPage() {
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [openCategories, setOpenCategories] = useState({});
  const toggleCategory = (categoryName) => {
    setOpenCategories((prev) => ({
      ...prev,
      [categoryName]: !prev[categoryName]
    }));
  };

  const handleExperimentClick = (experiment) => {
    setSelectedExperiment(experiment);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedExperiment(null);
  };

  return (
    <div className="flex">
      <Sidebar>
        <div className="space-y-2 p-4">
          {experimentsData.map((dataset) => (
            <div key={dataset.name} className="rounded-lg space-y-2">
              <div
                className="cursor-pointer hover:bg-gray-800 px-3 py-2 rounded-lg border border-white"
                onClick={() => toggleCategory(dataset.name)}
              >
                {dataset.name.toUpperCase()}
              </div>
              {openCategories[dataset.name] && (
                <div className="pl-4 space-y-2">
                  {dataset.splits.map((split) => (
                    <div key={split.name} className="rounded-lg space-y-2">
                      <div
                        className="cursor-pointer hover:bg-gray-800 px-3 py-2 rounded-lg border border-white"
                        onClick={() => toggleCategory(split.name)}
                      >
                        {split.name.toUpperCase()}
                      </div>
                      {openCategories[split.name] && (
                        <div className="pl-4 space-y-2">
                          {split.setups.map((setup) => {
                            const shortSetup = setup.name
                              .split('_')
                              .map((s) => s[0])
                              .join('');
                            return (
                              <div
                                key={setup.name}
                                className="rounded-lg space-y-2"
                              >
                                <div
                                  className="cursor-pointer hover:bg-gray-800 px-3 py-2 rounded-lg border border-white"
                                  onClick={() => toggleCategory(setup.name)}
                                >
                                  {shortSetup.toUpperCase()}
                                </div>
                                {openCategories[setup.name] && (
                                  <div className="pl-4 space-y-2">
                                    {setup.topKs.map((topK) => (
                                      <div
                                        key={topK.name}
                                        className="rounded-lg space-y-2"
                                      >
                                        <div
                                          className="cursor-pointer hover:bg-gray-800 px-3 py-2 rounded-lg border border-white"
                                          onClick={() =>
                                            toggleCategory(topK.name)
                                          }
                                        >
                                          {topK.name.toUpperCase()}
                                        </div>
                                        {openCategories[topK.name] && (
                                          <div className="pl-4 space-y-2">
                                            {topK.models.map((model) => (
                                              <div
                                                key={model.name}
                                                className="rounded-lg space-y-2"
                                              >
                                                <div
                                                  className="cursor-pointer hover:bg-gray-800 px-3 py-2 rounded-lg border border-white"
                                                  onClick={() =>
                                                    handleExperimentClick(model)
                                                  }
                                                >
                                                  {model.name}
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </Sidebar>
      <div className="flex-1 p-4">
        <Card>
          <CardHeader>
            <CardTitle>Experiments</CardTitle>
            <CardDescription>
              Select an experiment to view its details.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {selectedExperiment ? (
              <div>
                <h2>{selectedExperiment.name}</h2>
                <pre>{JSON.stringify(selectedExperiment.meta, null, 2)}</pre>
                <pre>{JSON.stringify(selectedExperiment.scores, null, 2)}</pre>
              </div>
            ) : (
              <p>No experiment selected.</p>
            )}
          </CardContent>
        </Card>
      </div>
      <Modal isOpen={isModalOpen} onClose={closeModal}>
        {selectedExperiment && (
          <div>
            <h2>{selectedExperiment.name}</h2>
            <pre>{JSON.stringify(selectedExperiment.experiments, null, 2)}</pre>
          </div>
        )}
      </Modal>
    </div>
  );
}
