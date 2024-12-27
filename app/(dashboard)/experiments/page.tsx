import { useState, useEffect } from 'react';
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

  const handleExperimentClick = (experiment) => {
    setSelectedExperiment(experiment);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedExperiment(null);
  };

  return (
    <div className="flex">
      <Sidebar>
        {experimentsData.splits.map((split) => (
          <div key={split.name}>
            <h2>{split.name}</h2>
            {split.setups.map((setup) => (
              <div key={setup.name}>
                <h3>{setup.name}</h3>
                {setup.topKs.map((topK) => (
                  <div key={topK.name}>
                    <h4>{topK.name}</h4>
                    {topK.models.map((model) => (
                      <div key={model.name}>
                        <h5>{model.name}</h5>
                        {model.experiments.map((experiment) => (
                          <Button
                            key={experiment.name}
                            onClick={() => handleExperimentClick(experiment)}
                          >
                            {experiment.name}
                          </Button>
                        ))}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ))}
          </div>
        ))}
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
            <pre>{JSON.stringify(selectedExperiment.results, null, 2)}</pre>
          </div>
        )}
      </Modal>
    </div>
  );
}
