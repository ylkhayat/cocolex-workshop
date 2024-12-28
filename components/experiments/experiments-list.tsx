import * as React from 'react';
import { Button } from '@/components/ui/button';
import experimentsData from '@/experiments.json';
import { useRouter } from 'next/navigation';

export function ExperimentsList() {
  const router = useRouter();

  return (
    <div className="p-4">
      {experimentsData.map((dataset) => (
        <div key={dataset.name} className="mb-4">
          <h3 className="font-semibold">{dataset.name.toUpperCase()}</h3>
          {dataset.splits?.map((split) => (
            <div key={split.name} className="ml-4">
              <h4 className="font-semibold">{split.name.toUpperCase()}</h4>
              {split.setups?.map((setup) => {
                const shortSetup = setup.name
                  .split('_')
                  .map((s) => s[0])
                  .join('')
                  .toUpperCase();
                return (
                  <div key={setup.name} className="ml-4">
                    <h5 className="font-semibold">{shortSetup}</h5>
                    {setup.topKs?.map((topK) => (
                      <div key={topK.name} className="ml-4">
                        <h6 className="font-semibold">
                          {topK.name.toUpperCase()}
                        </h6>
                        {topK.models?.map((model) => (
                          <div key={model.name} className="ml-4">
                            <Button
                              variant="outline"
                              onClick={() =>
                                router.push(
                                  `?model=${encodeURIComponent(model.path)}`
                                )
                              }
                            >
                              {model.name}
                            </Button>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
