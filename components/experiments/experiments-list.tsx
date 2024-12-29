import * as React from 'react';
import { Button } from '@/components/ui/button';
import experimentsData from '@/experiments.json';
import { useRouter } from 'next/navigation';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card';

export function ExperimentsList() {
  const router = useRouter();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Experiments</CardTitle>
        <CardDescription>
          Compare different models and setups on various datasets and splits.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {experimentsData.map((dataset) => (
            <div key={dataset.name} className="p-4 border rounded-lg shadow-sm">
              <h2 className="text-xl font-bold">
                {dataset.name.toUpperCase()}
              </h2>
              <div className="space-y-4 ml-4">
                {dataset.splits?.map((split) => (
                  <div key={split.name} className="p-2 border rounded-lg">
                    <h3 className="text-lg font-semibold">
                      {split.name.toUpperCase()}
                    </h3>
                    <div className="space-y-2 ml-4">
                      {split.setups?.map((setup) => {
                        const shortSetup = setup.name
                          .split('_')
                          .map((s) => s[0])
                          .join('')
                          .toUpperCase();
                        return (
                          <div
                            key={setup.name}
                            className="p-2 border rounded-lg"
                          >
                            <h4 className="text-md font-semibold">
                              {shortSetup}
                            </h4>
                            <div className="space-y-2 ml-4">
                              {setup.topKs?.map((topK) => (
                                <div
                                  key={topK.name}
                                  className="p-2 border rounded-lg"
                                >
                                  <h5 className="text-sm font-semibold">
                                    {topK.name.toUpperCase()}
                                  </h5>
                                  <div className="space-y-2 ml-4">
                                    {topK.models?.map((model) => (
                                      <Button
                                        key={model.name}
                                        variant="outline"
                                        onClick={() =>
                                          router.push(
                                            `?model=${encodeURIComponent(model.path)}`
                                          )
                                        }
                                      >
                                        {model.name}
                                      </Button>
                                    ))}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
