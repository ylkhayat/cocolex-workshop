import * as React from 'react';
import { Button } from '@/components/ui/button';
import { useRouter } from 'next/navigation';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '@/components/ui/accordion';
import { useExperiments } from '@/hooks/use-experiments';

export function ExperimentsList() {
  const experimentsData = useExperiments();
  const router = useRouter();

  return (
    <Card className="sticky top-0 z-10">
      <CardHeader>
        <CardTitle>Experiments</CardTitle>
        <CardDescription>
          Compare different models and setups on various datasets and splits.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Accordion type="multiple">
          {experimentsData.map((dataset, datasetIndex) => (
            <AccordionItem
              key={dataset.name}
              value={dataset.name}
              className={datasetIndex % 2 === 0 ? 'bg-gray-100' : 'bg-white'}
            >
              <AccordionTrigger>{dataset.name.toUpperCase()}</AccordionTrigger>
              <AccordionContent>
                {dataset.splits?.map((split: any, splitIndex: any) => (
                  <Accordion
                    key={split.name}
                    type="multiple"
                    className={
                      splitIndex % 2 === 0 ? 'bg-gray-50' : 'bg-gray-200'
                    }
                  >
                    <AccordionItem value={split.name}>
                      <AccordionTrigger className="ml-4">
                        {split.name.toUpperCase()}
                      </AccordionTrigger>
                      <AccordionContent>
                        {split.setups?.map((setup: any, setupIndex: any) => {
                          const shortSetup = setup.name
                            .split('_')
                            .map((s: any) => s[0])
                            .join('')
                            .toUpperCase();
                          return (
                            <Accordion
                              key={setup.name}
                              type="multiple"
                              className={
                                setupIndex % 2 === 0
                                  ? 'bg-gray-100'
                                  : 'bg-white'
                              }
                            >
                              <AccordionItem value={setup.name}>
                                <AccordionTrigger className="ml-8">
                                  {shortSetup}
                                </AccordionTrigger>
                                <AccordionContent>
                                  {setup.top_ks?.map(
                                    (topK: any, topKIndex: any) => (
                                      <Accordion
                                        key={topK.name}
                                        type="multiple"
                                        className={
                                          topKIndex % 2 === 0
                                            ? 'bg-gray-50'
                                            : 'bg-gray-200'
                                        }
                                      >
                                        <AccordionItem value={topK.name}>
                                          <AccordionTrigger className="ml-12">
                                            {topK.name.toUpperCase()}
                                          </AccordionTrigger>
                                          <AccordionContent className="space-y-2">
                                            {topK.models?.map(
                                              (model: any, modelIndex: any) => (
                                                <Button
                                                  key={model.name}
                                                  variant="outline"
                                                  className={`ml-16 ${modelIndex % 2 === 0 ? 'bg-gray-100' : 'bg-white'}`}
                                                  onClick={() =>
                                                    router.push(
                                                      `?model=${encodeURIComponent(model.path)}`
                                                    )
                                                  }
                                                >
                                                  {model.name}
                                                </Button>
                                              )
                                            )}
                                          </AccordionContent>
                                        </AccordionItem>
                                      </Accordion>
                                    )
                                  )}
                                </AccordionContent>
                              </AccordionItem>
                            </Accordion>
                          );
                        })}
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                ))}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </CardContent>
    </Card>
  );
}
