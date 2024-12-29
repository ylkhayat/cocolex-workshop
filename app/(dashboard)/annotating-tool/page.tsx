'use client';

import { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import experimentsData from '@/experiments.json';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '@/components/ui/accordion';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import { Loader } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function AnnotatePage() {
  const [formData, setFormData] = useState<{
    dataset: string;
    numberOfAnnotations: number;
    name: string;
    description: string;
    annotation: string;
    evaluations: Record<string, Record<string, Record<string, number>>>;
  }>({
    dataset: 'echr_qa',
    numberOfAnnotations: 25,
    name: '',
    description: '',
    annotation: '',
    evaluations: {}
  });

  const [annotations, setAnnotations] = useState<
    {
      docid: string;
      gold_text: string;
      previous_text: string;
      generations: Record<string, string>;
    }[]
  >([]);

  const [loading, setLoading] = useState(false);
  const [selectedTest, setSelectedTest] = useState<{
    docid: string;
    gold_text: string;
    previous_text: string;
    generations: Record<string, string>;
  } | null>(null);

  const handleChange = (e: any) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = (e: any) => {
    e.preventDefault();
    setAnnotations((prevAnnotations) => ({
      ...prevAnnotations,
      [formData.name]: [
        ...(prevAnnotations[formData.name as any] as any),
        {
          description: formData.description,
          annotation: formData.annotation
        }
      ]
    }));
    setFormData({
      dataset: '',
      numberOfAnnotations: -1,
      name: '',
      description: '',
      annotation: '',
      evaluations: {}
    });
  };

  useEffect(() => {
    if (formData.dataset && formData.numberOfAnnotations > 0) {
      setLoading(true);
      fetch(
        `/api/annotation?dataset=${formData.dataset}&number=${formData.numberOfAnnotations}`
      )
        .then((response) => response.json())
        .then((data) => setAnnotations(data))
        .finally(() => setLoading(false));
    }
  }, [formData.dataset, formData.numberOfAnnotations]);

  const datasetPicker = (
    <Card>
      <CardHeader>
        <CardTitle>Dataset</CardTitle>
        <CardDescription>Select a dataset to annotate.</CardDescription>
      </CardHeader>
      <CardContent>
        <Select
          value={formData.dataset}
          onValueChange={(value) =>
            setFormData((prevFormData) => ({
              ...prevFormData,
              dataset: value
            }))
          }
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select a dataset" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Datasets</SelectLabel>
              {experimentsData.map((dataset) => (
                <SelectItem key={dataset.name} value={dataset.name}>
                  {dataset.name}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </CardContent>
    </Card>
  );

  const numberOfAnnotationsOptions = [
    { label: '25', value: '25' },
    { label: '40', value: '40' },
    { label: '50', value: '50' }
  ];
  const numberOfAnnotationsPicker = (
    <Card>
      <CardHeader>
        <CardTitle>Number of Annotations</CardTitle>
        <CardDescription>
          Select the number of annotations to display.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ToggleGroup
          type="single"
          className="w-[180px]"
          value={formData.numberOfAnnotations.toString()}
          onValueChange={(value) =>
            setFormData((prevFormData) => ({
              ...prevFormData,
              numberOfAnnotations: parseInt(value)
            }))
          }
        >
          {numberOfAnnotationsOptions.map((option) => (
            <ToggleGroupItem key={option.value} value={option.value}>
              {option.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </CardContent>
    </Card>
  );

  const testsList = (
    <div className="w-1/8 border-r p-4">
      <h3 className="text-md font-semibold">Tests</h3>
      <ul className="grid grid-cols-2 gap-2">
        {annotations.map((test, index) => (
          <li key={index} className="mb-2">
            <Button variant="outline" onClick={() => setSelectedTest(test)}>
              Test {index + 1}
            </Button>
          </li>
        ))}
      </ul>
    </div>
  );

  const goldText = selectedTest?.gold_text || '';
  const generatedText = selectedTest?.generations && (
    <div className="w-1/2 border-l p-4">
      <h3 className="text-md font-semibold">Generated Text</h3>
      <Tabs defaultValue="A">
        <TabsList>
          <TabsTrigger value="A">A</TabsTrigger>
          <TabsTrigger value="B">B</TabsTrigger>
          <TabsTrigger value="C">C</TabsTrigger>
          <TabsTrigger value="D">D</TabsTrigger>
        </TabsList>
        {Object.entries(selectedTest?.generations).map(
          ([key, record], index) => (
            <TabsContent key={key} value={String.fromCharCode(65 + index)}>
              <p className="text-sm">{record}</p>
              <div className="mt-4">
                <h4 className="font-semibold">Fluency</h4>
                <p className="text-xs">
                  Rate the fluency of the generated text.
                </p>
                <ToggleGroup
                  type="single"
                  className="w-full"
                  value={
                    formData.evaluations[selectedTest?.docid]?.[
                      key
                    ]?.fluency?.toString() || ''
                  }
                  onValueChange={(value) =>
                    setFormData((prevFormData) => ({
                      ...prevFormData,
                      evaluations: {
                        ...prevFormData.evaluations,
                        [selectedTest?.docid]: {
                          ...prevFormData.evaluations[selectedTest?.docid],
                          [key]: {
                            ...prevFormData.evaluations[selectedTest?.docid]?.[
                              key
                            ],
                            fluency: parseInt(value)
                          }
                        }
                      }
                    }))
                  }
                >
                  {[1, 2, 3, 4, 5].map((value) => (
                    <ToggleGroupItem key={value} value={value.toString()}>
                      {value}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </div>
              <div className="mt-4">
                <h4 className="font-semibold">Correctness</h4>
                <p className="text-xs">
                  Rate the correctness of the generated text.
                </p>
                <ToggleGroup
                  type="single"
                  className="w-full"
                  value={
                    formData.evaluations[selectedTest?.docid]?.[
                      key
                    ]?.correctness?.toString() || ''
                  }
                  onValueChange={(value) =>
                    setFormData((prevFormData) => ({
                      ...prevFormData,
                      evaluations: {
                        ...prevFormData.evaluations,
                        [selectedTest?.docid]: {
                          ...prevFormData.evaluations[selectedTest?.docid],
                          [key]: {
                            ...prevFormData.evaluations[selectedTest?.docid]?.[
                              key
                            ],
                            correctness: parseInt(value)
                          }
                        }
                      }
                    }))
                  }
                >
                  {[1, 2, 3, 4, 5].map((value) => (
                    <ToggleGroupItem key={value} value={value.toString()}>
                      {value}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </div>
              <div className="mt-4">
                <h4 className="font-semibold">Faithfulness</h4>
                <p className="text-xs">
                  Rate the faithfulness of the generated text.
                </p>
                <ToggleGroup
                  type="single"
                  className="w-full"
                  value={
                    formData.evaluations[selectedTest?.docid]?.[
                      key
                    ]?.faithfulness?.toString() || ''
                  }
                  onValueChange={(value) =>
                    setFormData((prevFormData) => ({
                      ...prevFormData,
                      evaluations: {
                        ...prevFormData.evaluations,
                        [selectedTest?.docid]: {
                          ...prevFormData.evaluations[selectedTest?.docid],
                          [key]: {
                            ...prevFormData.evaluations[selectedTest?.docid]?.[
                              key
                            ],
                            faithfulness: parseInt(value)
                          }
                        }
                      }
                    }))
                  }
                >
                  {[1, 2, 3, 4, 5].map((value) => (
                    <ToggleGroupItem key={value} value={value.toString()}>
                      {value}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </div>
            </TabsContent>
          )
        )}
      </Tabs>
    </div>
  );

  return (
    <>
      <Accordion type="single" collapsible>
        <AccordionItem value="dataset">
          <AccordionTrigger>Dataset</AccordionTrigger>
          <AccordionContent>{datasetPicker}</AccordionContent>
        </AccordionItem>
        {formData.dataset && (
          <AccordionItem value="numberOfAnnotations">
            <AccordionTrigger>Number of Annotations</AccordionTrigger>
            <AccordionContent>{numberOfAnnotationsPicker}</AccordionContent>
          </AccordionItem>
        )}
      </Accordion>
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <p className="mr-2">Loading annotations...</p>
          <Loader className="animate-spin" />
        </div>
      ) : (
        <Card>
          <CardContent>
            <div className="mt-8">
              {selectedTest && (
                <p className="text-md font-semibold">
                  Currently Viewing: Test{' '}
                  {annotations.indexOf(selectedTest) + 1}
                </p>
              )}
              <div className="flex">
                {testsList}
                <div className="w-1/2 p-4">
                  <div className="mb-4">
                    <h3 className="text-md font-semibold">Previous Text</h3>
                    <p className="text-sm">
                      {selectedTest?.previous_text || ''}
                    </p>
                  </div>
                  <div>
                    <h3 className="text-md font-semibold">Gold Text</h3>
                    <p className="text-sm">{goldText}</p>
                  </div>
                </div>

                {generatedText}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
}
