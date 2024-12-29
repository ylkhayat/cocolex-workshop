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
import { Input } from '@/components/ui/input';
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
  const [formData, setFormData] = useState({
    dataset: 'obli_qa',
    numberOfAnnotations: 25,
    name: '',
    description: '',
    annotation: ''
  });

  const [annotations, setAnnotations] = useState([]);

  const [loading, setLoading] = useState(false);
  const [selectedTest, setSelectedTest] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setAnnotations((prevAnnotations) => ({
      ...prevAnnotations,
      [formData.name]: [
        ...prevAnnotations[formData.name],
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
      annotation: ''
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
      <ul>
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

  // const selectedTestRecords = {
  //   adacad: annotations.adacad.find(
  //     (record) => record.meta.docid === selectedTest
  //   ),
  //   knnlm_context_entropy: annotations.knnlm_context_entropy.find(
  //     (record) => record.meta.docid === selectedTest
  //   ),
  //   knnlm_context_entropy_plus: annotations.knnlm_context_entropy_plus.find(
  //     (record) => record.meta.docid === selectedTest
  //   ),
  //   rag: annotations.rag.find((record) => record.meta.docid === selectedTest)
  // };

  // const goldText = selectedTestRecords.adacad?.meta.gold_text || '';
  const goldText = selectedTest?.gold_text || '';

  const generatedText = selectedTest?.generations && (
    <div className="w-1/4 border-l p-4">
      <h3 className="text-md font-semibold">Generated Text</h3>
      {Object.entries(selectedTest?.generations).map(([key, record]) => (
        <div key={key} className="mt-2">
          <h4 className="font-semibold">{key.toUpperCase()}</h4>
          <p className="text-sm">{record?.gen}</p>
        </div>
      ))}
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
          <CardHeader>
            <CardTitle>Annotate</CardTitle>
            <CardDescription>Submit your annotations below.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="mt-8">
              <h2 className="text-lg font-medium">Annotations</h2>
              <div className="flex">
                {testsList}
                <div className="w-1/4 p-4">
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
