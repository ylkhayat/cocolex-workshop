'use client';

import { useEffect, useState } from 'react';
import { useForm, Controller, useWatch } from 'react-hook-form';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
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
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table';
import { useExperiments } from '@/components/hooks/use-experiments';
import { ScrollArea } from '@/components/ui/scroll-area';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog';

type FormValues = {
  id: number | null;
  dataset: string;
  numberOfAnnotations: number;
  username: string;
  evaluations: Record<string, Record<string, Record<string, number>>>;
  tests: {
    docid: string;
    gold_text: string;
    previous_text: string;
    generations: Record<string, string>;
  }[];
};

export default function AnnotatePage() {
  const experimentsData = useExperiments();

  const {
    handleSubmit,
    control,
    reset,
    formState: { errors }
  } = useForm<FormValues>({
    defaultValues: {
      id: null,
      dataset: 'echr_qa',
      numberOfAnnotations: 5,
      username: 'lawyer',
      evaluations: {},
      tests: []
    }
  });
  const [savedAnnotations, setSavedAnnotations] = useState<
    {
      id: number;
      dataset: string;
      numberOfAnnotations: number;
      username: string;
      evaluations: Record<string, Record<string, Record<string, number>>>;
      tests: {
        docid: string;
        gold_text: string;
        previous_text: string;
        citations: string[][];
        top_k_passages: string[];
        generations: Record<string, string>;
      }[];
    }[]
  >([]);
  const [loading, setLoading] = useState(false);
  const [id, dataset, numberOfAnnotations, username, evaluations, tests] =
    useWatch({
      control,
      name: [
        'id',
        'dataset',
        'numberOfAnnotations',
        'username',
        'evaluations',
        'tests'
      ]
    });

  const [selectedTest, setSelectedTest] = useState<{
    docid: string;
    gold_text: string;
    previous_text: string;
    citations: string[][];
    top_k_passages: string[];
    generations: Record<string, string>;
  } | null>(null);

  const onSubmit = async (data: FormValues) => {
    const isValid = tests.every((test) => {
      const evaluations = data.evaluations[test.docid];
      return (
        evaluations &&
        Object.values(evaluations).every((evals) => {
          return evals.fluency && evals.correctness && evals.faithfulness;
        })
      );
    });

    if (!isValid) {
      alert('Please complete the evaluations for all tests.');
      return;
    }

    setLoading(true);
    try {
      const url = data.id ? `/api/annotations/${data.id}` : '/api/annotations';
      const method = data.id ? 'PUT' : 'POST';

      await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });
      reset();
      alert('Annotation saved successfully!');
    } catch (error) {
      console.error('Error saving annotation:', error);
      alert('Failed to save annotation.');
    } finally {
      setLoading(false);
    }
    fetchSavedAnnotations();
  };

  useEffect(() => {
    if (dataset && numberOfAnnotations > 0 && tests.length === 0) {
      setLoading(true);
      fetch(`/api/tests?dataset=${dataset}&number=${numberOfAnnotations}`)
        .then((response) => response.json())
        .then((data) => {
          reset({
            dataset,
            numberOfAnnotations,
            username,
            evaluations: {},
            tests: data
          });
        })
        .finally(() => setLoading(false));
    }
  }, [dataset, numberOfAnnotations, tests, reset]);

  const fetchSavedAnnotations = () => {
    fetch(`/api/annotations`)
      .then((response) => {
        return response.json();
      })
      .then((data) => setSavedAnnotations(data));
  };
  useEffect(() => {
    fetchSavedAnnotations();
  }, []);

  const datasetPicker = (
    <Card>
      <CardHeader>
        <CardTitle>Dataset</CardTitle>
        <CardDescription>Select a dataset to annotate.</CardDescription>
      </CardHeader>
      <CardContent>
        <Controller
          name="dataset"
          control={control}
          defaultValue=""
          rules={{ required: 'Dataset is required' }}
          render={({ field }) => (
            <Select {...field}>
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
          )}
        />
        {errors.dataset && (
          <p className="text-red-500">{errors.dataset.message}</p>
        )}
      </CardContent>
    </Card>
  );

  const numberOfAnnotationsOptions = [
    { label: '5', value: '5' },
    { label: '10', value: '10' },
    { label: '25', value: '25' },
    { label: '40', value: '40' },
    { label: '50', value: '50' }
  ];
  const numberOfAnnotationsPicker = (
    <Card>
      <CardHeader>
        <CardTitle>Number of Annotations: {numberOfAnnotations}</CardTitle>
        <CardDescription>
          Select the number of tests to display.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Controller
          name="numberOfAnnotations"
          control={control}
          defaultValue={-1}
          rules={{ required: 'Number of tests is required' }}
          render={({ field }) => (
            <ToggleGroup
              type="single"
              className="w-[180px]"
              onValueChange={(value) => field.onChange(value)}
              value={`${field.value}`}
            >
              {numberOfAnnotationsOptions.map((option) => (
                <ToggleGroupItem key={option.value} value={option.value}>
                  {option.label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          )}
        />
        {errors.numberOfAnnotations && (
          <p className="text-red-500">{errors.numberOfAnnotations.message}</p>
        )}
      </CardContent>
    </Card>
  );

  const testsList = (
    <div className="w-1/8 border-r p-4">
      <h3 className="text-md font-semibold">Tests</h3>
      <ul className="grid grid-cols-2 gap-2">
        {tests?.map((test, index) => {
          const evaluationKeys = ['fluency', 'correctness', 'faithfulness'];
          const evaluationsForTest = evaluations?.[test.docid] || {};
          const completedKeys = Object.keys(evaluationsForTest).filter((key) =>
            evaluationKeys.every(
              (evalKey) => evaluationsForTest[key]?.[evalKey]
            )
          ).length;
          const completionPercentage = (completedKeys / 4) * 100;
          return (
            <li key={index} className="mb-2">
              <Button
                variant="outline"
                onClick={(e) => {
                  e.preventDefault();
                  setSelectedTest(test as any);
                }}
                className={
                  selectedTest?.docid === test.docid ? 'bg-gray-200' : ''
                }
              >
                Test {index + 1} ({completionPercentage}%)
              </Button>
            </li>
          );
        })}
      </ul>
    </div>
  );

  const savedAnnotationsList = (
    <div className="w-full p-4">
      <h3 className="text-md font-semibold">Saved Annotations</h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Id</TableHead>
            <TableHead>Annotator</TableHead>
            <TableHead>Dataset</TableHead>
            <TableHead>Number Of Annotations</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {savedAnnotations.map((annotation, index) => (
            <TableRow key={index}>
              <TableCell>{annotation.id}</TableCell>
              <TableCell>{annotation.username}</TableCell>
              <TableCell>{annotation.dataset}</TableCell>
              <TableCell>{annotation.numberOfAnnotations}</TableCell>
              <TableCell>
                <Button
                  variant="outline"
                  onClick={(e) => {
                    e.preventDefault();
                    reset(annotation);
                  }}
                >
                  View
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
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
                  Rate the fluency of the generated text (1: Least fluent, 5:
                  Most fluent) based on grammatical correctness.
                </p>
                <Controller
                  key={`evaluations.${selectedTest?.docid}.${key}.fluency`}
                  name={`evaluations.${selectedTest?.docid}.${key}.fluency`}
                  control={control}
                  render={({ field }) => (
                    <ToggleGroup
                      onValueChange={(value) => field.onChange(value)}
                      value={`${field.value}`}
                      type="single"
                      className="w-full"
                    >
                      {[1, 2, 3, 4, 5].map((value) => (
                        <ToggleGroupItem key={value} value={value.toString()}>
                          {value}
                        </ToggleGroupItem>
                      ))}
                    </ToggleGroup>
                  )}
                />
              </div>
              <div className="mt-4">
                <h4 className="font-semibold">Correctness</h4>
                <p className="text-xs">
                  Rate the correctness of the generated text; how aligned is the
                  following text with respect to the given gold text.
                </p>
                <Controller
                  key={`evaluations.${selectedTest?.docid}.${key}.correctness`}
                  name={`evaluations.${selectedTest?.docid}.${key}.correctness`}
                  control={control}
                  render={({ field }) => (
                    <ToggleGroup
                      onValueChange={(value) => field.onChange(value)}
                      value={`${field.value}`}
                      type="single"
                      className="w-full"
                    >
                      {[1, 2, 3, 4, 5].map((value) => (
                        <ToggleGroupItem key={value} value={value.toString()}>
                          {value}
                        </ToggleGroupItem>
                      ))}
                    </ToggleGroup>
                  )}
                />
              </div>
              <div className="mt-4">
                <h4 className="font-semibold">Faithfulness</h4>
                <p className="text-xs">
                  Rate the faithfulness of the generated text; how well does the
                  generated text capture the information from the given passages
                  and citations.
                </p>
                <Controller
                  key={`evaluations.${selectedTest?.docid}.${key}.faithfulness`}
                  name={`evaluations.${selectedTest?.docid}.${key}.faithfulness`}
                  control={control}
                  render={({ field }) => (
                    <ToggleGroup
                      onValueChange={(value) => field.onChange(value)}
                      value={`${field.value}`}
                      type="single"
                      className="w-full"
                    >
                      {[1, 2, 3, 4, 5].map((value) => (
                        <ToggleGroupItem key={value} value={value.toString()}>
                          {value}
                        </ToggleGroupItem>
                      ))}
                    </ToggleGroup>
                  )}
                />
              </div>
            </TabsContent>
          )
        )}
      </Tabs>
    </div>
  );
  const annotatorUsername = (
    <Card>
      <CardHeader>
        <CardTitle>Username</CardTitle>
        <CardDescription>Enter the username of the annotator.</CardDescription>
      </CardHeader>
      <CardContent>
        <Controller
          name="username"
          control={control}
          defaultValue=""
          rules={{ required: 'Username is required' }}
          render={({ field }) => <Input {...field} />}
        />
        {errors.username && (
          <p className="text-red-500">{errors.username.message}</p>
        )}
      </CardContent>
    </Card>
  );

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Accordion type="single" collapsible>
        <AccordionItem value="dataset">
          <AccordionTrigger>Dataset: {dataset}</AccordionTrigger>
          <AccordionContent>{datasetPicker}</AccordionContent>
        </AccordionItem>
        {dataset && (
          <AccordionItem value="numberOfAnnotations">
            <AccordionTrigger>
              Number of Annotations: {numberOfAnnotations}
            </AccordionTrigger>
            <AccordionContent>{numberOfAnnotationsPicker}</AccordionContent>
          </AccordionItem>
        )}
      </Accordion>
      {annotatorUsername}
      <Card className="mt-4 mb-4">
        <CardHeader>
          <CardTitle>Previously Saved Annotations</CardTitle>
        </CardHeader>
        <CardContent>{savedAnnotationsList}</CardContent>
      </Card>
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <p className="mr-2">Loading tests...</p>
          <Loader className="animate-spin" />
        </div>
      ) : (
        <Card>
          <CardContent>
            <div className="mt-8">
              <div className="flex justify-between items-center mb-4">
                {id && (
                  <>
                    <p className="text-lg font-semibold">
                      Editing annotation with id: {id}
                    </p>
                    <Button
                      onClick={(e) => {
                        e.preventDefault();
                        reset({
                          id: null,
                          dataset: 'echr_qa',
                          numberOfAnnotations: 5,
                          username: 'lawyer',
                          evaluations: {},
                          tests: []
                        });
                      }}
                    >
                      New Annotation
                    </Button>
                  </>
                )}
              </div>

              <div className="flex">
                {testsList}
                <div className="w-1/2 p-4">
                  <div className="mb-4">
                    <h3 className="text-md font-semibold">Previous Text</h3>
                    <p className="text-sm">
                      {selectedTest?.previous_text || ''}
                    </p>
                  </div>

                  <div className="mb-20">
                    <h3 className="text-md font-semibold mb-4">Gold Text</h3>
                    <p className="text-sm">{goldText}</p>
                  </div>
                  <div className="mb-4">
                    <hr className="border-t border-gray-300" />
                  </div>

                  <div className="mb-4">
                    <h3 className="text-md font-semibold">Passages</h3>
                    <p className="text-sm mb-4">
                      Select a passage to view more information.
                    </p>
                    <Tabs>
                      <TabsList>
                        {selectedTest?.top_k_passages
                          .slice(0, 3)
                          .map((passage, index) => (
                            <TabsTrigger
                              key={index}
                              value={String.fromCharCode(65 + index)}
                            >
                              {passage.split('\n')[0]}
                            </TabsTrigger>
                          ))}
                      </TabsList>
                      {selectedTest?.top_k_passages.map((passage, index) => (
                        <TabsContent
                          key={index}
                          value={String.fromCharCode(65 + index)}
                        >
                          <pre className="text-sm whitespace-pre-wrap">
                            {passage.split('\n')[1]}
                          </pre>
                        </TabsContent>
                      ))}
                    </Tabs>
                  </div>

                  <div className="mb-4">
                    <h3 className="text-md font-semibold">Citations</h3>
                    <p className="text-sm mb-4">
                      Select a citation to view more information.
                    </p>

                    {selectedTest?.citations.map((citation, index) => (
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="outline">{citation[0]}</Button>
                        </DialogTrigger>
                        <DialogContent className="w-4/5 h-4/5">
                          <DialogHeader>
                            <DialogTitle>{citation[0]}</DialogTitle>
                          </DialogHeader>
                          <ScrollArea className="rounded-md border">
                            <pre className="text-sm whitespace-pre-wrap">
                              {citation[1]}
                            </pre>
                          </ScrollArea>
                        </DialogContent>
                      </Dialog>
                    ))}
                  </div>
                </div>

                {generatedText}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      <Button type="submit" className="mt-4">
        Submit Annotation
      </Button>
    </form>
  );
}
