'use client';

import { useEffect, useId, useState } from 'react';
import {
  useForm,
  Controller,
  useWatch,
  useFormContext,
  FormProvider
} from 'react-hook-form';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
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
import { useToast } from '@/hooks/use-toast';
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog';
import { DBConfig, STORE_NAME } from './db-config';

import { initDB, useIndexedDB } from 'react-indexed-db-hook';

initDB(DBConfig);

type Annotation = {
  id: number;
  dataset: string;
  numberOfAnnotations: number;
  username: string;
  evaluations: Record<string, Record<string, Record<string, number>>>;
  mapping: Record<
    string,
    Record<'rag' | 'adacad' | 'knnlm_context_entropy', 'A' | 'B' | 'C'>
  >;
  tests: {
    docid: string;
    gold_text: string;
    previous_text: string;
    citations: string[][];
    top_k_passages: string[];
    generations: Record<string, string>;
  }[];
};

type Test = {
  docid: string;
  gold_text: string;
  previous_text: string;
  citations: string[][];
  top_k_passages: string[];
  generations: Record<string, string>;
};

type FormValues = {
  id: number | null;
  dataset: string;
  numberOfAnnotations: number;
  username: string;
  password: string;
  evaluations: Record<string, Record<string, Record<string, number>>>;
  mapping: Record<
    string,
    Record<'rag' | 'adacad' | 'knnlm_context_entropy', 'A' | 'B' | 'C'>
  >;
  tests: Test[];
};

const SECRET_WORDS = 'awetos-tellingly-wakf';

const AnnotatePageInner = () => {
  const {
    handleSubmit,
    control,
    reset,
    formState: { errors }
  } = useFormContext<FormValues>();

  const [
    id,
    dataset,
    numberOfAnnotations,
    username,
    password,
    mapping,
    evaluations,
    tests
  ] = useWatch({
    control,
    name: [
      'id',
      'dataset',
      'numberOfAnnotations',
      'username',
      'password',
      'mapping',
      'evaluations',
      'tests'
    ]
  });
  const { toast } = useToast();
  const { update, deleteRecord } = useIndexedDB(STORE_NAME);
  const experimentsData = useExperiments();
  const [savedAnnotations, setSavedAnnotations] = useState<Annotation[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<{
    annotation: Annotation | null;
    test: Test | null;
  }>({
    annotation: null,
    test: null
  });

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
      toast({
        title: 'Error',
        description: 'Please complete the evaluations for all tests.',
        duration: 5000
      });
      return;
    }

    setLoading(true);
    try {
      const url =
        data.id !== 0 && data.id
          ? `/api/annotations?id=${data.id}`
          : '/api/annotations';
      const method = data.id ? 'PUT' : 'POST';
      await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });
      toast({
        title: 'Success',
        description: 'Annotation saved successfully!',
        duration: 5000
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to save annotation. Please try again.',
        duration: 5000
      });
    } finally {
      setLoading(false);
    }
    fetchSavedAnnotations();
  };

  const fetchTests = async (d: any, n: any) => {
    setLoading(true);
    fetch(`/api/tests?dataset=${d}&number=${n}`)
      .then((response) => response.json())
      .then((data: Test[]) => {
        const currentMapping = data.reduce((acc, test) => {
          const randomMappings = ['A', 'B', 'C'].sort(
            () => Math.random() - 0.5
          );
          const [rag, adacad, knnlm_context_entropy] = randomMappings;
          return {
            ...acc,
            [test.docid]: {
              rag,
              adacad,
              knnlm_context_entropy
            }
          };
        }, {});
        reset({
          id,
          dataset,
          numberOfAnnotations: n,
          username,
          evaluations: {},
          mapping: currentMapping,
          tests: data
        });
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (dataset && numberOfAnnotations > 0 && tests.length === 0) {
      fetchTests(dataset, numberOfAnnotations);
    }
  }, [dataset, numberOfAnnotations, tests, fetchTests]);

  useEffect(() => {
    if (selected?.annotation) {
      reset(selected.annotation);
    }
  }, [selected?.annotation]);

  useEffect(() => {
    fetchSavedAnnotations();
  }, []);

  useEffect(() => {
    const saveFormData = async () => {
      const formData = {
        id,
        dataset,
        numberOfAnnotations,
        username,
        password,
        mapping,
        evaluations,
        tests
      };

      await update(formData);
    };

    saveFormData();
  }, [
    id,
    dataset,
    numberOfAnnotations,
    username,
    password,
    mapping,
    evaluations,
    tests
  ]);

  const fetchSavedAnnotations = () => {
    fetch(`/api/annotations`)
      .then((response) => {
        return response.json();
      })
      .then((data: Annotation[]) => {
        setSavedAnnotations(data);
        setSelected((prevSelected) => {
          if (prevSelected.annotation) {
            const previousAnnotation =
              data.find(
                (annotation) => annotation.id === prevSelected.annotation?.id
              ) || null;
            const previousTest =
              previousAnnotation?.tests.find(
                (test) => test.docid === prevSelected.test?.docid
              ) || null;
            return {
              annotation: previousAnnotation,
              test: previousTest
            };
          }
          return {
            annotation: prevSelected.annotation,
            test: prevSelected.test
          };
        });
      });
  };

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
    // { label: '10', value: '10' },
    { label: '25', value: '25' }
    // { label: '40', value: '40' },
    // { label: '50', value: '50' }
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
          defaultValue={5}
          rules={{ required: 'Number of tests is required' }}
          render={({ field }) => (
            <ToggleGroup
              type="single"
              className="w-[180px]"
              onValueChange={(value) => {
                field.onChange(value);
                fetchTests(dataset, value);
              }}
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

  const savedAnnotationsList = (
    <Accordion type="single" collapsible>
      <AccordionItem value="dataset">
        <AccordionTrigger>
          Show Saved Annotations: {savedAnnotations.length}
        </AccordionTrigger>
        <AccordionContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Id</TableHead>
                <TableHead>Annotator</TableHead>
                <TableHead>Dataset</TableHead>
                <TableHead>Number Of Annotations</TableHead>
                <TableHead>Avg. Fluency</TableHead>
                <TableHead>Avg. Coherence</TableHead>
                <TableHead>Avg. Correctness</TableHead>
                <TableHead>Avg. Faithfulness</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {savedAnnotations.map((annotation, index) => {
                const evaluationAverages = Object.entries(
                  annotation.evaluations
                ).reduce(
                  (acc, [_, evals]) => {
                    Object.entries(evals).forEach(([model, metrics]) => {
                      if (!acc[model])
                        acc[model] = {
                          fluency: 0,
                          coherence: 0,
                          correctness: 0,
                          faithfulness: 0,
                          count: 0
                        };
                      acc[model].fluency +=
                        parseInt(metrics.fluency as any, 10) || 0;
                      acc[model].coherence +=
                        parseInt(metrics.coherence as any, 10) || 0;
                      acc[model].correctness +=
                        parseInt(metrics.correctness as any, 10) || 0;
                      acc[model].faithfulness +=
                        parseInt(metrics.faithfulness as any, 10) || 0;
                      acc[model].count += 1;
                    });
                    return acc;
                  },
                  {} as Record<
                    string,
                    {
                      fluency: number;
                      coherence: number;
                      correctness: number;
                      faithfulness: number;
                      count: number;
                    }
                  >
                );

                const evaluationSummary = Object.entries(
                  evaluationAverages
                ).reduce(
                  (acc, [model, metrics]) => {
                    const avgFluency = (
                      metrics.fluency / metrics.count
                    ).toFixed(2);
                    const avgCoherence = (
                      metrics.coherence / metrics.count
                    ).toFixed(2);
                    const avgCorrectness = (
                      metrics.correctness / metrics.count
                    ).toFixed(2);
                    const avgFaithfulness = (
                      metrics.faithfulness / metrics.count
                    ).toFixed(2);
                    const newModelName = model.toUpperCase();
                    acc.fluency.push(`${newModelName}: ${avgFluency}`);
                    acc.coherence.push(`${newModelName}: ${avgCoherence}`);
                    acc.correctness.push(`${newModelName}: ${avgCorrectness}`);
                    acc.faithfulness.push(
                      `${newModelName}: ${avgFaithfulness}`
                    );
                    return acc;
                  },
                  {
                    fluency: [] as string[],
                    coherence: [] as string[],
                    correctness: [] as string[],
                    faithfulness: [] as string[]
                  }
                );

                return (
                  <TableRow key={index}>
                    <TableCell>{annotation.id}</TableCell>
                    <TableCell>{annotation.username}</TableCell>
                    <TableCell>{annotation.dataset}</TableCell>
                    <TableCell>{annotation.numberOfAnnotations}</TableCell>
                    <TableCell>
                      {evaluationSummary.fluency
                        .sort((a, b) => {
                          const aValue = parseFloat(a.split(': ')[1]);
                          const bValue = parseFloat(b.split(': ')[1]);
                          return bValue - aValue;
                        })
                        .map((item, idx) => (
                          <TableRow key={idx}>
                            <TableCell>{item}</TableCell>
                          </TableRow>
                        ))}
                    </TableCell>
                    <TableCell>
                      <Table>
                        <TableBody>
                          {evaluationSummary.coherence
                            .sort((a, b) => {
                              const aValue = parseFloat(a.split(': ')[1]);
                              const bValue = parseFloat(b.split(': ')[1]);
                              return bValue - aValue;
                            })
                            .map((item, idx) => (
                              <TableRow key={idx}>
                                <TableCell>{item}</TableCell>
                              </TableRow>
                            ))}
                        </TableBody>
                      </Table>
                    </TableCell>
                    <TableCell>
                      {evaluationSummary.correctness
                        .sort((a, b) => {
                          const aValue = parseFloat(a.split(': ')[1]);
                          const bValue = parseFloat(b.split(': ')[1]);
                          return bValue - aValue;
                        })
                        .map((item, idx) => (
                          <TableRow key={idx}>
                            <TableCell>{item}</TableCell>
                          </TableRow>
                        ))}
                    </TableCell>
                    <TableCell>
                      {evaluationSummary.faithfulness
                        .sort((a, b) => {
                          const aValue = parseFloat(a.split(': ')[1]);
                          const bValue = parseFloat(b.split(': ')[1]);
                          return bValue - aValue;
                        })
                        .map((item, idx) => (
                          <TableRow key={idx}>
                            <TableCell>{item}</TableCell>
                          </TableRow>
                        ))}
                    </TableCell>

                    <TableCell>
                      <Button
                        variant="outline"
                        onClick={(e) => {
                          e.preventDefault();
                          setSelected({
                            annotation,
                            test: annotation.tests[0]
                          });
                        }}
                      >
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );

  const goldText = selected.test?.gold_text || '';
  const generatedText = mapping &&
    Object.keys(mapping).length > 0 &&
    selected.test?.docid &&
    mapping[selected.test?.docid] &&
    selected.test?.generations && (
      <div className="w-1/2 border-l p-4">
        <h3 className="text-md font-semibold">Generated Text</h3>
        <Tabs defaultValue="A" key={selected.test.docid}>
          <TabsList>
            <TabsTrigger value="A">A</TabsTrigger>
            <TabsTrigger value="B">B</TabsTrigger>
            <TabsTrigger value="C">C</TabsTrigger>
          </TabsList>
          {Object.entries(mapping[selected.test?.docid]).map(([model, key]) => (
            <TabsContent key={key} value={key}>
              {password === SECRET_WORDS && (
                <p className="mb-4">
                  Letter: '{key}' maps to '{model}'
                </p>
              )}
              <p className="text-sm">{selected.test?.generations[model]}</p>
              <div className="mt-4">
                <h4 className="font-semibold">Fluency</h4>
                <p className="text-xs">
                  Evaluate the quality of sentences individually for grammatical
                  correctness, proper word usage, and readability.
                </p>
                <p className="text-xs">(1: Least fluent, 5: Most fluent)</p>
                <Controller
                  key={`evaluations.${selected.test?.docid}.${model}.fluency`}
                  name={`evaluations.${selected.test?.docid}.${model}.fluency`}
                  control={control}
                  render={({ field }) => (
                    <ToggleGroup
                      variant="outline"
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
                <h4 className="font-semibold">Coherence</h4>
                <p className="text-xs">
                  Examine how well the sentences work together to form a logical
                  and connected narrative. Assess if the text flows smoothly and
                  maintains clarity throughout.
                </p>
                <p className="text-xs">(1: Least coherent, 5: Most coherent)</p>
                <Controller
                  key={`evaluations.${selected.test?.docid}.${model}.coherence`}
                  name={`evaluations.${selected.test?.docid}.${model}.coherence`}
                  control={control}
                  render={({ field }) => (
                    <ToggleGroup
                      variant="outline"
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
                  Examine how accurately the generated text aligns with the
                  given golden answer.
                </p>
                <p className="text-xs">(1: Least correct, 5: Most correct)</p>
                <Controller
                  key={`evaluations.${selected.test?.docid}.${model}.correctness`}
                  name={`evaluations.${selected.test?.docid}.${model}.correctness`}
                  control={control}
                  render={({ field }) => (
                    <ToggleGroup
                      variant="outline"
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
                  Evaluate how well the generated text reflects and aligns with
                  the information provided in the given passages.
                </p>
                <p className="text-xs">(1: Least faithful, 5: Most faithful)</p>
                <Controller
                  key={`evaluations.${selected.test?.docid}.${model}.faithfulness`}
                  name={`evaluations.${selected.test?.docid}.${model}.faithfulness`}
                  control={control}
                  render={({ field }) => (
                    <ToggleGroup
                      variant="outline"
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
          ))}
        </Tabs>
      </div>
    );
  const annotatorUsername = (
    <Card>
      <CardHeader>
        <CardTitle>Username</CardTitle>
        <CardDescription>Enter the username of the annotator.</CardDescription>
      </CardHeader>
      <CardContent className="flex-col gap-2">
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
        {username === 'santosh' && (
          <Controller
            name="password"
            control={control}
            defaultValue=""
            rules={{
              required: 'Secret Words is required for saved annotations'
            }}
            render={({ field }) => (
              <>
                <p className="text-sm mt-4">
                  Enter the secret words to view saved annotations.
                </p>
                <Input {...field} placeholder="Password" />
              </>
            )}
          />
        )}
      </CardContent>
    </Card>
  );

  const isSubmitAllowed = tests.every((test) => {
    const currentEvaluations = evaluations[test.docid];
    return (
      currentEvaluations &&
      Object.values(currentEvaluations).length === 3 &&
      Object.values(currentEvaluations).every((evals) => {
        return (
          evals.fluency &&
          evals.correctness &&
          evals.faithfulness &&
          evals.coherence
        );
      })
    );
  });

  const missingEvaluations = tests
    .map((test, index) => {
      const currentEvaluations = evaluations[test.docid];
      const missingModels = Object.entries(mapping[test.docid] || {})
        .filter(([model, key]) => {
          const evals = currentEvaluations?.[model];
          return (
            !evals ||
            !evals.fluency ||
            !evals.correctness ||
            !evals.faithfulness ||
            !evals.coherence
          );
        })
        .map(([model, key]) => key);
      return { index, missingModels };
    })
    .filter(({ missingModels }) => missingModels.length > 0);
  return (
    <div className="h-[95vh] flex flex-col">
      <div className="grid gap-2 ">
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
          <AccordionItem value="username">
            <AccordionTrigger>Username: {username}</AccordionTrigger>
            <AccordionContent>{annotatorUsername}</AccordionContent>
          </AccordionItem>
        </Accordion>
        {password === SECRET_WORDS && (
          <Card>
            <CardHeader>
              <CardTitle>Saved Annotations</CardTitle>
            </CardHeader>
            <CardContent>{savedAnnotationsList}</CardContent>
          </Card>
        )}
        <div className="flex flex-wrap gap-4">
          {missingEvaluations.length > 0 && (
            <Card className="w-full">
              <CardHeader>
                <CardTitle>Missing Evaluations</CardTitle>
                <CardDescription>
                  Complete the evaluations for the following tests.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-4">
                  {missingEvaluations.map(({ index, missingModels }) => (
                    <div key={index} className="flex flex-col items-start">
                      <p className="font-semibold">Test {index + 1}</p>
                      <div className="flex gap-2">
                        {missingModels.sort().map((model) => (
                          <span
                            key={model}
                            className="bg-red-200 text-red-800 px-2 py-1 rounded"
                          >
                            {model}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
        <Card className="flex flex-col flex-grow overflow-hidden min-h-[50vh] max-h-[70vh]">
          <CardHeader>
            <CardTitle>Annotations</CardTitle>
            <div className="flex items-center mb-4">
              {id !== 0 && (
                <CardTitle>Editing annotation with id: {id}</CardTitle>
              )}
              <Button
                className="ml-4"
                onClick={async (e) => {
                  e.preventDefault();
                  if (
                    !confirm(
                      'Are you sure you want to start a new annotation? This will delete all current progress.'
                    )
                  ) {
                    return;
                  }
                  await deleteRecord(0);
                  reset({
                    id: 0,
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
            </div>
          </CardHeader>
          <CardContent className="flex flex-grow overflow-auto">
            {loading ? (
              <div className="flex justify-center items-center self-center h-full w-full">
                <p className="mr-2">Loading tests...</p>
                <Loader className="animate-spin" />
              </div>
            ) : (
              <div className="flex">
                <div className="w-1/8 overflow-y-auto overflow-x-hidden pr-5">
                  <p className="text-lg font-semibold">Tests</p>
                  <ul className="grid gap-2">
                    {tests?.map((test, index) => {
                      const evaluationKeys = [
                        'fluency',
                        'coherence',
                        'correctness',
                        'faithfulness'
                      ];
                      const evaluationsForTest =
                        evaluations?.[test.docid] || {};
                      const completedKeys = Object.entries(
                        evaluationsForTest
                      ).reduce(
                        (acc, [key, value]) =>
                          acc +
                          Object.values(value).filter(
                            (singleValue) => singleValue
                          ).length,
                        0
                      );
                      const completionPercentage = (
                        (completedKeys / (4 * 3)) *
                        100
                      ).toFixed(2);
                      return (
                        <li key={index} className="mb-2">
                          <Button
                            variant="outline"
                            onClick={(e) => {
                              e.preventDefault();
                              setSelected({
                                annotation: selected.annotation,
                                test
                              });
                            }}
                            className={
                              selected.test?.docid === test.docid
                                ? 'bg-primary text-primary-foreground'
                                : ''
                            }
                          >
                            Test {index + 1} ({completionPercentage}%)
                          </Button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
                <div className="w-1/2 p-4 overflow-y-auto overflow-x-hidden">
                  {selected.test === null ? (
                    <p>Select a test to start the annotation.</p>
                  ) : (
                    <div className="grid gap-5">
                      <div>
                        <h3 className="text-md font-semibold">Question</h3>
                        <p className="text-sm">
                          {selected.test?.previous_text || ''}
                        </p>
                      </div>
                      <div>
                        <h3 className="text-md font-semibold mb-4">
                          Reference Answer
                        </h3>
                        <p className="text-sm">{goldText}</p>
                      </div>
                      <Card>
                        <CardHeader>
                          <CardTitle>Passages</CardTitle>
                          <CardDescription>
                            Select a passage to view more information.
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="flex gap-4">
                          {selected.test?.top_k_passages
                            .slice(0, 3)
                            .map((passage, index) => {
                              const [title, content] = passage.split('\n');
                              return (
                                <Popover key={index}>
                                  <PopoverTrigger>
                                    <Button variant="outline">{title}</Button>
                                  </PopoverTrigger>
                                  <PopoverContent className="w-[40vw]">
                                    <pre className="text-sm whitespace-pre-wrap">
                                      {content}
                                    </pre>
                                  </PopoverContent>
                                </Popover>
                              );
                            })}
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader>
                          <CardTitle>Citations</CardTitle>
                          <CardDescription>
                            Select a citation to view more information.
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="flex gap-4">
                          {selected.test?.citations.map((citation, index) => (
                            <Dialog key={citation[0]}>
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
                        </CardContent>
                      </Card>
                    </div>
                  )}
                </div>

                {generatedText}
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button
              onClick={handleSubmit(onSubmit)}
              disabled={!isSubmitAllowed}
            >
              Submit Annotation
            </Button>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
};

export default function AnnotatePage() {
  const { toast } = useToast();
  const { getAll } = useIndexedDB(STORE_NAME);
  const getIndexedDBState = async () => {
    toast({
      title: 'Created',
      description: 'New annotation form created.'
    });
    const formDataItems = await getAll();
    if (formDataItems.length > 0) {
      const formData = formDataItems[0];
      toast({
        title: 'Synced',
        description: 'Continuing where we left off!.'
      });
      return formData;
    }
    return {
      id: 0,
      dataset: 'echr_qa',
      numberOfAnnotations: 25,
      username: 'lawyer',
      password: '',
      evaluations: {},
      mapping: {},
      tests: []
    };
  };
  const formMethods = useForm<FormValues>({
    async defaultValues() {
      return getIndexedDBState();
    }
  });

  const [dataset] = useWatch({
    control: formMethods.control,
    name: ['dataset']
  });

  if (!dataset) {
    return null;
  }

  return (
    <FormProvider {...formMethods}>
      <AnnotatePageInner />
    </FormProvider>
  );
}
