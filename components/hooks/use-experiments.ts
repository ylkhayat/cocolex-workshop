import { useEffect, useState } from 'react';

export const useExperiments = () => {
  const [experimentsData, setExperimentsData] = useState<any[]>([]);
  useEffect(() => {
    fetch('/api/experiments')
      .then((res) => res.json())
      .then((data) => setExperimentsData(data));
  }, []);
  return experimentsData;
};
