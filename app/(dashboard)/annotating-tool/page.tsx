import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';

export default function AnnotatePage() {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    annotation: ''
  });

  const [annotations, setAnnotations] = useState([]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setAnnotations([...annotations, formData]);
    setFormData({
      name: '',
      description: '',
      annotation: ''
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Annotate</CardTitle>
        <CardDescription>Submit your annotations below.</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="name" className="block text-sm font-medium text-gray-700">
              Name
            </label>
            <Input
              id="name"
              name="name"
              type="text"
              value={formData.name}
              onChange={handleChange}
              required
            />
          </div>
          <div className="mb-4">
            <label htmlFor="description" className="block text-sm font-medium text-gray-700">
              Description
            </label>
            <Input
              id="description"
              name="description"
              type="text"
              value={formData.description}
              onChange={handleChange}
              required
            />
          </div>
          <div className="mb-4">
            <label htmlFor="annotation" className="block text-sm font-medium text-gray-700">
              Annotation
            </label>
            <Textarea
              id="annotation"
              name="annotation"
              value={formData.annotation}
              onChange={handleChange}
              required
            />
          </div>
          <Button type="submit">Submit</Button>
        </form>
        <div className="mt-8">
          <h2 className="text-lg font-medium">Annotations</h2>
          {annotations.map((annotation, index) => (
            <div key={index} className="mt-4">
              <h3 className="text-md font-semibold">{annotation.name}</h3>
              <p className="text-sm">{annotation.description}</p>
              <p className="text-sm">{annotation.annotation}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
