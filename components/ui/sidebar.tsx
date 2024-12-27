import * as React from 'react';
import { useRouter } from 'next/router';
import { Button } from '@/components/ui/button';

interface SidebarProps {
  children: React.ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const router = useRouter();

  const handleNavigation = (path: string) => {
    router.push(path);
  };

  return (
    <div className="w-64 bg-gray-800 text-white">
      <div className="p-4">
        <h2 className="text-lg font-semibold">Experiments</h2>
      </div>
      <div className="p-4">
        {React.Children.map(children, (child) => (
          <div className="mb-2">
            <Button
              variant="ghost"
              className="w-full text-left"
              onClick={() => handleNavigation(child.props.path)}
            >
              {child.props.name}
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
}
