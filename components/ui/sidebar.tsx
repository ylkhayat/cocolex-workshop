import * as React from 'react';
import { Button } from '@/components/ui/button';
import { useRouter } from 'next/compat/router';

interface SidebarProps {
  children: React.ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const router = useRouter();

  const handleNavigation = (path: string) => {
    router.push(path);
  };

  return (
    <div className="min-w-[500px] bg-gray-500 text-white">
      <div className="p-4">
        <h2 className="text-lg font-semibold">Experiments</h2>
      </div>
      <div className="p-4">
        {children}
        {/* {React.Children.map(children, (child) => (
          <div className="mb-2">
            <Button
              variant="ghost"
              className="w-full text-left"
              onClick={() => handleNavigation(child.props.path)}
            >
              {child.props.name}
            </Button>
          </div>
        ))} */}
      </div>
    </div>
  );
}
