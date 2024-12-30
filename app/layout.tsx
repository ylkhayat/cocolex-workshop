import './globals.css';

import { Analytics } from '@vercel/analytics/react';

export const metadata = {
  title: 'CoCoLex Dashboard',
  description:
    'Dashboard for CoCoLex project, a tool for monitoring and annotating. Built with Next.js and Vercel.'
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="flex min-h-screen w-full flex-col">{children}</body>
      <Analytics />
    </html>
  );
}
