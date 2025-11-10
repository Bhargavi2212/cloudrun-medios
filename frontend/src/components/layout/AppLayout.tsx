import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { SessionTimeoutWarning } from '@/components/auth/SessionTimeoutWarning';

interface AppLayoutProps {
  children: React.ReactNode;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  return (
    // Main container fills the entire screen
    <div className="flex h-screen bg-gray-100 dark:bg-gray-950">
      {/* Session timeout monitoring */}
      <SessionTimeoutWarning />
      
      {/* 1. The Sidebar (Fixed Width) */}
      <Sidebar />

      {/* 2. The Main Content Area (Takes remaining space) */}
      <div className="flex-1 flex flex-col overflow-hidden">
        
        {/* The Header */}
        <Header />

               {/* The Page Content (Scrollable) */}
               <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100 dark:bg-gray-950 p-8" role="main">
                 {children} 
               </main>

      </div>
    </div>
  );
};