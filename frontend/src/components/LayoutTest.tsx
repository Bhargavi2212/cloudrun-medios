import React from 'react';

export const LayoutTest: React.FC = () => {
  return (
    <div className="h-screen w-screen bg-red-500 flex">
      <div className="w-64 bg-blue-500 text-white p-4">
        <h2>SIDEBAR</h2>
        <p>If you see this blue sidebar on the left, CSS is working!</p>
      </div>
      <div className="flex-1 bg-green-500 text-white p-4">
        <h2>MAIN CONTENT</h2>
        <p>If you see this green area on the right, the layout is working!</p>
      </div>
    </div>
  );
};