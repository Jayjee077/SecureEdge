import React from 'react';
import './App.css';
import CosmicBackground from './CosmicBackground';
import HolographicChart from './HolographicChart';
import GlassmorphismUI from './GlassmorphismUI';

function App() {
  const data = [
    { date: '2025-01-01', value: 100 },
    { date: '2025-01-02', value: 120 },
    { date: '2025-01-03', value: 150 },
    // Add more data for chart here
  ];

  return (
    <div className="App">
      <CosmicBackground />
      <div className="content">
        <HolographicChart data={data} />
        <GlassmorphismUI />
      </div>
    </div>
  );
}

export default App;
