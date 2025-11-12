import React from 'react';
import EigenrecursionConvergence from './components/EigenrecursionConvergence';
import TriaxialMetacognitiveCascade from './components/TriaxialMetacognitiveCascade';

const App: React.FC = () => {
  return (
    <div style={{ fontFamily: 'Inter, system-ui, sans-serif', padding: '1.5rem' }}>
      <h1 style={{ marginBottom: '1rem' }}>
        Recursive Categorical Framework Labs
      </h1>
      <p style={{ marginBottom: '1.5rem', color: '#4b5563' }}>
        Interactive notebooks demonstrating eigenrecursion convergence and triaxial metacognitive cascades.
      </p>
      
      <section style={{ marginBottom: '2rem' }}>
        <h2 style={{ marginBottom: '0.75rem' }}>Eigenrecursion Convergence Notebook</h2>
        <EigenrecursionConvergence />
      </section>

      <section>
        <h2 style={{ marginBottom: '0.75rem' }}>Triaxial Metacognitive Cascade Notebook</h2>
        <TriaxialMetacognitiveCascade />
      </section>
    </div>
  );
};

export default App;
