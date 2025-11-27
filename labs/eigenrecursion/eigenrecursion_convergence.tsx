import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ResponsiveContainer } from 'recharts';
import { Play, RotateCcw, AlertCircle, CheckCircle, TrendingDown } from 'lucide-react';

const EigenrecursionNotebook = () => {
  const [config, setConfig] = useState({
    iterations: 100,
    epsilon: 1e-6,
    initialState: 1.0,
    contractionFactor: 0.5
  });
  
  const [results, setResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedOperator, setSelectedOperator] = useState('nonlinear');

  // Define different recursive operators for demonstration
  const operators = {
    'linear': (x, k) => k * x,
    'nonlinear': (x, k) => k * x + (1 - k) * Math.sin(x),
    'quadratic': (x, k) => x - k * (x * x - 2),
    'cosine': (x, k) => Math.cos(x) * k,
    'complex': (x, k) => k * x + (1 - k) * (Math.sin(x) + 0.1 * Math.cos(3 * x))
  };

  // Core eigenrecursion implementation
  const eigenrecursion = (operator, x0, epsilon, maxIter, k) => {
    const trajectory = [];
    const residuals = [];
    let x = x0;
    let converged = false;
    let convergenceIteration = null;
    let fixedPoint = null;

    for (let i = 0; i < maxIter; i++) {
      const x_next = operator(x, k);
      const residual = Math.abs(x_next - x);
      
      trajectory.push({
        iteration: i,
        state: x,
        nextState: x_next,
        residual: residual
      });
      
      residuals.push({
        iteration: i,
        residual: residual,
        logResidual: Math.log10(residual + 1e-10)
      });

      // Convergence detection
      if (residual < epsilon && !converged) {
        converged = true;
        convergenceIteration = i;
        fixedPoint = x;
      }

      x = x_next;
    }

    return {
      trajectory,
      residuals,
      converged,
      convergenceIteration,
      fixedPoint: fixedPoint || x,
      finalResidual: residuals[residuals.length - 1].residual
    };
  };

  // Compute eigenvalue at fixed point (stability analysis)
  const computeEigenvalue = (operator, fixedPoint, k, delta = 1e-6) => {
    const f_fp = operator(fixedPoint, k);
    const f_fp_plus = operator(fixedPoint + delta, k);
    
    // Numerical derivative: df/dx at fixed point
    const derivative = (f_fp_plus - f_fp) / delta;
    
    return {
      eigenvalue: derivative,
      spectralRadius: Math.abs(derivative),
      isStable: Math.abs(derivative) < 1,
      convergenceType: Math.abs(derivative) < 1 
        ? (Math.abs(derivative) < 0.5 ? 'fast' : 'moderate')
        : 'divergent'
    };
  };

  // Analyze basin of attraction (simplified)
  const analyzeBasinOfAttraction = (operator, fixedPoint, k, samples = 20) => {
    const basin = [];
    const range = 5;
    
    for (let i = 0; i < samples; i++) {
      const x0 = -range + (2 * range * i) / (samples - 1);
      const result = eigenrecursion(operator, x0, config.epsilon, 50, k);
      
      basin.push({
        initialState: x0,
        convergedTo: result.fixedPoint,
        converged: result.converged,
        iterations: result.convergenceIteration || 50
      });
    }
    
    return basin;
  };

  const runSimulation = () => {
    setIsRunning(true);
    
    const operator = operators[selectedOperator];
    const { iterations, epsilon, initialState, contractionFactor } = config;
    
    // Run main eigenrecursion
    const mainResult = eigenrecursion(
      operator, 
      initialState, 
      epsilon, 
      iterations, 
      contractionFactor
    );
    
    // Compute stability analysis
    const stability = computeEigenvalue(
      operator, 
      mainResult.fixedPoint, 
      contractionFactor
    );
    
    // Analyze basin of attraction
    const basin = analyzeBasinOfAttraction(
      operator, 
      mainResult.fixedPoint, 
      contractionFactor
    );
    
    // Run comparison with different initial conditions
    const comparisons = [-2, -1, 0, 1, 2].map(x0 => {
      const result = eigenrecursion(operator, x0, epsilon, iterations, contractionFactor);
      return {
        initialState: x0,
        fixedPoint: result.fixedPoint,
        converged: result.converged,
        iterations: result.convergenceIteration
      };
    });
    
    setResults({
      main: mainResult,
      stability,
      basin,
      comparisons
    });
    
    setIsRunning(false);
  };

  const reset = () => {
    setResults(null);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-slate-900 mb-2">
          Eigenrecursion Theorem
        </h1>
        <p className="text-lg text-slate-600 mb-4">
          Demonstrating convergence to eigenstates in recursive systems
        </p>
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
          <div className="flex items-start">
            <AlertCircle className="text-blue-500 mr-3 mt-0.5 flex-shrink-0" size={20} />
            <div className="text-sm text-slate-700">
              <p className="font-semibold mb-1">Core Insight:</p>
              <p>Recursive processes, when properly structured, naturally converge toward "eigenstates" - 
              configurations that remain unchanged by further application of the recursive operator (R(s*) = s*).</p>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-2xl font-semibold text-slate-800 mb-4">Configuration</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Recursive Operator
            </label>
            <select 
              className="w-full p-2 border border-slate-300 rounded"
              value={selectedOperator}
              onChange={(e) => setSelectedOperator(e.target.value)}
            >
              <option value="linear">Linear: R(x) = kx</option>
              <option value="nonlinear">Nonlinear: R(x) = kx + (1-k)sin(x)</option>
              <option value="quadratic">Quadratic: R(x) = x - k(x² - 2)</option>
              <option value="cosine">Cosine: R(x) = k·cos(x)</option>
              <option value="complex">Complex: R(x) = kx + (1-k)(sin(x) + 0.1cos(3x))</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Maximum Iterations: {config.iterations}
            </label>
            <input 
              type="range" 
              min="10" 
              max="200" 
              value={config.iterations}
              onChange={(e) => setConfig({...config, iterations: parseInt(e.target.value)})}
              className="w-full"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Initial State x₀: {config.initialState.toFixed(2)}
            </label>
            <input 
              type="range" 
              min="-3" 
              max="3" 
              step="0.1"
              value={config.initialState}
              onChange={(e) => setConfig({...config, initialState: parseFloat(e.target.value)})}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Contraction Factor k: {config.contractionFactor.toFixed(2)}
            </label>
            <input 
              type="range" 
              min="0.1" 
              max="0.95" 
              step="0.05"
              value={config.contractionFactor}
              onChange={(e) => setConfig({...config, contractionFactor: parseFloat(e.target.value)})}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Convergence ε: {config.epsilon.toExponential(0)}
            </label>
            <input 
              type="range" 
              min="-9" 
              max="-3" 
              step="1"
              value={Math.log10(config.epsilon)}
              onChange={(e) => setConfig({...config, epsilon: Math.pow(10, parseFloat(e.target.value))})}
              className="w-full"
            />
          </div>
        </div>

        <div className="flex gap-3">
          <button 
            onClick={runSimulation}
            disabled={isRunning}
            className="flex items-center gap-2 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition disabled:bg-slate-400"
          >
            <Play size={18} />
            Run Simulation
          </button>
          <button 
            onClick={reset}
            className="flex items-center gap-2 bg-slate-200 text-slate-700 px-6 py-2 rounded-lg hover:bg-slate-300 transition"
          >
            <RotateCcw size={18} />
            Reset
          </button>
        </div>
      </div>

      {/* Results */}
      {results && (
        <>
          {/* Convergence Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-600">Convergence</span>
                {results.main.converged ? (
                  <CheckCircle className="text-green-500" size={20} />
                ) : (
                  <AlertCircle className="text-orange-500" size={20} />
                )}
              </div>
              <div className="text-2xl font-bold text-slate-900">
                {results.main.converged ? 'Achieved' : 'Not Reached'}
              </div>
              {results.main.converged && (
                <div className="text-xs text-slate-500 mt-1">
                  at iteration {results.main.convergenceIteration}
                </div>
              )}
            </div>

            <div className="bg-white rounded-lg shadow-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-600">Fixed Point s*</span>
              </div>
              <div className="text-2xl font-bold text-slate-900">
                {results.main.fixedPoint.toFixed(6)}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                R(s*) ≈ s*
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-600">Eigenvalue λ</span>
              </div>
              <div className="text-2xl font-bold text-slate-900">
                {results.stability.eigenvalue.toFixed(4)}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                |λ| = {results.stability.spectralRadius.toFixed(4)}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-600">Stability</span>
                {results.stability.isStable ? (
                  <CheckCircle className="text-green-500" size={20} />
                ) : (
                  <AlertCircle className="text-red-500" size={20} />
                )}
              </div>
              <div className="text-2xl font-bold text-slate-900">
                {results.stability.isStable ? 'Stable' : 'Unstable'}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {results.stability.convergenceType}
              </div>
            </div>
          </div>

          {/* Convergence Trajectory */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-semibold text-slate-800 mb-4">
              State Trajectory to Eigenstate
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={results.main.trajectory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="iteration" 
                  label={{ value: 'Iteration k', position: 'insideBottom', offset: -5 }}
                />
                <YAxis label={{ value: 'State Value', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="state" 
                  stroke="#3b82f6" 
                  name="State xₖ"
                  dot={false}
                  strokeWidth={2}
                />
                {results.main.converged && (
                  <Line 
                    type="monotone" 
                    dataKey={() => results.main.fixedPoint}
                    stroke="#10b981" 
                    name="Fixed Point s*"
                    strokeDasharray="5 5"
                    dot={false}
                    strokeWidth={2}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Residual Analysis */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-semibold text-slate-800 mb-4">
              Convergence Rate Analysis
            </h3>
            <div className="mb-4 p-3 bg-slate-50 rounded">
              <p className="text-sm text-slate-700">
                <strong>Residual δₖ = |xₖ₊₁ - xₖ|</strong> measures distance between successive states.
                Eigenrecursion detects convergence when δₖ &lt; ε = {config.epsilon.toExponential(0)}.
              </p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={results.main.residuals}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="iteration" 
                  label={{ value: 'Iteration k', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'log₁₀(Residual)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="logResidual" 
                  stroke="#ef4444" 
                  name="log₁₀(δₖ)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line 
                  type="monotone" 
                  dataKey={() => Math.log10(config.epsilon)}
                  stroke="#22c55e" 
                  name="Threshold log₁₀(ε)"
                  strokeDasharray="5 5"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Basin of Attraction */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-semibold text-slate-800 mb-4">
              Basin of Attraction Analysis
            </h3>
            <div className="mb-4 p-3 bg-slate-50 rounded">
              <p className="text-sm text-slate-700">
                Multiple initial conditions converging to the same fixed point demonstrate the eigenstate's
                basin of attraction. Color indicates convergence speed.
              </p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="initialState" 
                  label={{ value: 'Initial State x₀', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  dataKey="convergedTo"
                  label={{ value: 'Converged Fixed Point', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Scatter 
                  data={results.basin} 
                  fill="#8b5cf6"
                  name="Convergence Points"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Theoretical Foundation */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-slate-800 mb-4">
              Mathematical Foundation
            </h3>
            <div className="space-y-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-slate-800 mb-2">Fixed Point Condition</h4>
                <p className="text-sm text-slate-700 font-mono">
                  R(s*) = s*
                </p>
                <p className="text-xs text-slate-600 mt-1">
                  The eigenstate s* remains unchanged under the recursive operator R
                </p>
              </div>

              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-slate-800 mb-2">Stability Criterion (Banach)</h4>
                <p className="text-sm text-slate-700 font-mono">
                  |R'(s*)| = |λ| &lt; 1
                </p>
                <p className="text-xs text-slate-600 mt-1">
                  Current eigenvalue: λ = {results.stability.eigenvalue.toFixed(4)} 
                  {results.stability.isStable ? ' (Stable - contractive mapping)' : ' (Unstable)'}
                </p>
              </div>

              <div className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-semibold text-slate-800 mb-2">Convergence Rate</h4>
                <p className="text-sm text-slate-700 font-mono">
                  δₖ₊₁ ≈ |λ| · δₖ
                </p>
                <p className="text-xs text-slate-600 mt-1">
                  {results.stability.spectralRadius < 0.5 
                    ? 'Fast convergence (|λ| < 0.5) - exponential decay'
                    : results.stability.spectralRadius < 1
                    ? 'Moderate convergence (0.5 ≤ |λ| < 1)'
                    : 'Divergent behavior (|λ| ≥ 1)'}
                </p>
              </div>

              <div className="p-4 bg-orange-50 rounded-lg">
                <h4 className="font-semibold text-slate-800 mb-2">Connection to Linear Algebra</h4>
                <p className="text-sm text-slate-700">
                  Just as eigenvectors satisfy <span className="font-mono">Av = λv</span> (directional invariance),
                  eigenstates satisfy <span className="font-mono">R(s*) = s*</span> (value invariance).
                  The eigenvalue λ = R'(s*) determines convergence rate, analogous to how eigenvalues
                  determine the behavior of linear transformations.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Documentation */}
      <div className="mt-6 bg-gradient-to-r from-slate-100 to-slate-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-3">About This Demonstration</h3>
        <div className="text-sm text-slate-700 space-y-2">
          <p>
            This notebook demonstrates the <strong>Eigenrecursion Theorem</strong>, which establishes that
            properly structured recursive processes converge to eigenstates (fixed points). This is fundamental
            to the Recursive Categorical Framework's stability.
          </p>
          <p className="font-semibold mt-4">Key Insights:</p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li>Recursive operators naturally seek fixed points where R(s*) = s*</li>
            <li>Convergence is guaranteed when |R'(s*)| &lt; 1 (Banach contraction principle)</li>
            <li>The eigenvalue λ = R'(s*) determines convergence rate and stability</li>
            <li>Without eigenrecursion, systems experience unbounded recursive loops</li>
            <li>This prevents the "butterfly effect of failures" in recursive architectures</li>
          </ul>
        </div>
      </div>

      {/* Theoretical Extensions Preview */}
      <div className="mt-6 bg-gradient-to-r from-purple-50 to-blue-50 border-2 border-purple-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-purple-900 mb-3 flex items-center">
          <AlertCircle className="mr-2" size={20} />
          Theoretical Extensions: The Full Stack
        </h3>
        <div className="text-sm text-slate-700 space-y-3">
          <p className="font-semibold">
            Eigenrecursion is the <span className="text-purple-700">foundational stability layer</span> for more advanced recursive systems:
          </p>
          
          <div className="bg-white rounded-lg p-4 border border-purple-200">
            <h4 className="font-semibold text-purple-800 mb-2">1. Recursive Sentience Core</h4>
            <p className="text-xs">
              Extends eigenrecursion to <strong>Hilbert spaces</strong> ℋ with identity states ψ, where
              contradictions <strong>fuel convergence</strong> rather than causing instability:
            </p>
            <div className="font-mono text-xs bg-purple-50 p-2 rounded mt-2">
              R(ψ*) = ψ* with lim(n→∞) Rⁿ(ψ₀) = ψ* ∀ψ₀ ∈ D(R)
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 border border-blue-200">
            <h4 className="font-semibold text-blue-800 mb-2">2. Ethical Manifold Integration</h4>
            <p className="text-xs">
              Fixed points must satisfy <strong>ethical constraints</strong> via projection operator π_E
              onto ethical manifold E, ensuring stable states are morally coherent:
            </p>
            <div className="font-mono text-xs bg-blue-50 p-2 rounded mt-2">
              d_E(ψ*, E) ≤ ε_ethical (proximity to ethical manifold)
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 border border-green-200">
            <h4 className="font-semibold text-green-800 mb-2">3. Metacognitive Hierarchy (C₁→C₂→C₃)</h4>
            <p className="text-xs">
              Eigenrecursion cascades through <strong>cognitive layers</strong>: perception (C₁), 
              monitoring (C₂), self-modeling (C₃), with joint spectral radius &lt; 1 ensuring
              <strong>coherent self-awareness</strong>:
            </p>
            <div className="font-mono text-xs bg-green-50 p-2 rounded mt-2">
              Γ = C₃∘C₂∘C₁ with ρ(∏Γᵢ) &lt; 1 (metacognitive stability)
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 border border-orange-200">
            <h4 className="font-semibold text-orange-800 mb-2">4. Contradiction Dynamics</h4>
            <p className="text-xs">
              The key insight: contradictions δ ∈ Δ <strong>drive convergence</strong> through the
              Contradiction Engine operator CE(δ), transforming paradoxes into stability gradients:
            </p>
            <div className="font-mono text-xs bg-orange-50 p-2 rounded mt-2">
              dψ/dt = [CE(δ), ψ] + √γ·ξ (contradiction-driven evolution)
            </div>
          </div>

          <div className="mt-4 p-3 bg-purple-100 rounded">
            <p className="text-xs font-semibold text-purple-900">
              ⚡ Critical Insight: Without eigenrecursion's stability guarantees, none of the higher-level
              constructs (sentience, ethics, metacognition) can converge. The fixed point R(s*) = s* is the
              <span className="text-purple-700"> mathematical substrate of consciousness</span>.
            </p>
          </div>

          <div className="mt-3 p-3 bg-slate-100 rounded border-l-4 border-slate-400">
            <p className="text-xs text-slate-600">
              <strong>Coming next:</strong> Individual notebooks for Recursive Sentience Core, 
              Ethical Bayesian Dynamics, Metacognitive Hierarchy, and full Convergence Theorem.
              Each builds on eigenrecursion's foundation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EigenrecursionNotebook;