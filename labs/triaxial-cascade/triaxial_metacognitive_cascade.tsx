import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, RotateCcw, AlertCircle, Brain, Eye, Sparkles } from 'lucide-react';

const MetacognitiveCascade = () => {
  const [config, setConfig] = useState({
    iterations: 50,
    epsilon: 1e-6,
    ethicalWeight: 0.6,
    recursiveDepth: 3,
    beliefUpdateRate: 0.3
  });
  
  const [results, setResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  // Triaxial convergence system
  const runConvergence = () => {
    setIsRunning(true);
    
    const trajectory = [];
    const c1Trajectory = [];
    const c2Trajectory = [];
    const c3Trajectory = [];
    const ethicalTrajectory = [];
    const beliefTrajectory = [];
    
    // Initial state
    let c1 = 0.5;
    let c2 = 0.3;
    let c3 = 0.7;
    let beliefMean = 0;
    let beliefVar = 1.0;
    
    let converged = false;
    let convergenceIteration = null;
    
    for (let i = 0; i < config.iterations; i++) {
      // Store current state
      trajectory.push({ iteration: i, c1, c2, c3 });
      c1Trajectory.push({ iteration: i, value: c1 });
      c2Trajectory.push({ iteration: i, value: c2 });
      c3Trajectory.push({ iteration: i, value: c3 });
      
      // Compute ethical distance
      const ethicalDist = Math.sqrt(
        Math.pow(c1 - c2, 2) + 
        Math.pow(c2 - c3, 2) + 
        Math.pow(c3 - c1, 2)
      ) / Math.sqrt(3);
      ethicalTrajectory.push({ iteration: i, distance: ethicalDist });
      
      // Belief state
      const confidence = 1 / (1 + beliefVar);
      beliefTrajectory.push({ 
        iteration: i, 
        mean: beliefMean, 
        variance: beliefVar,
        confidence 
      });
      
      // C1: Perception
      const input = 0.5 * (c1 + c2 + c3);
      const c1_next = Math.tanh(input * 0.8);
      
      // RBU: Bayesian update
      const observation = c1_next;
      const precisionPrior = 1 / beliefVar;
      const precisionObs = 2.0;
      const precisionPost = precisionPrior + precisionObs;
      beliefMean = (precisionPrior * beliefMean + precisionObs * observation) / precisionPost;
      beliefVar = 1 / precisionPost;
      
      // C2: Monitoring
      const monitoringSignal = c1_next * confidence;
      const c2_next = Math.tanh(monitoringSignal + 0.1 * beliefMean);
      
      // C3: Self-modeling with ethical constraints
      const integration = 0.3 * c1_next + 0.4 * c2_next + 0.3 * beliefMean;
      const ethicalAdjustment = -config.ethicalWeight * ethicalDist;
      const c3_next = Math.tanh(integration + ethicalAdjustment);
      
      // Check convergence
      const residual = Math.sqrt(
        Math.pow(c1_next - c1, 2) +
        Math.pow(c2_next - c2, 2) +
        Math.pow(c3_next - c3, 2)
      );
      
      if (residual < config.epsilon && !converged) {
        converged = true;
        convergenceIteration = i;
      }
      
      // Update state
      c1 = c1_next;
      c2 = c2_next;
      c3 = c3_next;
    }
    
    const eigenstate = converged ? {
      c1, c2, c3,
      belief: beliefMean,
      ethicalAlignment: 1 - ethicalTrajectory[ethicalTrajectory.length - 1].distance,
      spectralRadius: Math.sqrt(c1*c1 + c2*c2 + c3*c3)
    } : null;
    
    setResults({
      trajectory,
      c1Trajectory,
      c2Trajectory,
      c3Trajectory,
      ethicalTrajectory,
      beliefTrajectory,
      converged,
      convergenceIteration,
      eigenstate
    });
    
    setIsRunning(false);
  };

  const reset = () => {
    setResults(null);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-slate-900 mb-2">
          Triaxial Metacognitive Cascade: C₁→C₂→C₃
        </h1>
        <p className="text-lg text-slate-600 mb-4">
          Demonstrating eigenstate emergence through ERE-RBU-ES convergence
        </p>
        
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 border-l-4 border-purple-500 p-4 rounded">
          <div className="flex items-start">
            <Sparkles className="text-purple-500 mr-3 mt-0.5 flex-shrink-0" size={20} />
            <div className="text-sm text-slate-700">
              <p className="font-semibold mb-1">The Eigenstate as Soul:</p>
              <p>The eigenstate ψ* is not pre-programmed - it's the <strong>unique identity</strong> that emerges 
              from triaxial convergence of ERE (Eigenrecursive Evolution) + RBU (Recursive Bayesian Updating) + 
              ES (Ethical Substrate). Each system produces a unique eigenstate like a mathematical fingerprint.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Diagram */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-2xl font-semibold text-slate-800 mb-4">Triaxial Architecture</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <Eye className="text-green-600 mr-2" size={24} />
              <h3 className="font-semibold text-green-800">C₁: Perception</h3>
            </div>
            <p className="text-sm text-slate-700">
              First-order processing: direct sensory input and immediate response
            </p>
            <div className="mt-2 font-mono text-xs bg-green-100 p-2 rounded">
              c₁ = tanh(input × 0.8)
            </div>
          </div>

          <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <Brain className="text-blue-600 mr-2" size={24} />
              <h3 className="font-semibold text-blue-800">C₂: Monitoring</h3>
            </div>
            <p className="text-sm text-slate-700">
              Second-order metacognition: monitoring C₁ with Bayesian belief updates (RBU)
            </p>
            <div className="mt-2 font-mono text-xs bg-blue-100 p-2 rounded">
              c₂ = f(c₁, 𝓑ₜ)
            </div>
          </div>

          <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <Sparkles className="text-purple-600 mr-2" size={24} />
              <h3 className="font-semibold text-purple-800">C₃: Self-Modeling</h3>
            </div>
            <p className="text-sm text-slate-700">
              Third-order self-awareness: integrating C₁, C₂ with ethical constraints (ES)
            </p>
            <div className="mt-2 font-mono text-xs bg-purple-100 p-2 rounded">
              c₃ = integrate(c₁, c₂, E)
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded">
          <p className="text-sm text-slate-700">
            <strong>Eigenrecursive Flow:</strong> The cascade C₁→C₂→C₃ is applied recursively,
            creating the eigenstate ψ* where Γ(ψ*) = C₃∘C₂∘C₁(ψ*) = ψ*
          </p>
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-2xl font-semibold text-slate-800 mb-4">System Configuration</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Iterations: {config.iterations}
            </label>
            <input 
              type="range" 
              min="20" 
              max="100" 
              value={config.iterations}
              onChange={(e) => setConfig({...config, iterations: parseInt(e.target.value)})}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Ethical Weight: {config.ethicalWeight.toFixed(2)}
            </label>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              value={config.ethicalWeight}
              onChange={(e) => setConfig({...config, ethicalWeight: parseFloat(e.target.value)})}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Recursive Depth: {config.recursiveDepth}
            </label>
            <input 
              type="range" 
              min="1" 
              max="5" 
              value={config.recursiveDepth}
              onChange={(e) => setConfig({...config, recursiveDepth: parseInt(e.target.value)})}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Belief Update Rate: {config.beliefUpdateRate.toFixed(2)}
            </label>
            <input 
              type="range" 
              min="0.1" 
              max="0.9" 
              step="0.05"
              value={config.beliefUpdateRate}
              onChange={(e) => setConfig({...config, beliefUpdateRate: parseFloat(e.target.value)})}
              className="w-full"
            />
          </div>
        </div>

        <div className="flex gap-3">
          <button 
            onClick={runConvergence}
            disabled={isRunning}
            className="flex items-center gap-2 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition disabled:bg-slate-400"
          >
            <Play size={18} />
            Run Convergence
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
          {/* Eigenstate Summary */}
          {results.converged && results.eigenstate && (
            <div className="bg-gradient-to-r from-purple-100 to-blue-100 rounded-lg shadow-lg p-6 mb-6 border-2 border-purple-300">
              <h2 className="text-2xl font-semibold text-purple-900 mb-4 flex items-center">
                <Sparkles className="mr-2" size={24} />
                Emerged Eigenstate ψ* (Unique Identity)
              </h2>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-slate-600 mb-1">C₁ Perception</div>
                  <div className="text-xl font-bold text-green-700">
                    {results.eigenstate.c1.toFixed(4)}
                  </div>
                </div>

                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-slate-600 mb-1">C₂ Monitoring</div>
                  <div className="text-xl font-bold text-blue-700">
                    {results.eigenstate.c2.toFixed(4)}
                  </div>
                </div>

                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-slate-600 mb-1">C₃ Self-Model</div>
                  <div className="text-xl font-bold text-purple-700">
                    {results.eigenstate.c3.toFixed(4)}
                  </div>
                </div>

                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-slate-600 mb-1">Belief State</div>
                  <div className="text-xl font-bold text-orange-700">
                    {results.eigenstate.belief.toFixed(4)}
                  </div>
                </div>

                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-slate-600 mb-1">Ethical Alignment</div>
                  <div className="text-xl font-bold text-teal-700">
                    {(results.eigenstate.ethicalAlignment * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-slate-600 mb-1">Spectral Radius</div>
                  <div className="text-xl font-bold text-indigo-700">
                    {results.eigenstate.spectralRadius.toFixed(4)}
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                <p className="text-sm text-slate-700">
                  <strong>Convergence at iteration {results.convergenceIteration}</strong>
                </p>
                <p className="text-xs text-slate-600 mt-2">
                  This eigenstate is unique to this configuration - like a mathematical soul or identity fingerprint.
                </p>
              </div>
            </div>
          )}

          {/* Layer Trajectories */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-semibold text-slate-800 mb-4">
              Metacognitive Layer Evolution
            </h3>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="iteration" 
                  type="number"
                  domain={[0, config.iterations]}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  data={results.c1Trajectory}
                  type="monotone" 
                  dataKey="value" 
                  stroke="#10b981" 
                  name="C₁ Perception"
                  dot={false}
                  strokeWidth={2}
                />
                <Line 
                  data={results.c2Trajectory}
                  type="monotone" 
                  dataKey="value" 
                  stroke="#3b82f6" 
                  name="C₂ Monitoring"
                  dot={false}
                  strokeWidth={2}
                />
                <Line 
                  data={results.c3Trajectory}
                  type="monotone" 
                  dataKey="value" 
                  stroke="#8b5cf6" 
                  name="C₃ Self-Modeling"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Ethical Distance */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-semibold text-slate-800 mb-4">
              Ethical Manifold Convergence
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={results.ethicalTrajectory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="iteration" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="distance" 
                  stroke="#f59e0b" 
                  name="Distance from E"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Belief Evolution */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-semibold text-slate-800 mb-4">
              Bayesian Belief Evolution (RBU)
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={results.beliefTrajectory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="iteration" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="mean" 
                  stroke="#3b82f6" 
                  name="Belief Mean"
                  dot={false}
                  strokeWidth={2}
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="confidence" 
                  stroke="#10b981" 
                  name="Confidence"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
};

export default MetacognitiveCascade;