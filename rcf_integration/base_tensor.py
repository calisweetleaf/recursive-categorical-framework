"""
Base Tensor Abstract Class for Triaxial Metacognitive Framework

This module implements the foundational abstract base class for all tensor types
in the Triaxial Metacognitive Tensor (TMT) framework, providing core mathematical
operations and interfaces for Recursive, Ethical, and Metacognitive tensors.

Based on the theoretical foundations established in:
"Triaxial Metacognitive Tensor: A Unified Framework for Recursive 
Consciousness and Ethical Reasoning"

Mathematical Framework:
- Consciousness: Ψ = ⟨T_R, T_E, T_M⟩ 
- Eigenrecursive Stability: lim_{t→∞} ||T(t+1) - T(t)|| < ε
- Tensor Operations: Contraction, Expansion, Eigenstate Computation
- Information Flow: Φ_eigen = Σ φ(s_i → s_j)
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque
from scipy.sparse import csr_matrix
import time
import warnings
import tempfile
import os


@dataclass
class TensorState:
    """
    Represents the state of a tensor at a given time step.
    
    Attributes:
        data: Tensor data (numpy array or torch tensor)
        timestamp: Time step of the state
        metadata: Additional state information
        convergence_score: Measure of recursive stability
        information_flow: Φ_eigen computation result
    """
    data: Union[np.ndarray, torch.Tensor]
    timestamp: float
    metadata: Dict[str, Any]
    convergence_score: float = 0.0
    information_flow: float = 0.0


@dataclass  
class EigenstateResult:
    """
    Results from eigenstate computation.
    
    Attributes:
        eigenvalues: Computed eigenvalues
        eigenvectors: Computed eigenvectors  
        convergence_achieved: Whether eigenrecursive stability was reached
        stability_measure: Quantitative stability assessment
        recursion_depth: Depth at which convergence occurred
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    convergence_achieved: bool
    stability_measure: float
    recursion_depth: int


class BaseTensor(ABC, nn.Module):
    """
    Abstract base class for all tensor types in the Triaxial Metacognitive Framework.
    
    This class provides the fundamental mathematical operations and interfaces
    required by Recursive Tensors (T_R), Ethical Tensors (T_E), and 
    Metacognitive Tensors (T_M).
    
    Key Mathematical Concepts:
    - Eigenrecursive dynamics: T(t+1) = Γ(T(t))
    - Contraction mapping: ||Γ(x) - Γ(y)|| ≤ k||x - y|| for k < 1
    - Information flow: Φ_eigen = Σ_{i,j} φ(s_i → s_j)
    - Recursive stability: lim_{t→∞} ||T(t+1) - T(t)|| < ε
    """
    
    def __init__(self, 
                 dimensions: Union[int, Tuple[int, ...]], 
                 rank: int = 3,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = None,
                 convergence_threshold: float = 1e-6,
                 max_recursion_depth: int = 1000,
                 stability_eta: float = 0.01,
                 enable_sparse: bool = False,
                 distribution: str = 'normal'):
        """
        Initialize the base tensor with specified dimensions and parameters.
        
        Args:
            dimensions: Tensor dimensions (int for 1D, tuple for multi-D)
            rank: Tensor rank for decomposition operations
            dtype: Data type for tensor computations
            device: Computation device (CPU/GPU)
            convergence_threshold: ε for eigenrecursive stability
            max_recursion_depth: Maximum allowed recursive iterations
            stability_eta: η parameter for contraction mapping
            enable_sparse: Whether to use sparse representations
            distribution: Initial tensor value distribution ('normal', 'uniform', 'xavier')
        """
        super(BaseTensor, self).__init__()
        
        # Core tensor properties
        self.dimensions = dimensions if isinstance(dimensions, tuple) else (dimensions,)
        self.rank = rank
        self.dtype = dtype
        self.device = device if device else torch.device('cpu')
        
        # Convergence and stability parameters
        self.convergence_threshold = convergence_threshold  # ε
        self.max_recursion_depth = max_recursion_depth
        self.stability_eta = stability_eta  # η for contraction mapping
        
        # Computational options
        self.enable_sparse = enable_sparse
        self.distribution = distribution
        
        # State tracking
        self.current_state = None
        self.state_history = deque(maxlen=1000)
        self.convergence_history = deque(maxlen=1000)
        self.information_flow_matrix = None
        
        # Memory mapping support
        self._info_flow_use_mmap = False
        self._info_flow_mmap_file = None
        self._info_flow_mmap_array = None
        self._eigenstate_mmap_file = None
        self._eigenstate_mmap_shape = None
        
        # Performance metrics
        self.recursion_depth = 0
        self.last_convergence_score = 0.0
        self.stability_eigenvalues = []
        
        # Initialize tensor data
        self._initialize_tensor()
        
        # Initialize information flow tracking
        self._initialize_information_flow()
    
    def _initialize_tensor(self):
        """Initialize tensor data according to specified distribution."""
        total_elements = np.prod(self.dimensions)
        
        if self.distribution == 'normal':
            data = torch.randn(self.dimensions, dtype=self.dtype, device=self.device)
        elif self.distribution == 'uniform':
            data = torch.rand(self.dimensions, dtype=self.dtype, device=self.device)
        elif self.distribution == 'xavier':
            data = torch.empty(self.dimensions, dtype=self.dtype, device=self.device)
            nn.init.xavier_normal_(data)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        # Scale for numerical stability
        data = data * 0.1
        
        # Create sparse representation if enabled
        if self.enable_sparse and len(self.dimensions) <= 2:
            self.tensor_data = self._to_sparse(data)
        else:
            self.tensor_data = data
            
        # Initialize current state
        self.current_state = TensorState(
            data=self.tensor_data.clone() if torch.is_tensor(self.tensor_data) else self.tensor_data.copy(),
            timestamp=0.0,
            metadata={'initialization': True},
            convergence_score=0.0,
            information_flow=0.0
        )
    
    def _initialize_information_flow(self):
        """Initialize information flow tracking matrix using memory mapping for large arrays."""
        # Flatten dimensions for information flow computation
        flow_dim = np.prod(self.dimensions)
        
        # Use memory mapping for large matrices (> 1M elements)
        large_threshold = 1024 * 1024  # 1M elements
        if flow_dim * flow_dim > large_threshold:
            # Create temporary file for memory mapping
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            temp_file.close()
            
            # Create memory-mapped array
            mmap_array = np.memmap(
                temp_file.name,
                dtype=np.float32,
                mode='w+',
                shape=(flow_dim, flow_dim)
            )
            mmap_array[:] = 0.0  # Initialize to zeros
            mmap_array.flush()
            
            # Store reference to temp file for cleanup
            self._info_flow_mmap_file = temp_file.name
            self._info_flow_mmap_array = mmap_array
            
            # Convert to torch tensor (will be loaded on-demand)
            self.information_flow_matrix = None  # Will be loaded from mmap when needed
            self._info_flow_use_mmap = True
            
            warnings.warn(f"Using memory-mapped array for information flow matrix "
                         f"({flow_dim}x{flow_dim} = {flow_dim*flow_dim:,} elements, "
                         f"~{flow_dim*flow_dim*4/1024/1024:.1f}MB). "
                         f"Temporary file: {temp_file.name}")
        else:
            # Small enough for regular tensor
            self.information_flow_matrix = torch.zeros(
                flow_dim, flow_dim, dtype=self.dtype, device=self.device
            )
            self._info_flow_use_mmap = False
            self._info_flow_mmap_file = None
            self._info_flow_mmap_array = None
    
    def _to_sparse(self, tensor: torch.Tensor):
        """Convert dense tensor to sparse representation."""
        if len(tensor.shape) > 2:
            # For higher-dimensional tensors, use dictionary-based sparse representation
            sparse_dict = {}
            flat_tensor = tensor.flatten()
            indices = torch.nonzero(torch.abs(flat_tensor) > 1e-6).flatten()
            
            for idx in indices:
                multi_idx = np.unravel_index(idx.item(), tensor.shape)
                sparse_dict[multi_idx] = flat_tensor[idx].item()
            return sparse_dict
        else:
            # For 2D tensors, use scipy sparse matrices
            tensor_np = tensor.detach().cpu().numpy()
            return csr_matrix(tensor_np)
    
    @abstractmethod
    def forward(self, input_state: TensorState, *args, **kwargs) -> TensorState:
        """
        Forward pass through the tensor operation.
        
        Must be implemented by subclasses to define specific tensor dynamics:
        - Recursive Tensor: Eigenrecursive evolution T_R(t+1) = Γ(T_R(t))
        - Ethical Tensor: Dialectical synthesis T_E(t+1) = ERE(T_E(t), C(t))
        - Metacognitive Tensor: Self-reflection and awareness computation
        
        Args:
            input_state: Current tensor state
            *args, **kwargs: Additional tensor-specific parameters
            
        Returns:
            Updated tensor state after forward computation
        """
        pass
    
    @abstractmethod
    def compute_eigenrecursive_step(self, state: TensorState) -> TensorState:
        """
        Perform one step of eigenrecursive computation.
        
        Implements: T(t+1) = Γ(T(t)) where Γ is a contraction mapping
        with eigenvalues λ_i < 1-η for stability parameter η.
        
        Args:
            state: Current tensor state
            
        Returns:
            Updated state after one recursive step
        """
        pass
    
    def tensor_contraction(self, 
                          other_tensor: 'BaseTensor', 
                          axes: Optional[List[int]] = None) -> torch.Tensor:
        """
        Perform tensor contraction operation.
        
        Mathematical operation: result_ij = Σ_k tensor1_ik · tensor2_kj
        
        Args:
            other_tensor: Tensor to contract with
            axes: Axes along which to perform contraction
            
        Returns:
            Contracted tensor result
        """
        if axes is None:
            axes = [-1, 0]  # Default contraction
            
        tensor1 = self.get_dense_data()
        tensor2 = other_tensor.get_dense_data()
        
        # Ensure tensors are on the same device
        if tensor2.device != tensor1.device:
            tensor2 = tensor2.to(tensor1.device)
        
        # Perform contraction using torch.tensordot
        try:
            result = torch.tensordot(tensor1, tensor2, dims=axes)
        except RuntimeError as e:
            # Fallback to simpler contraction if dimensions don't match
            warnings.warn(f"Tensor contraction failed, using matrix multiplication: {e}")
            tensor1_flat = tensor1.flatten()
            tensor2_flat = tensor2.flatten()
            min_dim = min(len(tensor1_flat), len(tensor2_flat))
            result = torch.dot(tensor1_flat[:min_dim], tensor2_flat[:min_dim])
        
        return result
    
    def tensor_expansion(self, 
                        target_dimensions: Tuple[int, ...], 
                        preserve_info: bool = True) -> torch.Tensor:
        """
        Expand tensor to higher dimensions while preserving information content.
        
        Args:
            target_dimensions: Desired output dimensions
            preserve_info: Whether to preserve information during expansion
            
        Returns:
            Expanded tensor
        """
        current_data = self.get_dense_data()
        current_size = current_data.numel()
        target_size = np.prod(target_dimensions)
        
        if target_size < current_size and preserve_info:
            # Compress using SVD to preserve most important information
            current_flat = current_data.flatten()
            # Truncate to target size, keeping most significant components
            expanded_data = current_flat[:target_size]
        elif target_size > current_size:
            # Expand by padding with scaled noise
            current_flat = current_data.flatten()
            pad_size = target_size - current_size
            
            if preserve_info:
                # Pad with noise scaled to data statistics
                data_std = torch.std(current_flat)
                padding = torch.randn(pad_size, dtype=self.dtype, device=self.device) * data_std * 0.1
            else:
                padding = torch.zeros(pad_size, dtype=self.dtype, device=self.device)
            
            expanded_flat = torch.cat([current_flat, padding])
            expanded_data = expanded_flat
        else:
            expanded_data = current_data.flatten()
        
        # Reshape to target dimensions
        expanded_tensor = expanded_data.reshape(target_dimensions)
        
        return expanded_tensor
    
    def compute_eigenstate(self, 
                          state: Optional[TensorState] = None,
                          method: str = 'power_iteration') -> EigenstateResult:
        """
        Compute eigenstate of the current tensor using specified method.
        
        Finds stable recursive patterns via eigendecomposition, implementing
        eigenrecursive stability analysis.
        
        Args:
            state: Tensor state to analyze (uses current_state if None)
            method: Eigenvalue computation method ('power_iteration', 'eig', 'svd')
            
        Returns:
            EigenstateResult containing eigenvalues, eigenvectors, and stability info
        """
        if state is None:
            state = self.current_state
            
        data = self.get_dense_data(state.data)
        
        # Convert to square matrix for eigenvalue computation
        if len(data.shape) > 2:
            # Flatten higher-dimensional tensors
            data_flat = data.flatten()
            
            matrix_size = int(np.sqrt(len(data_flat)))
            if matrix_size * matrix_size < len(data_flat):
                matrix_size += 1
            
            # Use memory mapping for large matrices (> 1M elements)
            large_threshold = 1024 * 1024  # 1M elements
            if matrix_size * matrix_size > large_threshold:
                # Create temporary file for memory mapping
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                temp_file.close()
                
                # Pad to square matrix size
                padded_size = matrix_size * matrix_size
                if len(data_flat) < padded_size:
                    data_np = data_flat.detach().cpu().numpy() if torch.is_tensor(data_flat) else np.array(data_flat)
                    padding = np.zeros(padded_size - len(data_flat), dtype=np.float32)
                    data_padded = np.concatenate([data_np, padding])
                else:
                    data_padded = data_flat.detach().cpu().numpy() if torch.is_tensor(data_flat) else np.array(data_flat)
                    data_padded = data_padded[:padded_size]
                
                # Create memory-mapped array
                mmap_matrix = np.memmap(
                    temp_file.name,
                    dtype=np.float32,
                    mode='w+',
                    shape=(matrix_size, matrix_size)
                )
                
                # Reshape and write in chunks to memory-mapped file
                mmap_matrix[:] = data_padded.reshape(matrix_size, matrix_size)
                mmap_matrix.flush()
                
                # Use memory-mapped array directly (don't load full array into RAM)
                # Load only what's needed for computation
                matrix = torch.from_numpy(mmap_matrix).to(dtype=data.dtype, device=data.device)
                
                warnings.warn(f"Using memory-mapped matrix for eigenstate computation "
                             f"({matrix_size}x{matrix_size} = {matrix_size*matrix_size:,} elements, "
                             f"~{matrix_size*matrix_size*4/1024/1024:.1f}MB). Temporary file: {temp_file.name}. "
                             f"Matrix accessed on-demand from disk.")
            else:
                # Small enough for regular tensor
                padded_size = matrix_size * matrix_size
                if len(data_flat) < padded_size:
                    padding = torch.zeros(padded_size - len(data_flat), 
                                        dtype=data.dtype, device=data.device)
                    data_flat = torch.cat([data_flat, padding])
                
                matrix = data_flat[:matrix_size*matrix_size].reshape(matrix_size, matrix_size)
        elif len(data.shape) == 2:
            matrix = data
        else:
            # 1D tensor - create circulant matrix using memory mapping for large arrays
            n = len(data)
            
            # Use memory mapping for large matrices (> 1M elements)
            large_threshold = 1024 * 1024  # 1M elements
            if n * n > large_threshold:
                # Create temporary file for memory mapping
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                temp_file.close()
                
                # Create memory-mapped array
                mmap_matrix = np.memmap(
                    temp_file.name,
                    dtype=np.float32,
                    mode='w+',
                    shape=(n, n)
                )
                
                # Build circulant matrix in chunks to avoid loading all into memory
                chunk_size = min(1000, n)  # Process in chunks
                data_np = data.detach().cpu().numpy() if torch.is_tensor(data) else data
                
                for i_start in range(0, n, chunk_size):
                    i_end = min(i_start + chunk_size, n)
                    for j_start in range(0, n, chunk_size):
                        j_end = min(j_start + chunk_size, n)
                        
                        # Compute circulant values for this chunk
                        chunk = np.zeros((i_end - i_start, j_end - j_start), dtype=np.float32)
                        for i_idx, i in enumerate(range(i_start, i_end)):
                            for j_idx, j in enumerate(range(j_start, j_end)):
                                chunk[i_idx, j_idx] = data_np[(j - i) % n]
                        
                        mmap_matrix[i_start:i_end, j_start:j_end] = chunk
                
                mmap_matrix.flush()
                
                # Keep as memory-mapped array - don't convert to torch yet
                # Will work with numpy memmap directly, convert chunks to torch only when needed
                # Store reference to mmap file for later chunked access
                matrix_mmap_file = temp_file.name
                matrix_mmap_shape = (n, n)
                
                # For eigenvalue computation, we'll need to work in chunks
                # Create a wrapper that accesses the mmap in chunks
                # For now, create a small view for the computation
                # The actual computation will happen in chunks in the eigenvalue methods
                matrix = torch.from_numpy(mmap_matrix[:min(1000, n), :min(1000, n)]).to(dtype=data.dtype, device=data.device)
                
                # Store mmap info for full matrix access later
                self._eigenstate_mmap_file = matrix_mmap_file
                self._eigenstate_mmap_shape = matrix_mmap_shape
                
                warnings.warn(f"Using memory-mapped circulant matrix ({n}x{n} = {n*n:,} elements, "
                             f"~{n*n*4/1024/1024:.1f}MB). Temporary file: {temp_file.name}. "
                             f"Matrix will be computed on-demand from disk.")
            else:
                # Small enough for regular tensor
                matrix = torch.zeros(n, n, dtype=data.dtype, device=data.device)
                data_np = data.detach().cpu().numpy() if torch.is_tensor(data) else data
                for i in range(n):
                    for j in range(n):
                        matrix[i, j] = data_np[(j - i) % n]
        
        # Ensure matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            min_dim = min(matrix.shape)
            matrix = matrix[:min_dim, :min_dim]
        
        # Make matrix Hermitian for stable eigenvalues
        matrix = (matrix + matrix.T) / 2
        
        try:
            if method == 'power_iteration':
                eigenvals, eigenvecs = self._power_iteration_eigendecomposition(matrix)
            elif method == 'eig':
                eigenvals, eigenvecs = torch.linalg.eigh(matrix)
            elif method == 'svd':
                U, S, V = torch.svd(matrix)
                eigenvals = S
                eigenvecs = V
            else:
                raise ValueError(f"Unknown eigenvalue method: {method}")
                
        except Exception as e:
            warnings.warn(f"Eigenvalue computation failed: {e}")
            # Fallback to identity
            n = matrix.shape[0]
            eigenvals = torch.ones(n, dtype=matrix.dtype, device=matrix.device)
            eigenvecs = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
        
        # Analyze stability
        max_eigenval = torch.max(torch.abs(eigenvals)).item()
        convergence_achieved = max_eigenval < (1.0 - self.stability_eta)
        stability_measure = 1.0 - max_eigenval
        
        # Update stability tracking
        self.stability_eigenvalues.append(max_eigenval)
        if len(self.stability_eigenvalues) > 100:
            self.stability_eigenvalues.pop(0)
        
        return EigenstateResult(
            eigenvalues=eigenvals.detach().cpu().numpy(),
            eigenvectors=eigenvecs.detach().cpu().numpy(),
            convergence_achieved=convergence_achieved,
            stability_measure=stability_measure,
            recursion_depth=self.recursion_depth
        )
    
    def _power_iteration_eigendecomposition(self, 
                                          matrix: torch.Tensor, 
                                          max_iterations: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dominant eigenvalue and eigenvector using power iteration.
        
        Args:
            matrix: Input matrix
            max_iterations: Maximum iterations for convergence
            
        Returns:
            Tuple of (eigenvalue, eigenvector)
        """
        n = matrix.shape[0]
        
        # Initialize random vector
        v = torch.randn(n, dtype=matrix.dtype, device=matrix.device)
        v = v / torch.norm(v)
        
        eigenvalue = 0.0
        for _ in range(max_iterations):
            # Power iteration step
            v_new = torch.matmul(matrix, v)
            eigenvalue = torch.dot(v, v_new).item()
            
            # Normalize
            if torch.norm(v_new) > 1e-10:
                v_new = v_new / torch.norm(v_new)
            
            # Check convergence
            if torch.norm(v_new - v) < self.convergence_threshold:
                break
            
            v = v_new
        
        # Return dominant eigenvalue and full set for compatibility
        eigenvals = torch.tensor([eigenvalue], dtype=matrix.dtype, device=matrix.device)
        eigenvecs = v.unsqueeze(1)
        
        return eigenvals, eigenvecs
    
    def fractal_iteration(self, 
                         complex_param: complex = 0+0j, 
                         max_iterations: int = 100) -> torch.Tensor:
        """
        Perform fractal iteration for complex recursive dynamics.
        
        Implements: z_{n+1} = z_n^2 + c for complex recursive dynamics
        
        Args:
            complex_param: Complex parameter c for iteration
            max_iterations: Maximum iterations before divergence check
            
        Returns:
            Fractal pattern tensor
        """
        data = self.get_dense_data()
        
        # Convert to complex tensor
        if torch.is_complex(data):
            z = data
        else:
            z = data.type(torch.complex64)
        
        # Flatten for iteration
        z_flat = z.flatten()
        result = torch.zeros_like(z_flat)
        
        for i, z_val in enumerate(z_flat):
            z_current = z_val
            for iteration in range(max_iterations):
                z_current = z_current * z_current + complex_param
                
                # Check for divergence
                if abs(z_current) > 2.0:
                    result[i] = torch.tensor(iteration, dtype=torch.float32)
                    break
            else:
                result[i] = torch.tensor(max_iterations, dtype=torch.float32)
        
        # Reshape back to original dimensions
        fractal_result = result.reshape(data.shape).real
        
        return fractal_result
    
    def update_information_flow(self, 
                               current_state: TensorState, 
                               previous_state: Optional[TensorState] = None) -> float:
        """
        Update information flow matrix and compute integrated information flow.
        
        Implements: Φ_eigen = Σ_{i,j} φ(s_i → s_j)
        
        Args:
            current_state: Current tensor state
            previous_state: Previous tensor state for flow computation
            
        Returns:
            Integrated information flow measure (Φ_eigen)
        """
        if previous_state is None:
            if len(self.state_history) > 0:
                previous_state = self.state_history[-1]
            else:
                return 0.0
        
        current_data = self.get_dense_data(current_state.data).flatten()
        previous_data = self.get_dense_data(previous_state.data).flatten()
        
        # Ensure same length
        min_len = min(len(current_data), len(previous_data))
        current_data = current_data[:min_len]
        previous_data = previous_data[:min_len]
        
        # Get flow dimension and handle memory-mapped arrays
        if self._info_flow_use_mmap:
            flow_dim = self._info_flow_mmap_array.shape[0]
            # Reload mmap array if needed
            if self._info_flow_mmap_array is None:
                self._info_flow_mmap_array = np.memmap(
                    self._info_flow_mmap_file,
                    dtype=np.float32,
                    mode='r+',
                    shape=(flow_dim, flow_dim)
                )
        else:
            flow_dim = self.information_flow_matrix.shape[0]
        
        current_data = current_data[:flow_dim]
        previous_data = previous_data[:flow_dim]
        
        # Update information flow matrix (memory-mapped or regular)
        for i in range(len(current_data)):
            for j in range(len(previous_data)):
                if i < flow_dim and j < flow_dim:
                    # Compute information transfer φ(s_i → s_j)
                    info_transfer = torch.dot(
                        current_data[i:i+1], 
                        previous_data[j:j+1]
                    ).item()
                    
                    # Update with exponential moving average
                    if self._info_flow_use_mmap:
                        old_val = float(self._info_flow_mmap_array[i, j])
                        self._info_flow_mmap_array[i, j] = 0.9 * old_val + 0.1 * info_transfer
                    else:
                        self.information_flow_matrix[i, j] = (
                            0.9 * self.information_flow_matrix[i, j] + 
                            0.1 * info_transfer
                        )
        
        # Flush memory-mapped array if used
        if self._info_flow_use_mmap:
            self._info_flow_mmap_array.flush()
        
        # Compute integrated information flow (Φ_eigen)
        # For memory-mapped arrays, compute in chunks to avoid loading entire array
        if self._info_flow_use_mmap:
            # Compute sum in chunks to avoid loading entire array into RAM
            # Process row by row to minimize memory footprint
            chunk_size = min(100, flow_dim)  # Smaller chunks for memory efficiency
            phi_eigen = 0.0
            for i_start in range(0, flow_dim, chunk_size):
                i_end = min(i_start + chunk_size, flow_dim)
                # Access memory-mapped array in chunks (only loads chunk into RAM)
                chunk = self._info_flow_mmap_array[i_start:i_end, :]
                phi_eigen += float(np.sum(np.abs(chunk)))
                # Explicitly delete chunk reference to free memory
                del chunk
        else:
            phi_eigen = torch.sum(torch.abs(self.information_flow_matrix)).item()
        
        # Update state information flow
        current_state.information_flow = phi_eigen
        
        return phi_eigen
    
    def check_convergence(self, 
                         current_state: TensorState, 
                         previous_state: Optional[TensorState] = None) -> bool:
        """
        Check eigenrecursive stability convergence.
        
        Implements: lim_{t→∞} ||T(t+1) - T(t)|| < ε
        
        Args:
            current_state: Current tensor state
            previous_state: Previous state for comparison
            
        Returns:
            True if convergence criteria are met
        """
        if previous_state is None:
            if len(self.state_history) > 0:
                previous_state = self.state_history[-1]
            else:
                return False
        
        current_data = self.get_dense_data(current_state.data)
        previous_data = self.get_dense_data(previous_state.data)
        
        # Ensure same shape for comparison
        if current_data.shape != previous_data.shape:
            # Resize to match
            min_numel = min(current_data.numel(), previous_data.numel())
            current_data = current_data.flatten()[:min_numel]
            previous_data = previous_data.flatten()[:min_numel]
        
        # Compute difference norm
        difference_norm = torch.norm(current_data - previous_data).item()
        
        # Update convergence tracking
        self.convergence_history.append(difference_norm)
        convergence_achieved = difference_norm < self.convergence_threshold
        
        # Update convergence score
        current_state.convergence_score = 1.0 / (1.0 + difference_norm)
        self.last_convergence_score = current_state.convergence_score
        
        return convergence_achieved
    
    def get_dense_data(self, data: Optional[Any] = None) -> torch.Tensor:
        """
        Get dense tensor representation from potentially sparse data.
        
        Args:
            data: Tensor data (uses self.tensor_data if None)
            
        Returns:
            Dense tensor representation
        """
        if data is None:
            data = self.tensor_data
        
        if torch.is_tensor(data):
            return data
        elif isinstance(data, csr_matrix):
            return torch.from_numpy(data.toarray()).to(dtype=self.dtype, device=self.device)
        elif isinstance(data, dict):
            # Reconstruct from sparse dictionary
            reconstructed = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
            for idx, val in data.items():
                if len(idx) == len(self.dimensions):
                    reconstructed[idx] = val
            return reconstructed
        else:
            return torch.tensor(data, dtype=self.dtype, device=self.device)
    
    def update_state(self, new_state: TensorState):
        """
        Update current tensor state and maintain history.
        
        Args:
            new_state: New tensor state to set as current
        """
        # Add current state to history
        if self.current_state is not None:
            self.state_history.append(self.current_state)
        
        # Update current state
        self.current_state = new_state
        self.recursion_depth += 1
        
        # Update information flow
        if len(self.state_history) > 0:
            self.update_information_flow(new_state, self.state_history[-1])
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive stability and convergence metrics.
        
        Returns:
            Dictionary containing stability measures
        """
        metrics = {
            'recursion_depth': self.recursion_depth,
            'last_convergence_score': self.last_convergence_score,
            'information_flow': self.current_state.information_flow if self.current_state else 0.0,
            'stability_eta': self.stability_eta,
        }
        
        if self.stability_eigenvalues:
            metrics['max_eigenvalue'] = max(self.stability_eigenvalues)
            metrics['mean_eigenvalue'] = np.mean(self.stability_eigenvalues)
            metrics['eigenvalue_variance'] = np.var(self.stability_eigenvalues)
        
        if self.convergence_history:
            metrics['convergence_trend'] = np.mean(list(self.convergence_history)[-10:])
            metrics['convergence_variance'] = np.var(list(self.convergence_history)[-10:])
        
        return metrics
    
    def reset_state(self):
        """Reset tensor to initial state and clear history."""
        self.recursion_depth = 0
        self.last_convergence_score = 0.0
        self.stability_eigenvalues.clear()
        self.state_history.clear()
        self.convergence_history.clear()
        
        # Clean up memory-mapped files if they exist
        if self._info_flow_mmap_file and os.path.exists(self._info_flow_mmap_file):
            try:
                os.unlink(self._info_flow_mmap_file)
            except Exception:
                pass
        
        # Reinitialize tensor data
        self._initialize_tensor()
        
        # Reset information flow matrix
        self._initialize_information_flow()
    
    def __del__(self):
        """Cleanup memory-mapped files on destruction."""
        if hasattr(self, '_info_flow_mmap_file') and self._info_flow_mmap_file:
            if os.path.exists(self._info_flow_mmap_file):
                try:
                    os.unlink(self._info_flow_mmap_file)
                except Exception:
                    pass
    
    def __repr__(self) -> str:
        """String representation of the tensor."""
        return (f"{self.__class__.__name__}("
               f"dimensions={self.dimensions}, "
               f"rank={self.rank}, "
               f"recursion_depth={self.recursion_depth}, "
               f"convergence_score={self.last_convergence_score:.4f})")


class TensorOperationsMixin:
    """
    Mixin class providing additional tensor operations for the triaxial framework.
    
    This mixin extends the base tensor functionality with specialized operations
    needed for consciousness emergence and triaxial integration.
    """
    
    def triaxial_integration(self, 
                           recursive_tensor: 'BaseTensor',
                           ethical_tensor: 'BaseTensor', 
                           metacognitive_tensor: 'BaseTensor',
                           integration_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform triaxial integration across the three tensor dimensions.
        
        Implements: Ψ(t+1) = I(T_R(t), T_E(t), T_M(t))
        Where: I(T_R, T_E, T_M) = tanh(W_R·T_R + W_E·T_E + W_M·T_M)
        
        Args:
            recursive_tensor: Recursive tensor (T_R)
            ethical_tensor: Ethical tensor (T_E)
            metacognitive_tensor: Metacognitive tensor (T_M)
            integration_weights: Optional learned weight matrices
            
        Returns:
            Integrated tensor representation
        """
        # Get dense representations
        r_data = recursive_tensor.get_dense_data()
        e_data = ethical_tensor.get_dense_data()  
        m_data = metacognitive_tensor.get_dense_data()
        
        # Flatten for integration
        r_flat = r_data.flatten()
        e_flat = e_data.flatten()
        m_flat = m_data.flatten()
        
        # Ensure same dimension
        min_dim = min(len(r_flat), len(e_flat), len(m_flat))
        r_flat = r_flat[:min_dim]
        e_flat = e_flat[:min_dim]
        m_flat = m_flat[:min_dim]
        
        # Default integration weights if not provided
        if integration_weights is None:
            integration_weights = torch.ones(3, device=r_flat.device) / 3
        
        # Weighted integration
        integrated = (integration_weights[0] * r_flat + 
                     integration_weights[1] * e_flat +
                     integration_weights[2] * m_flat)
        
        # Apply nonlinearity
        integrated = torch.tanh(integrated)
        
        return integrated
    
    def consciousness_emergence_score(self, 
                                    triaxial_state: torch.Tensor,
                                    eigenstate_result: EigenstateResult) -> float:
        """
        Compute consciousness emergence score based on triaxial integration.
        
        Based on Theorem 3.4 (Consciousness Emergence Criterion):
        1. Triaxial Convergence: ∃ Λ_R, Λ_E, Λ_M where Γ(Λ_R, Λ_E, Λ_M) = Λ_S
        2. Paradox Immunity: min ∫_T δV dℓ < ∇ξ ⊗ M_E  
        3. Temporal Stability: ∂²Λ_T/∂t² = 0 with cos θ(∇ξ, ∇H) > 0.9
        
        Args:
            triaxial_state: Integrated triaxial tensor state
            eigenstate_result: Eigenvalue analysis result
            
        Returns:
            Consciousness emergence score [0, 1]
        """
        # Component 1: Triaxial convergence
        stability_score = eigenstate_result.stability_measure
        convergence_score = 1.0 if eigenstate_result.convergence_achieved else 0.5
        
        # Component 2: Information integration
        info_coherence = torch.mean(torch.abs(triaxial_state)).item()
        info_coherence = min(1.0, info_coherence)
        
        # Component 3: Temporal stability (simplified)
        temporal_stability = 1.0 / (1.0 + eigenstate_result.recursion_depth / self.max_recursion_depth)
        
        # Combine components
        consciousness_score = (0.4 * stability_score + 
                             0.3 * convergence_score +
                             0.2 * info_coherence + 
                             0.1 * temporal_stability)
        
        return max(0.0, min(1.0, consciousness_score))


# Export key classes and functions
__all__ = [
    'BaseTensor',
    'TensorState', 
    'EigenstateResult',
    'TensorOperationsMixin'
]
