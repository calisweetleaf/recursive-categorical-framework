#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RecursiveTensor Implementation

A complete implementation of recursive tensor fields with dynamic contractions,
supporting higher-dimensional processing and hyperdimensional embedding.
This can serve as a model agnostic processing engine and is a computational agnostic engine for higher dimensional recursive processing for complex data

This implementation is based on the Eigenrecursion architecture and the triaxial fiber bundle recursive axis

The Recursive Tensor Architecture (RTA) introduces a fundamentally new approach to representing and manipulating high-dimensional data structures within machine learning systems. Unlike traditional tensor representations that treat data as static n-dimensional arrays, recursive tensors incorporate mechanisms for self-reference, pattern recursion, and dimensional interdependence that enable a new class of computational capabilities.
By formalizing a 5-dimensional sparse structure with specific semantic meanings assigned to each dimension, RTA enables models to efficiently represent and manipulate hierarchical, self-similar, and recursively defined patterns. This architecture serves as both a theoretical framework and a practical implementation guide for systems requiring advanced pattern recognition, iterative refinement, and emergent information processing capabilities.
"""

import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from numpy.fft import fft, ifft
import time
import logging
import os
import json
import gzip
import uuid
import pickle
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Callable, Union, Any, BinaryIO
from collections import deque

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("PyTorch not available - some features will be limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RecursiveTensor")

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


class RecursiveTensor:
    """
    Implementation of recursive tensor fields with dynamic contractions
    supporting higher-dimensional processing and hyperdimensional embedding.
    
    RecursiveTensors enable operations on multi-dimensional data with support for:
    - Sparse and dense representations
    - Tensor contractions and expansions
    - Projections onto subspaces
    - Embeddings into higher dimensions
    - Linear transformations
    - Eigenstate computation
    - Fractal iterations
    - Visualizations
    - Serialization
    
    The class maintains a computational history and supports both real and complex values.
    """
    def __init__(self, dimensions, rank=4, dtype=np.float32, distribution='normal', sparsity=0.1):
        """
        Initialize a recursive tensor structure.
        
        Args:
            dimensions: int - Dimensions of the tensor
            rank: int - Tensor rank (number of indices)
            dtype: data type for tensor elements (e.g., np.float32, np.complex64)
            distribution: str - Initial distribution ('normal', 'uniform', 'power_law')
            sparsity: float - Target sparsity level (0-1)
        """
        self.dimensions = dimensions
        self.rank = rank
        self.dtype = dtype
        self.distribution = distribution
        self.sparsity = sparsity
        self.creation_time = time.time()
        self.uuid = str(uuid.uuid4())
        
        # Initialize tensor data
        if distribution == 'normal':
            shape = tuple([self.dimensions] * self.rank)
            # Use sparse representation for efficiency
            num_elements = int(self.dimensions**self.rank * (1-self.sparsity))
            if num_elements > 0:
                values = np.random.normal(0, 1/np.sqrt(self.dimensions), num_elements)
                # Ensure indices are within bounds
                indices = np.array([np.random.randint(0, self.dimensions, len(values)) for _ in range(self.rank)]).T
                self.data = self._construct_sparse_tensor(indices, values, shape)
            else:
                # Handle case where no elements should be non-zero
                if self.rank <= 2:
                    self.data = csr_matrix(shape[:2]) if len(shape) >= 2 else csr_matrix((0, 0))
                else:
                    self.data = {}
        elif distribution == 'uniform':
            self.data = np.random.uniform(-0.01, 0.01, tuple([self.dimensions] * self.rank)).astype(self.dtype)
        elif distribution == 'power_law':
            # Power law initialization for scale-free properties
            base = np.random.power(2.5, tuple([self.dimensions] * self.rank)).astype(self.dtype) * 0.1
            signs = np.random.choice([-1, 1], tuple([self.dimensions] * self.rank))
            self.data = base * signs
        elif distribution == 'complex_gaussian':
            real_part = np.random.normal(0, 1/np.sqrt(2*self.dimensions), tuple([self.dimensions] * self.rank))
            imag_part = np.random.normal(0, 1/np.sqrt(2*self.dimensions), tuple([self.dimensions] * self.rank))
            self.data = (real_part + 1j * imag_part).astype(np.complex64)
        elif distribution == 'orthogonal':
            if self.rank >= 2:
                random_matrix = np.random.normal(0, 1, (self.dimensions, self.dimensions))
                q, _ = np.linalg.qr(random_matrix)
                
                # Extend to full tensor
                shape = tuple([self.dimensions] * self.rank)
                full_tensor = np.zeros(shape)
                for i in range(self.dimensions):
                    idx = (i,) + (0,) * (self.rank - 1)
                    for j in range(self.dimensions):
                        idx2 = (i, j) + (0,) * (self.rank - 2)
                        if self.rank >= 2:
                            full_tensor[idx2] = q[i, j]
                self.data = full_tensor
            else:
                # Fall back to normal for rank < 2
                self.data = np.random.normal(0, 1/np.sqrt(self.dimensions), tuple([self.dimensions] * self.rank)).astype(self.dtype)
        else:
            raise ValueError(f"Unknown distribution type: {distribution}")
        
        # Core operations dictionary: maps operation names to functions
        self.operations = {
            'contract': self.contract,
            'expand': self.expand,
            'project': self.project,
            'embed': self.embed,
            'transform': self.transform
        }
        
        # Pattern and reference tracking
        self._references = []
        self._patterns = []
        
        # Computational history for recursive operations
        self.operation_history = []
        
        # Eigenstate cache for efficiency
        self._eigenstate_cache = {}
        
        # Metadata for tracking
        self.metadata = {
            "id": self.uuid,  # Added id
            "created_at": self.creation_time,  # Renamed 'created' to 'created_at'
            "modified": self.creation_time,
            "operations_count": 0,
            "description": f"RecursiveTensor({self.dimensions}, rank={self.rank})"
        }
        
        logger.debug(f"Created {distribution} RecursiveTensor with dimensions={self.dimensions}, rank={self.rank}")

    @property
    def shape(self):
        """Return the shape of the tensor."""
        if self.rank == 0: # Scalar case
            return ()
        return tuple([self.dimensions] * self.rank)
    
    def _construct_sparse_tensor(self, indices, values, shape):
        """
        Construct a sparse tensor representation from indices and values
        
        Args:
            indices: array of indices
            values: array of values
            shape: target tensor shape
            
        Returns:
            Sparse tensor representation (matrix or dictionary)
        """
        if len(shape) <= 2:
            # For rank 2 or less, use scipy sparse matrix
            rows, cols = indices[:, 0], indices[:, 1]
            return csr_matrix((values, (rows, cols)), shape=shape[:2])
        else:
            # For higher ranks, use dictionary-based sparse representation
            tensor = {}
            for idx, val in zip(map(tuple, indices), values):
                # Validate that all indices are within bounds
                if all(0 <= i < self.dimensions for i in idx) and abs(val) > 1e-6:  # Threshold to maintain sparsity
                    tensor[idx] = val
            return tensor
    
    def _sparse_to_dense(self, sparse_dict):
        """
        Convert sparse dictionary representation to dense tensor
        
        Args:
            sparse_dict: Dictionary-based sparse tensor
            
        Returns:
            numpy.ndarray: Dense tensor
        """
        # Determine shape from the keys
        if not sparse_dict:
            return np.zeros((self.dimensions,) * self.rank, dtype=self.dtype)
            
        max_indices = tuple(max(idx[i] for idx in sparse_dict.keys()) + 1 
                           for i in range(len(next(iter(sparse_dict.keys())))))
        dense = np.zeros(max_indices, dtype=self.dtype)
        for idx, val in sparse_dict.items():
            dense[idx] = val
        return dense
    
    def contract(self, other_tensor, axes=((0,), (0,))):
        """
        Contract this tensor with another tensor along specified axes.
        
        Args:
            other_tensor: RecursiveTensor or numpy.ndarray
            axes: tuple of axis specifications for contraction
            
        Returns:
            RecursiveTensor: Result of contraction
        """
        self.operation_history.append(('contract', id(other_tensor), axes))
        self.metadata["operations_count"] += 1
        self.metadata["modified"] = time.time()
        
        # Handle different tensor types
        if isinstance(other_tensor, RecursiveTensor):
            other_data = other_tensor.data
        else:
            other_data = other_tensor
            
        # Perform tensor contraction
        if isinstance(self.data, dict) or isinstance(other_data, dict):
            # Sparse tensor contraction
            result_data = self._sparse_tensor_contract(self.data, other_data, axes)
        else:
            # Ensure both inputs are numpy arrays for tensordot
            data1 = self.data.toarray() if hasattr(self.data, 'toarray') else self.data
            data2 = other_data.toarray() if hasattr(other_data, 'toarray') else other_data
            result_data = np.tensordot(data1, data2, axes)
        
        # Create result tensor with appropriate dimensions
        if isinstance(result_data, np.ndarray):
            result_dims = result_data.shape[0] if result_data.ndim > 0 else 1
        else:
            if isinstance(other_tensor, RecursiveTensor):
                other_dim_for_max = other_tensor.dimensions
            else:
                other_dim_for_max = self.dimensions
            result_dims = max(self.dimensions, other_dim_for_max)
            
        result = RecursiveTensor(result_dims, 
                               rank=(result_data.ndim if isinstance(result_data, np.ndarray) 
                                     else self.rank - len(axes[0])),
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        result.data = result_data
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Contraction of {self.metadata['description']} with {getattr(other_tensor, 'metadata', {}).get('description', 'external tensor')}"
        if isinstance(other_tensor, RecursiveTensor):
            result.operation_history = self.operation_history.copy()
            result.operation_history.extend(other_tensor.operation_history)
        return result

    def _sparse_tensor_contract(self, tensor1, tensor2, axes):
        """
        Custom implementation of tensor contraction for sparse representations
        
        Args:
            tensor1: First tensor (sparse dict or ndarray)
            tensor2: Second tensor (sparse dict or ndarray)
            axes: Axes to contract
            
        Returns:
            Contracted tensor (sparse dict or ndarray)
        """
        if isinstance(tensor1, dict) and isinstance(tensor2, dict):
            result = {}
            for idx1, val1 in tensor1.items():
                for idx2, val2 in tensor2.items():
                    if all(idx1[ax1] == idx2[ax2] for ax1, ax2 in zip(axes[0], axes[1])):
                        result_idx = self._compute_result_idx(idx1, idx2, axes)
                        result[result_idx] = result.get(result_idx, 0) + val1 * val2
            return result
        else:
            if isinstance(tensor1, dict):
                tensor1 = self._sparse_to_dense(tensor1)
            if isinstance(tensor2, dict):
                tensor2 = self._sparse_to_dense(tensor2)
            return np.tensordot(tensor1, tensor2, axes)
    
    def _compute_result_idx(self, idx1, idx2, axes):
        """
        Helper to compute the resulting index in tensor contraction
        
        Args:
            idx1: Index from first tensor
            idx2: Index from second tensor
            axes: Contraction axes
            
        Returns:
            tuple: Resulting index
        """
        remaining_idx1 = [i for j, i in enumerate(idx1) if j not in axes[0]]
        remaining_idx2 = [i for j, i in enumerate(idx2) if j not in axes[1]]
        return tuple(remaining_idx1 + remaining_idx2)
    
    def expand(self, new_dimensions):
        """
        Expand tensor to higher dimensions through tensor product with basis.
        
        Args:
            new_dimensions: int or tuple - target dimension(s) to expand to
            
        Returns:
            RecursiveTensor: Expanded tensor
        """
        if isinstance(new_dimensions, int):
            new_dimensions = (new_dimensions,)
        if len(new_dimensions) > self.rank:
            raise ValueError("Cannot expand to more dimensions than the current rank")
        self.operation_history.append(('expand', new_dimensions))
        self.metadata["operations_count"] += 1
        self.metadata["modified"] = time.time()
        self.metadata["description"] = f"Expanded {self.metadata['description']} to {new_dimensions}"
        if isinstance(self.data, dict):
            new_data = {}
            for idx, val in self.data.items():
                new_idx = (0,) * len(new_dimensions) + idx
                new_data[new_idx] = val
        else:
            new_data = self.create_tensor_expansion(new_dimensions)
            
            
        max_dim = max(self.dimensions, max(new_dimensions))
        result = RecursiveTensor(max_dim, 
                               rank=self.rank + len(new_dimensions), 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        result.data = new_data
        result.operation_history = self.operation_history.copy()
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Expanded {self.metadata['description']} to include {new_dimensions} dimensions"
        return result

    def create_tensor_expansion(self, new_dimensions):
        expansion_shape = new_dimensions + self.data.shape
        expansion = np.ones(expansion_shape, dtype=self.dtype)
        new_data = np.tensordot(expansion, self.data, axes=0)
        new_data = new_data.reshape(expansion_shape)
        return new_data

    def project(self, subspace_basis, axes=(0,)):
        """
        Project tensor onto a subspace defined by basis vectors.
        
        Args:
            subspace_basis: array-like basis vectors
            axes: tuple - axes to project along
            
        Returns:
            RecursiveTensor: Projected tensor
        """
        self.operation_history.append(('project', subspace_basis.shape if hasattr(subspace_basis, 'shape') else None, axes))
        self.metadata["operations_count"] += 1
        self.metadata["modified"] = time.time()
        self.metadata["description"] = f"Projection of {self.metadata['description']} onto subspace"
        if not isinstance(subspace_basis, np.ndarray):
            subspace_basis = np.array(subspace_basis)
            
        if subspace_basis.ndim > 1:
            norms = np.linalg.norm(subspace_basis, axis=1)
            normalized_basis = subspace_basis / norms[:, np.newaxis]
        else:
            normalized_basis = subspace_basis / np.linalg.norm(subspace_basis)
        if isinstance(self.data, dict):
            projected_data = self._sparse_project(normalized_basis, axes)
        else:
            projected_data = self.data.copy()
            for ax in axes:
                tensor_shape = projected_data.shape
                reshaped = projected_data.swapaxes(0, ax).reshape(tensor_shape[ax], -1)
                projected = normalized_basis @ reshaped
                new_shape = list(tensor_shape)
                new_shape[ax] = normalized_basis.shape[0]
                projected_data = projected.reshape(new_shape).swapaxes(0, ax)
        
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        result.data = projected_data
        result.operation_history = self.operation_history.copy()
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Projection of {self.metadata['description']} onto subspace"
        
        return result
    
    def _sparse_project(self, basis, axes):
        """
        Project sparse tensor data onto basis
        
        Args:
            basis: Normalized basis vectors
            axes: Axes to project along
            
        Returns:
            dict: Projected sparse tensor
        """
        if len(axes) > 1:
            result = self.data.copy() if isinstance(self.data, dict) else {idx: val for idx, val in np.ndenumerate(self.data) if val != 0}
            for ax in axes:
                result = self._sparse_project_single_axis(result, basis, ax)
            return result
        else:
            return self._sparse_project_single_axis(self.data, basis, axes[0])
    
    def _sparse_project_single_axis(self, sparse_data, basis, axis):
        """
        Project sparse tensor data onto basis along a single axis
        
        Args:
            sparse_data: Sparse tensor data (dictionary)
            basis: Normalized basis vectors
            axis: Axis to project along
            
        Returns:
            dict: Projected sparse tensor
        """
        result = {}
        
        for idx, val in sparse_data.items():
            for b_idx, basis_vec in enumerate(basis):
                proj_idx = list(idx)
                proj_idx[axis] = b_idx
                proj_idx = tuple(proj_idx)
                
                proj_val = val * basis_vec[idx[axis]]
                if abs(proj_val) > 1e-10:  
                    result[proj_idx] = result.get(proj_idx, 0) + proj_val
                
        return result
    
    def embed(self, embedding_function, new_rank=None):
        """
        Embed tensor in higher-dimensional space using embedding function.
        
        Args:
            embedding_function: Callable - function to compute embedding
            new_rank: int - rank of resulting tensor (default: self.rank + 1)
            
        Returns:
            RecursiveTensor: Embedded tensor
        """
        if new_rank is None:
            new_rank = self.rank + 1
            
        self.operation_history.append(('embed', new_rank))
        self.metadata["operations_count"] += 1
        self.metadata["modified"] = time.time()
        
        result = RecursiveTensor(self.dimensions, 
                               rank=new_rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        
        if callable(embedding_function):
            if isinstance(self.data, dict):
                embedded_data = {}
                for idx, val in self.data.items():
                    embedded_val = embedding_function(val, idx)
                    if isinstance(embedded_val, dict):  
                        for e_idx, e_val in embedded_val.items():
                            if abs(e_val) > 1e-10:  
                                embedded_data[idx + e_idx] = e_val
                    else: 
                        if isinstance(embedded_val, (int, float, complex, np.number)):
                            if abs(embedded_val) > 1e-10:
                                embedded_data[idx + (0,)] = embedded_val
                result.data = embedded_data
            else:
                result.data = embedding_function(self.data)
        else:
            raise TypeError("embedding_function must be callable")
        
        result.operation_history = self.operation_history.copy()
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Embedding of {self.metadata['description']} to rank {new_rank}"
            
        return result
    
    def transform(self, transformation_matrix, axes=None):
        """
        Apply linear transformation to tensor.
        
        Args:
            transformation_matrix: array-like transformation matrix
            axes: tuple - axes to transform (default: all)
            
        Returns:
            RecursiveTensor: Transformed tensor
        """
        if axes is None:
            axes = tuple(range(self.rank))
            
        self.operation_history.append(('transform', transformation_matrix.shape if hasattr(transformation_matrix, 'shape') else None, axes))
        self.metadata["operations_count"] += 1
        self.metadata["modified"] = time.time()
        
        if not isinstance(transformation_matrix, np.ndarray):
            transformation_matrix = np.array(transformation_matrix)
        if transformation_matrix.ndim != 2:
            raise ValueError("Transformation matrix must be 2D")
        if transformation_matrix.shape[1] != self.dimensions:
            raise ValueError("Transformation matrix must match tensor dimensions")
        if transformation_matrix.shape[0] > self.dimensions:
            raise ValueError("Transformation matrix cannot expand tensor dimensions")
        if isinstance(self.data, dict):
            transformed_data = self._sparse_transform(transformation_matrix, axes)
        else:
            transformed_data = self.data.copy()
            for ax in axes:
                tensor_shape = transformed_data.shape
                reshaped = transformed_data.swapaxes(0, ax).reshape(tensor_shape[ax], -1)
                transformed = transformation_matrix @ reshaped
                new_shape = list(tensor_shape)
                new_shape[ax] = transformation_matrix.shape[0]
                transformed_data = transformed.reshape(new_shape).swapaxes(0, ax)
        
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        result.data = transformed_data
        result.operation_history = self.operation_history.copy()
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Transformation of {self.metadata['description']}"

        return result
    
    def _sparse_transform(self, matrix, axes):
        """
        Apply transformation to sparse tensor
        
        Args:
            matrix: Transformation matrix
            axes: Axes to transform
            
        Returns:
            dict: Transformed sparse tensor
        """
        result = {}
        
        for idx, val in self.data.items():
            for ax in axes:
                ax_idx = idx[ax]
                
                for out_idx in range(matrix.shape[0]):
                    trans_val = matrix[out_idx, ax_idx] * val
                    if abs(trans_val) > 1e-10:  
                        new_idx = list(idx)
                        new_idx[ax] = out_idx
                        new_idx = tuple(new_idx)
                        result[new_idx] = result.get(new_idx, 0) + trans_val
                        
        return result
    
    def compute_eigenstates(self, axes=(0, 1), k=6, convergence_threshold=1e-8):
        """
        Compute dominant eigenstates using the Eigenrecursion theorem.
        
        This method implements the recursive stability protocol described in the
        Eigenrecursion theorem, ensuring convergence to stable eigenstates through
        iterative refinement and fixed-point detection.
        
        Args:
            axes: tuple - axes to flatten for eigendecomposition
            k: int - number of eigenstates to compute
            convergence_threshold: float - threshold for eigenrecursion convergence
            
        Returns:
            tuple: (eigenvalues, eigenvectors) following eigenrecursion convergence
        """
        cache_key = (axes, k, convergence_threshold)
        if cache_key in self._eigenstate_cache:
            return self._eigenstate_cache[cache_key]
        
        # Step 1: Construct matrix representation for eigenrecursion
        if isinstance(self.data, dict):
            matrix = self._sparse_to_matrix(axes)
        else:
            # Flatten tensor along specified axes for eigenrecursion
            matrix_shape = [
                np.prod([self.data.shape[ax] for ax in axes[0]]),
                np.prod([self.data.shape[ax] for ax in axes[1]])
            ]
            matrix = self.data.transpose(axes[0] + axes[1]).reshape(matrix_shape)
        
        # Step 2: Apply eigenrecursion protocol for convergence
        if isinstance(matrix, csr_matrix) or matrix.shape[0] > 1000:
            try:
                # Sparse eigenvalue computation with eigenrecursion refinement
                eigenvalues, eigenvectors = eigsh(
                    matrix, 
                    k=min(k, matrix.shape[0]-1), 
                    which='LM',
                    tol=convergence_threshold
                )
            except Exception:
                if isinstance(matrix, csr_matrix):
                    matrix = matrix.toarray()
                eigenvalues, eigenvectors = self._eigenrecursion_solve(
                    matrix, k, convergence_threshold
                )
        else:
            # Dense eigenvalue computation with eigenrecursion
            eigenvalues, eigenvectors = self._eigenrecursion_solve(
                matrix, k, convergence_threshold
            )
        
        # Step 3: Eigenrecursion convergence verification
        # Ensure eigenvectors are properly normalized and oriented
        eigenvalues, eigenvectors = self._verify_eigenrecursion_convergence(
            matrix, eigenvalues, eigenvectors, convergence_threshold
        )
        
        # Step 4: Reshape eigenvectors back to tensor structure
        eigenvectors = self._reshape_eigenvectors_to_tensor(
            eigenvectors, axes, k
        )
        
        self._eigenstate_cache[cache_key] = (eigenvalues, eigenvectors)
        return eigenvalues, eigenvectors
    
    def _eigenrecursion_solve(self, matrix, k, threshold):
        """
        Solve eigenvalue problem using eigenrecursion theorem principles.
        
        Implements the recursive stability protocol to ensure convergence
        to true eigenstates through iterative refinement.
        """
        # Ensure matrix is Hermitian for real eigenvalues
        if not np.allclose(matrix, matrix.conj().T):
            matrix = (matrix + matrix.conj().T) / 2
            
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Sort by magnitude for dominant eigenstates
        idx = np.argsort(np.abs(eigenvalues))[::-1][:k]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _verify_eigenrecursion_convergence(self, matrix, eigenvalues, eigenvectors, threshold):
        """
        Verify eigenrecursion convergence using fixed-point detection.
        
        Implements the stability protocol from the Eigenrecursion theorem
        to ensure computed eigenstates are true fixed points.
        """
        # Check eigenvalue accuracy using Rayleigh quotient
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            rayleigh_quotient = np.vdot(v, matrix @ v) / np.vdot(v, v)
            
            # Apply eigenrecursion refinement if needed
            if abs(rayleigh_quotient - eigenvalues[i]) > threshold:
                # Refine using inverse iteration
                shifted_matrix = matrix - eigenvalues[i] * np.eye(matrix.shape[0])
                try:
                    refined_v = np.linalg.solve(shifted_matrix, v)
                    refined_v = refined_v / np.linalg.norm(refined_v)
                    eigenvectors[:, i] = refined_v
                    
                    # Recompute eigenvalue
                    eigenvalues[i] = np.vdot(refined_v, matrix @ refined_v)
                except np.linalg.LinAlgError:
                    # Singular matrix - use original eigenvector
                    pass
        
        return eigenvalues, eigenvectors
    
    def _reshape_eigenvectors_to_tensor(self, eigenvectors, axes, k):
        """
        Reshape eigenvectors back to tensor structure following eigenrecursion.
        
        Maps the flattened eigenvectors back to the original tensor space
        while preserving the eigenrecursion convergence properties.
        """
        if len(axes[1]) == 0:
            # Handle edge case
            return eigenvectors.T
            
        # Compute target shape for eigenvectors
        target_shape = [k] + [self.data.shape[ax] for ax in axes[1]]
        
        # Reshape each eigenvector
        reshaped_eigenvectors = []
        for i in range(k):
            vec = eigenvectors[:, i]
            tensor_vec = vec.reshape([self.data.shape[ax] for ax in axes[1]])
            reshaped_eigenvectors.append(tensor_vec)
        
        return np.array(reshaped_eigenvectors)
    
    def _sparse_to_matrix(self, axes):
        """
        Convert sparse tensor to matrix representation for eigenrecursion.
        
        Optimized sparse conversion that preserves the tensor structure
        needed for eigenrecursion convergence analysis.
        """
        dim1 = self.dimensions ** len(axes[0])
        dim2 = self.dimensions ** len(axes[1])
        
        matrix = lil_matrix((dim1, dim2))
        
        for idx, val in self.data.items():
            # Map tensor indices to matrix indices using eigenrecursion indexing
            row_idx = sum(idx[ax] * (self.dimensions ** i) 
                         for i, ax in enumerate(axes[0]))
            col_idx = sum(idx[ax] * (self.dimensions ** i) 
                         for i, ax in enumerate(axes[1]))
            matrix[row_idx, col_idx] = val
            
        return matrix.tocsr()
    
    def fractal_iteration(self, c_function, max_iter=10):
        """
        Apply fractal iteration to tensor values following Mandelbrot-Julia pattern.
        
        Args:
            c_function: Callable - function to generate c parameter
            max_iter: int - maximum iterations
            
        Returns:
            RecursiveTensor: Result of fractal iteration
        """
        self.operation_history.append(('fractal', max_iter))
        self.metadata["operations_count"] += 1
        self.metadata["modified"] = time.time()
        
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        
        if isinstance(self.data, dict):
            result_data = {}
            for idx, z_val in self.data.items():
                c_val = c_function(idx, z_val)
                for _ in range(max_iter):
                    z_val = z_val**2 + c_val
                    if abs(z_val) > 2:
                        break
                if abs(z_val) > 1e-10:  
                    result_data[idx] = z_val
            result.data = result_data
        else:
            z = self.data.copy()
            c = c_function(None, z)  
            
            for _ in range(max_iter):
                z = z**2 + c
                if np.issubdtype(z.dtype, np.complexfloating):
                    escaped = np.abs(z) > 2
                    if np.all(escaped):
                        break
            
            result.data = z
        
        result.operation_history = self.operation_history.copy()
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Fractal iteration of {self.metadata['description']} ({max_iter} steps)"
            
        return result
    def visualize_slice(self, axes=(0, 1), slice_indices=None, cmap='viridis', title=None, show=True):
        """
        Visualize a 2D slice of the tensor
        
        Args:
            axes: tuple - which two axes to use for the 2D visualization
            slice_indices: dict - indices for other axes
            cmap: str - matplotlib colormap name
            title: str - plot title
            show: bool - whether to display the plot (or just return the figure)
            
        Returns:
            matplotlib.figure.Figure: Figure object with the visualization
        """
        if self.rank < 2:
            raise ValueError("Tensor must have rank >= 2 for 2D visualization")
            
        if slice_indices is None:
            slice_indices = {i: 0 for i in range(self.rank) if i not in axes}
        
        if isinstance(self.data, dict):
            slice_matrix = np.zeros((self.dimensions, self.dimensions))
            for idx, val in self.data.items():
                if all(idx[ax] == slice_idx for ax, slice_idx in slice_indices.items()):
                    if len(idx) > max(axes) and idx[axes[0]] < self.dimensions and idx[axes[1]] < self.dimensions:
                        slice_matrix[idx[axes[0]], idx[axes[1]]] = val
        else:
            slice_idx = tuple(slice(None) if i in axes else slice_indices.get(i, 0) 
                             for i in range(self.rank))
            slice_matrix = self.data[slice_idx]
        
        plt.figure(figsize=(10, 8))
        if np.iscomplexobj(slice_matrix):
            im = plt.imshow(np.abs(slice_matrix), cmap=cmap)
            plt.colorbar(im, label='Magnitude')
        else:
            im = plt.imshow(np.real(slice_matrix), cmap=cmap)
            plt.colorbar(im, label='Value')
        
        plt.title(title or f"Tensor Slice along axes {axes}")
        plt.xlabel(f"Axis {axes[1]}")
        plt.ylabel(f"Axis {axes[0]}")
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return plt.gcf()
    def visualize_eigenspectrum(self, axes=(0, 1), k=10, show=True):
        """
        Visualize the eigenvalue spectrum of the tensor
        
        Args:
            axes: tuple - axes to use for eigendecomposition
            k: int - number of eigenvalues to show
            show: bool - whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: Figure with eigenspectrum visualization
        """
        try:
            eigenvalues, _ = self.compute_eigenstates(axes=axes, k=k)
            
            # Defensive check: ensure eigenvalues is array-like
            if not hasattr(eigenvalues, '__len__') or not hasattr(eigenvalues, '__iter__'):
                logger.warning("compute_eigenstates returned non-iterable eigenvalues")
                return None
            
            eigenvalues = np.asarray(eigenvalues)
            if eigenvalues.size == 0:
                logger.warning("No eigenvalues computed")
                return None
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(eigenvalues)+1), np.abs(eigenvalues), 'o-', markersize=8)
            plt.yscale('log')
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            plt.xlabel('Eigenvalue Index')
            plt.ylabel('Absolute Eigenvalue')
            plt.title('Eigenvalue Spectrum')
            
            if show:
                plt.show()
                
            return plt.gcf()
        except Exception as e:
            logger.warning(f"Failed to visualize eigenspectrum: {e}")
            return None
    
    def plot_operation_history(self, show=True):
        """
        Visualize the operation history of this tensor
        
        Args:
            show: bool - whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: Figure with operation history visualization
        """
        if not self.operation_history:
            print("No operation history available")
            return None
        
        op_counts = {}
        for op in self.operation_history:
            op_type = op[0]
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        plt.figure(figsize=(10, 6))
        plt.bar(op_counts.keys(), op_counts.values())
        plt.ylabel('Count')
        plt.title('Tensor Operation History')
        plt.grid(True, alpha=0.3)
        
        if show:
            plt.show()
            
        return plt.gcf()
    
    def compute_density_function(self, resolution=50, threshold=0.01):
        """
        Compute density function of tensor values
        
        Args:
            resolution: int - number of bins for histogram
            threshold: float - threshold for including values
            
        Returns:
            tuple: (bin_edges, histogram)
        """
        if isinstance(self.data, dict):
            values = np.array(list(self.data.values()))
        else:
            values = self.data.flatten()
        
        if np.iscomplexobj(values):
            values = np.abs(values)
        
        filtered_values = values[np.abs(values) > threshold]
        
        hist, bin_edges = np.histogram(filtered_values, bins=resolution, density=True)
        
        return bin_edges, hist
    
    def visualize_density(self, resolution=50, threshold=0.01, show=True):
        """
        Visualize density function of tensor values
        
        Args:
            resolution: int - number of bins for histogram
            threshold: float - threshold for including values
            show: bool - whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: Figure with density visualization
        """
        bin_edges, hist = self.compute_density_function(resolution, threshold)
        
        plt.figure(figsize=(10, 6))
        plt.stairs(hist, bin_edges)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Tensor Value Distribution')
        plt.grid(True, alpha=0.3)
        
        if show:
            plt.show()
            
        return plt.gcf()
    
    def compute_entropy(self):
        """
        Compute the information entropy of the tensor
        
        Returns:
            float: entropy value
        """
        if isinstance(self.data, dict):
            values = np.array(list(self.data.values()))
            if values.size == 0:
                return 0.0
        else:
            values = self.data.flatten()
            if values.size == 0:
                return 0.0
            values = values[values > 0]
        if np.iscomplexobj(values):
            values = np.abs(values)

        prob_dist = values / np.sum(values) if np.sum(values) > 0 else values
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))

        return entropy
    
    def normalize(self, norm_type=2):
        """
        Normalize the tensor
        
        Args:
            norm_type: int or str - type of normalization (1, 2, 'inf', 'fro')
            
        Returns:
            RecursiveTensor: normalized tensor
        """
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
                               
        if isinstance(self.data, dict):
            values = []
            for val in self.data.values():
                # Handle PyTorch tensors in sparse data
                try:
                    import torch
                    if isinstance(val, torch.Tensor):
                        if val.requires_grad:
                            values.append(val.detach().numpy())
                        else:
                            values.append(val.numpy())
                    else:
                        values.append(val)
                except ImportError:
                    values.append(val)
            
            values = np.array(values)
            if values.size == 0:
                result.data = {}
                return result
                
            if norm_type == 2:
                norm = np.sqrt(np.sum(np.abs(values)**2))
            elif norm_type == 1:
                norm = np.sum(np.abs(values))
            elif norm_type == 'inf':
                norm = np.max(np.abs(values))
            else:
                norm = np.sqrt(np.sum(np.abs(values)**2))
                
            if norm > 0:
                result.data = {}
                for idx, val in self.data.items():
                    try:
                        import torch
                        if isinstance(val, torch.Tensor):
                            result.data[idx] = val / norm
                        else:
                            result.data[idx] = val / norm
                    except ImportError:
                        result.data[idx] = val / norm
            else:
                result.data = self.data.copy()
        else:
            # Handle PyTorch tensors
            try:
                import torch
                if isinstance(self.data, torch.Tensor):
                    if self.data.requires_grad:
                        # Detach and convert to numpy for computation
                        data_np = self.data.detach().numpy()
                    else:
                        data_np = self.data.numpy()
                    
                    if self.data.size == 0:
                        result.data = torch.zeros_like(self.data)
                        return result
                        
                    norm = np.linalg.norm(data_np, ord=norm_type)
                    if norm > 0:
                        result.data = self.data / norm
                    else:
                        result.data = self.data.clone()
                else:
                    # NumPy array
                    if self.data.size == 0:
                        result.data = np.zeros_like(self.data)
                        return result
                        
                    norm = np.linalg.norm(self.data, ord=norm_type)
                    if norm > 0:
                        result.data = self.data / norm
                    else:
                        result.data = self.data.copy()
            except ImportError:
                # Fallback for numpy-only case
                if self.data.size == 0:
                    result.data = np.zeros_like(self.data)
                    return result
                    
                norm = np.linalg.norm(self.data, ord=norm_type)
                if norm > 0:
                    result.data = self.data / norm
                else:
                    result.data = self.data.copy()
        
        result.operation_history = self.operation_history.copy()
        result.operation_history.append(('normalize', norm_type))
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Normalized {self.metadata['description']}"
        
        return result
    
    def apply_function(self, func, threshold=None):
        """
        Apply a function elementwise to the tensor
        
        Args:
            func: callable - function to apply
            threshold: float - threshold for filtering values (optional)
            
        Returns:
            RecursiveTensor: transformed tensor
        """
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        
        if isinstance(self.data, dict):
            result_data = {}
            for idx, val in self.data.items():
                new_val = func(val)
                if threshold is None or abs(new_val) > threshold:
                    result_data[idx] = new_val
            result.data = result_data
        else:
            result.data = func(self.data)
            if threshold is not None:
                result.data[np.abs(result.data) <= threshold] = 0
        
        result.operation_history = self.operation_history.copy()
        result.operation_history.append(('apply_function', func.__name__ if hasattr(func, '__name__') else 'unnamed_function'))
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Function {func.__name__ if hasattr(func, '__name__') else 'unnamed_function'} applied to {self.metadata['description']}"
        result.sparsity = self.sparsity

        return result
    
    def compute_dict_compatibility(self, other_tensor):
        """
        Compute compatibility score between this tensor and another
        
        Args:
            other_tensor: RecursiveTensor - tensor to compare with
            
        Returns:
            float: compatibility score (0-1)
        """
        if not isinstance(other_tensor, RecursiveTensor):
            raise TypeError("Can only compute compatibility with another RecursiveTensor")
        
        data1 = self.data
        data2 = other_tensor.data
        
        if isinstance(data1, dict) and not isinstance(data2, dict):
            data2 = {idx: val for idx, val in np.ndenumerate(data2) if val != 0}
        elif not isinstance(data1, dict) and isinstance(data2, dict):
            data1 = {idx: val for idx, val in np.ndenumerate(data1) if val != 0}
            
        if isinstance(data1, dict) and isinstance(data2, dict):
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())
            if not keys1 and not keys2:
                return 1.0

            if not keys1 or not keys2:
                return 0.0

            common_keys = keys1.intersection(keys2)
            all_keys = keys1.union(keys2)
            key_similarity = len(common_keys) / len(all_keys)
            
            value_similarities = []
            for key in common_keys:
                val1 = data1[key]
                val2 = data2[key]
                
                if np.iscomplexobj(val1) or np.iscomplexobj(val2):
                    mag_sim = 1.0 - min(1.0, abs(abs(val1) - abs(val2)) / max(abs(val1), abs(val2), 1e-10))
                    
                    phase1 = np.angle(val1)
                    phase2 = np.angle(val2)
                    phase_diff = min(abs(phase1 - phase2), 2*np.pi - abs(phase1 - phase2)) / np.pi
                    phase_sim = 1.0 - phase_diff
                    
                    value_similarities.append(0.7 * mag_sim + 0.3 * phase_sim)
                else:
                    if max(abs(val1), abs(val2)) > 1e-10:
                        value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max(abs(val1), abs(val2))))
                    else:
                        value_similarities.append(1.0)
            
            if value_similarities:
                value_similarity = sum(value_similarities) / len(value_similarities)
                return 0.5 * key_similarity + 0.5 * value_similarity
            else:
                return key_similarity
        
        elif not isinstance(data1, dict) and not isinstance(data2, dict):
            data1_flat = data1.flatten()
            data2_flat = data2.flatten()
            
            norm1 = np.linalg.norm(data1_flat)
            norm2 = np.linalg.norm(data2_flat)
            
            if norm1 > 0 and norm2 > 0:
                dot_product = np.abs(np.vdot(data1_flat, data2_flat))
                return float(dot_product / (norm1 * norm2))
            elif norm1 == 0 and norm2 == 0:
                return 1.0  
            else:
                return 0.0  
        
        return 0.0
    
    def serialize(self):
        """
        Serialize tensor to dictionary
        
        Returns:
            dict: Serialized representation of the tensor
        """
        tensor_dict = {
            'dimensions': self.dimensions,
            'rank': self.rank,
            'distribution': self.distribution,
            'sparsity': self.sparsity,
            'operation_history': self.operation_history,
            'metadata': self.metadata,
            'version': '2.1.0',
            'timestamp': time.time(),
            'uuid': self.uuid
        }
        
        if isinstance(self.data, dict):
            tensor_dict['data_format'] = 'sparse'
            
            if np.iscomplexobj(next(iter(self.data.values()), 0)):
                tensor_dict['data'] = [
                    [list(idx), float(val.real), float(val.imag)] 
                    for idx, val in self.data.items()
                ]
                tensor_dict['complex'] = True
            else:
                tensor_dict['data'] = [
                    [list(idx), float(val)] 
                    for idx, val in self.data.items()
                ]
                tensor_dict['complex'] = False
        else:
            tensor_dict['data_format'] = 'dense'
            if np.iscomplexobj(self.data):
                tensor_dict['data_real'] = self.data.real.tolist()
                tensor_dict['data_imag'] = self.data.imag.tolist()
                tensor_dict['complex'] = True
            else:
                tensor_dict['data'] = self.data.tolist()
                tensor_dict['complex'] = False
        
        tensor_dict['dtype'] = str(self.dtype)
            
        return tensor_dict
    
    @classmethod
    def deserialize(cls, tensor_dict):
        """
        Create tensor from serialized dictionary
        
        Args:
            tensor_dict: dict - Dictionary from serialize() method
            
        Returns:
            RecursiveTensor: Reconstructed tensor object
        """
        dimensions = tensor_dict['dimensions']
        rank = tensor_dict['rank']
        distribution = tensor_dict['distribution']
        sparsity = tensor_dict['sparsity']
        if isinstance(dimensions, list):
            dimensions = tuple(dimensions)
        else:
            dimensions = (dimensions,)
        tensor = cls(
            dimensions=dimensions,
            rank=rank,
            distribution=distribution,
            sparsity=sparsity
        )
        tensor.dtype = np.dtype(tensor_dict['dtype'])
        tensor.data = tensor_dict['data']
        tensor.distribution = tensor_dict['distribution']
        tensor.sparsity = tensor_dict['sparsity']
        tensor.operation_history = tensor_dict['operation_history']
        tensor.metadata = tensor_dict.get('metadata', {
            "created": time.time(),
            "modified": time.time(),
            "operations_count": 0,
            "description": f"Deserialized RecursiveTensor({dimensions}, rank={rank})"
        })
        tensor.uuid = tensor_dict.get('uuid', str(uuid.uuid4()))

        is_complex = tensor_dict.get('complex', False)

        if tensor_dict['data_format'] == 'sparse':
            if is_complex:
                tensor.data = {
                    tuple(item[0]): complex(item[1], item[2]) 
                    for item in tensor_dict['data']
                }
            else:
                tensor.data = {
                    tuple(item[0]): item[1]
                    for item in tensor_dict['data']
                }
        else:
            if is_complex:
                real_part = np.array(tensor_dict['data_real'])
                imag_part = np.array(tensor_dict['data_imag'])
                tensor.data = real_part + 1j * imag_part
            else:
                tensor.data = np.array(tensor_dict['data'])
        
        return tensor
    
    def save(self, filepath, format='json', compress=True):
        """
        Save tensor to file
        
        Args:
            filepath: Path to save file
            format: 'json' or 'pickle' - serialization format
            compress: Whether to use gzip compression
            
        Returns:
            str: Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = self.serialize()
        
        data['save_timestamp'] = time.time()
        data['filepath'] = str(filepath)
        
        mode = 'wb' if format == 'pickle' or compress else 'w'
        
        if compress:
            if not filepath.endswith('.gz'):
                filepath += '.gz'
            
            if format == 'json':
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    f.write(json.dumps(data))
            else: 
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(data, f)
        else:
            with open(filepath, mode) as f:
                if format == 'json':
                    json.dump(data, f, indent=2)
                else:  # pickle
                    pickle.dump(data, f)
        
        logger.info(f"Tensor saved to {filepath} (format: {format}, compressed: {compress})")
        return filepath
    
    def save_rta(self, filepath: str):
        """
        Save tensor in RTA binary format according to specification.
        
        Args:
            filepath: Path to save RTA file
            
        Returns:
            str: Path to saved file
        """
        import struct
        import zlib
        
        with open(filepath, 'wb') as f:
            self._write_rta_format(f)
        
        logger.info(f"Tensor saved to {filepath} in RTA binary format")
        return filepath
    
    def save(self, file_handle):
        """
        Save tensor to file handle in RTA format (for LQT integration).
        
        Args:
            file_handle: Binary file handle to write to
        """
        import struct
        import zlib
        
        self._write_rta_format(file_handle)
    
    def _write_rta_format(self, f):
        """
        Write complete RTA binary format to file handle.
        
        Args:
            f: Binary file handle
        """
        import struct
        import zlib
        
        # C.2.1 Magic Header (8 bytes)
        f.write(b'RTATNSR\0')
        
        # C.2.2 Format Version (4 bytes)
        f.write(struct.pack('<HH', 1, 0))  # Major v1, Minor v0
        
        # C.2.3 Metadata Block
        self._write_metadata_block(f)
        
        # C.2.4 Dimension Block (24 bytes)
        self._write_dimension_block(f)
        
        # C.2.5 Sparsity Block
        self._write_sparsity_block(f)
        
        # C.2.6 Index Block
        self._write_index_block(f)
        
        # C.2.7 Value Block
        self._write_value_block(f)
        
        # C.2.8 Reference Block
        self._write_reference_block(f)
        
        # C.2.9 Pattern Block
        self._write_pattern_block(f)
        
        # C.2.10 Footer (16 bytes)
        self._write_footer(f)
    
    def _write_metadata_block(self, f):
        """Write metadata block in TLV format."""
        import struct
        
        metadata_data = b''
        entry_count = 0
        
        for key, value in self.metadata.items():
            # Key
            key_bytes = str(key).encode('utf-8')
            metadata_data += struct.pack('<H', len(key_bytes))
            metadata_data += key_bytes
            
            # Value type and data
            if isinstance(value, str):
                value_bytes = value.encode('utf-8')
                metadata_data += struct.pack('<BI', 0x00, len(value_bytes))
                metadata_data += value_bytes
            elif isinstance(value, int):
                metadata_data += struct.pack('<BIq', 0x01, 8, value)
            elif isinstance(value, float):
                metadata_data += struct.pack('<BId', 0x02, 8, value)
            elif isinstance(value, bool):
                metadata_data += struct.pack('<BIB', 0x03, 1, int(value))
            else:
                # Serialize complex objects as JSON
                json_bytes = json.dumps(value).encode('utf-8')
                metadata_data += struct.pack('<BI', 0x00, len(json_bytes))
                metadata_data += json_bytes
            
            entry_count += 1
        
        # Write block header and data
        block_size = len(metadata_data) + 8
        f.write(struct.pack('<II', block_size, entry_count))
        f.write(metadata_data)
        
        # Align to 8-byte boundary
        self._align_to_boundary(f, 8)
    
    def _write_dimension_block(self, f):
        """Write dimension block (24 bytes)."""
        # Delegate to spec-compliant implementation below to avoid divergence
        self._write_dimension_block_spec(f)

    def _write_dimension_block_spec(self, file_handle: BinaryIO) -> None:
        """Spec-compliant 24-byte dimension block (C.2.4)."""
        dist_map = {'normal': 0, 'uniform': 1, 'exponential': 2}
        distribution_type = dist_map.get(self.distribution, 3)
        flags = bytearray(7)
        # Bit 0: custom distribution
        if distribution_type == 3:
            flags[0] |= 0x01
        # Bit 1: pattern compression present
        if (getattr(self, '_patterns', None) and len(self._patterns) > 0) or getattr(self, '_has_patterns', False):
            flags[0] |= 0x02
        # Bit 2: quantized values (strictly when quantization metadata set)
        if self.metadata.get('quantized', False):
            flags[0] |= 0x04
        # Bit 3: self-references (reference block non-empty OR dedup references prepared)
        if (getattr(self, '_references', None) and len(self._references) > 0) or getattr(self, '_dedup_references', None):
            flags[0] |= 0x08
        # Bit 4: fractal parameters (fractal op in history)
        if any(op[0] == 'fractal' for op in getattr(self, 'operation_history', [])):
            flags[0] |= 0x10
        # Bit 5: double precision (float64/complex128)
        if self.dtype in (np.float64, np.complex128):
            flags[0] |= 0x20
        block = struct.pack('<I', self.dimensions)
        block += struct.pack('<B', self.rank)
        block += struct.pack('<I', distribution_type)
        block += struct.pack('<f', float(self.sparsity))
        block += struct.pack('<I', 0)  # reserved
        block += bytes(flags)
        assert len(block) == 24
        file_handle.write(block)
    
    def _write_sparsity_block(self, f):
        """Write sparsity block."""
        import struct
        
        # Use uniform sparsity for now
        encoding_method = 0x00
        density_map = b''  # No density map for uniform
        
        block_size = 4 + 1 + len(density_map)  # size field + encoding + data
        f.write(struct.pack('<I', block_size))
        f.write(struct.pack('<B', encoding_method))
        f.write(density_map)
        
        # Align to 8-byte boundary
        self._align_to_boundary(f, 8)
    
    def _write_index_block(self, f):
        """Write index block using COO format."""
        import struct
        
        encoding_method = 0x00  # COO format
        # Use deduplicated indices if available
        if isinstance(self.data, dict):
            if getattr(self, '_dedup_ready', False) and hasattr(self, '_dedup_indices'):
                indices = self._dedup_indices
            else:
                indices = list(self.data.keys())
            num_elements = len(indices)
            index_data = struct.pack('<B', encoding_method)
            for idx in indices:
                padded_idx = list(idx) + [0] * (5 - len(idx))
                index_data += struct.pack('<5H', *padded_idx[:5])
        else:
            # Dense tensor - find non-zero elements
            flat_data = self.data.flatten()
            non_zero_indices = np.where(np.abs(flat_data) > 1e-10)[0]
            num_elements = len(non_zero_indices)
            
            index_data = struct.pack('<B', encoding_method)
            for flat_idx in non_zero_indices:
                # Convert flat index to multidimensional
                multi_idx = np.unravel_index(flat_idx, self.data.shape)
                # Pad to rank=5
                padded_idx = list(multi_idx) + [0] * (5 - len(multi_idx))
                index_data += struct.pack('<5H', *padded_idx[:5])
        
        block_size = len(index_data)
        f.write(struct.pack('<QQ', block_size, num_elements))
        f.write(index_data)
        
        # Align to 8-byte boundary
        self._align_to_boundary(f, 8)
    
    def _write_value_block(self, f):
        """Write value block."""
        import struct
        
        encoding_method = 0x00  # float32
        # Prepare dedup if not already attempted
        if isinstance(self.data, dict) and not getattr(self, '_dedup_checked', False):
            try:
                self._prepare_deduplication()
            except Exception:
                self._dedup_ready = False
        if isinstance(self.data, dict):
            if getattr(self, '_dedup_ready', False) and hasattr(self, '_dedup_values'):
                source_values = self._dedup_values
            else:
                source_values = list(self.data.values())
            value_data = struct.pack('<B', encoding_method)
            for val in source_values:
                if np.iscomplexobj(val):
                    value_data += struct.pack('<ff', float(val.real), float(val.imag))
                else:
                    value_data += struct.pack('<f', float(val))
        else:
            # Dense tensor - get non-zero values
            flat_data = self.data.flatten()
            non_zero_indices = np.where(np.abs(flat_data) > 1e-10)[0]
            
            value_data = struct.pack('<B', encoding_method)
            for idx in non_zero_indices:
                val = flat_data[idx]
                if np.iscomplexobj(val):
                    value_data += struct.pack('<ff', float(val.real), float(val.imag))
                else:
                    value_data += struct.pack('<f', float(val))
        
        block_size = len(value_data)
        f.write(struct.pack('<Q', block_size))
        f.write(value_data)
        
        # Align to 8-byte boundary
        self._align_to_boundary(f, 8)
    
    def _write_footer(self, f):
        """Write footer with CRC-64 checksum."""
        import struct
        try:
            if f.seekable():
                pos = f.tell()
                f.seek(0)
                data = f.read(pos)
                crc64 = self._calculate_crc64(data)
                f.seek(pos)
            else:
                crc64 = 0
        except Exception:
            crc64 = 0
        f.write(struct.pack('<Q', crc64))
        f.write(b'RTAEND\0\0')
    
    def _align_to_boundary(self, f, boundary: int):
        """Align file position to boundary with zero padding."""
        pos = f.tell()
        aligned = (pos + boundary - 1) // boundary * boundary
        if aligned > pos:
            f.write(b'\x00' * (aligned - pos))
    
    @classmethod
    def load(cls, filepath):
        """
        Load tensor from file
        
        Args:
            filepath: Path to load file from
            
        Returns:
            RecursiveTensor: Loaded tensor
        """
        # Determine if file is compressed
        is_compressed = filepath.endswith('.gz')
        
        # Open with appropriate method
        try:
            if is_compressed:
                # Try JSON first (most common) - use text mode for JSON
                try:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        data = json.loads(f.read())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Fall back to pickle which requires binary mode
                    with gzip.open(filepath, 'rb') as f:
                        data = pickle.load(f)
            else:
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        # Fall back to pickle
                        f.close()
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                            
            # Handle version compatibility
            if 'version' in data and data['version'] != '2.1.0':
                logger.warning(f"Loading tensor with different version: {data['version']}")
                # Here you could add version-specific conversion logic
            
            # Deserialize
            tensor = cls.deserialize(data)
            logger.info(f"Tensor loaded from {filepath}")
            return tensor
            
        except Exception as e:
            logger.error(f"Error loading tensor from {filepath}: {e}")
            raise
    
    def save_chunked(self, filepath, chunk_size=1000):
        """
        Save very large tensor in chunks to avoid memory issues
        
        Args:
            filepath: Path to save file
            chunk_size: Number of elements per chunk
            
        Returns:
            str: Path to manifest file
        """
        # Create parent directories if they don't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Get serialized data
        data = self.serialize()
        
        # Handle chunking differently for sparse vs dense
        if data['data_format'] == 'sparse':
            # For sparse tensors, split the data array into chunks
            chunks = []
            total_elements = len(data['data'])
            chunk_count = (total_elements + chunk_size - 1) // chunk_size
            
            for i in range(chunk_count):
                chunk_data = data.copy()
                
                if data.get('complex', False):
                    # Complex data has [idx, real, imag] format
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, total_elements)
                    chunk_data['data'] = data['data'][start_idx:end_idx]
                else:
                    # Real data has [idx, val] format
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, total_elements)
                    chunk_data['data'] = data['data'][start_idx:end_idx]
                
                chunk_data['chunk_id'] = i
                chunk_data['total_chunks'] = chunk_count
                
                chunk_path = f"{filepath}.chunk{i}"
                with gzip.open(chunk_path, 'wb') as f:
                    f.write(json.dumps(chunk_data).encode('utf-8'))
                
                chunks.append(chunk_path)
            
            # Save manifest
            manifest = {
                'tensor_id': self.uuid,
                'dimensions': self.dimensions,
                'rank': self.rank,
                'chunks': chunks,
                'total_chunks': len(chunks),
                'timestamp': time.time(),
                'data_format': 'sparse',
                'complex': data.get('complex', False)
            }
            
            manifest_path = f"{filepath}.manifest"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
            return manifest_path
        else:
            # For dense tensors, it's more complicated - just use normal save with compression
            return self.save(filepath, compress=True)

    # RTA Binary Format Implementation
    @staticmethod
    def _crc64_table():
        """Generate CRC-64 lookup table using ECMA-182 polynomial"""
        poly = 0xC96C5795D7870F42
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
            table.append(crc)
        return table

    @staticmethod
    def _calculate_crc64(data: bytes) -> int:
        """Calculate CRC-64 checksum for data"""
        table = RecursiveTensor._crc64_table()
        crc = 0xFFFFFFFFFFFFFFFF
        
        for byte in data:
            crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
            
        return crc ^ 0xFFFFFFFFFFFFFFFF

    def _write_aligned(self, file_handle: BinaryIO, data: bytes) -> None:
        """Write data with 8-byte alignment padding"""
        file_handle.write(data)
        # Add padding to align to 8-byte boundary
        padding = (8 - (len(data) % 8)) % 8
        if padding > 0:
            file_handle.write(b'\x00' * padding)

    def _encode_metadata_value(self, value: Any) -> Tuple[int, bytes]:
        """Encode metadata value according to RTA format"""
        if isinstance(value, str):
            data = value.encode('utf-8')
            return 0x00, data
        elif isinstance(value, int):
            return 0x01, struct.pack('<q', value)
        elif isinstance(value, float):
            return 0x02, struct.pack('<d', value)
        elif isinstance(value, bool):
            return 0x03, struct.pack('<B', int(value))
        elif isinstance(value, bytes):
            return 0x04, value
        elif isinstance(value, (list, tuple)):
            # Array type - encode count followed by elements
            array_data = struct.pack('<I', len(value))
            for item in value:
                item_type, item_data = self._encode_metadata_value(item)
                array_data += struct.pack('<BI', item_type, len(item_data)) + item_data
            return 0x05, array_data
        else:
            # Custom type - serialize as JSON
            json_data = json.dumps(value).encode('utf-8')
            return 0xFF, json_data

    def _decode_metadata_value(self, value_type: int, data: bytes) -> Any:
        """Decode metadata value according to RTA format"""
        if value_type == 0x00:  # String
            return data.decode('utf-8')
        elif value_type == 0x01:  # Integer
            return struct.unpack('<q', data)[0]
        elif value_type == 0x02:  # Float
            return struct.unpack('<d', data)[0]
        elif value_type == 0x03:  # Boolean
            return bool(struct.unpack('<B', data)[0])
        elif value_type == 0x04:  # Binary
            return data
        elif value_type == 0x05:  # Array
            count = struct.unpack('<I', data[:4])[0]
            offset = 4
            result = []
            for _ in range(count):
                item_type = struct.unpack('<B', data[offset:offset+1])[0]
                item_length = struct.unpack('<I', data[offset+1:offset+5])[0]
                item_data = data[offset+5:offset+5+item_length]
                result.append(self._decode_metadata_value(item_type, item_data))
                offset += 5 + item_length
            return result
        elif value_type == 0xFF:  # Custom
            return json.loads(data.decode('utf-8'))
        else:
            raise ValueError(f"Unknown metadata value type: {value_type}")

    def _write_metadata_block(self, file_handle: BinaryIO) -> None:
        """Write metadata block in TLV format"""
        # Prepare metadata
        metadata = self.metadata.copy()
        metadata.update({
            'tensor_uuid': self.uuid,
            'creation_time': self.creation_time,
            'version': '2.1.0',
            'operation_count': len(self.operation_history),
            'last_modified': time.time()
        })
        
        # Calculate total size first
        metadata_entries = []
        total_size = 8  # size + count fields
        
        for key, value in metadata.items():
            key_bytes = key.encode('utf-8')
            value_type, value_data = self._encode_metadata_value(value)
            entry_size = 2 + len(key_bytes) + 1 + 4 + len(value_data)  # key_len + key + type + val_len + val
            total_size += entry_size
            metadata_entries.append((key_bytes, value_type, value_data))
        
        # Write metadata block
        file_handle.write(struct.pack('<II', total_size, len(metadata_entries)))
        
        for key_bytes, value_type, value_data in metadata_entries:
            file_handle.write(struct.pack('<H', len(key_bytes)))
            file_handle.write(key_bytes)
            file_handle.write(struct.pack('<BI', value_type, len(value_data)))
            file_handle.write(value_data)

    def _read_metadata_block(self, file_handle: BinaryIO) -> Dict[str, Any]:
        """Read metadata block in TLV format"""
        metadata_size, num_entries = struct.unpack('<II', file_handle.read(8))
        metadata = {}
        
        for _ in range(num_entries):
            key_length = struct.unpack('<H', file_handle.read(2))[0]
            key = file_handle.read(key_length).decode('utf-8')
            value_type = struct.unpack('<B', file_handle.read(1))[0]
            value_length = struct.unpack('<I', file_handle.read(4))[0]
            value_data = file_handle.read(value_length)
            
            metadata[key] = self._decode_metadata_value(value_type, value_data)
        
        return metadata

    def _write_dimension_block(self, file_handle: BinaryIO) -> None:
        """Write 24-byte dimension block (spec)."""
        dist_map = {'normal': 0, 'uniform': 1, 'exponential': 2}
        distribution_type = dist_map.get(self.distribution, 3)
        flags = bytearray(7)
        if distribution_type == 3:
            flags[0] |= 0x01  # custom distribution
        if getattr(self, '_patterns', None) and len(getattr(self, '_patterns')) > 0:
            flags[0] |= 0x02  # pattern compression
        if self.metadata.get('quantized', False):
            flags[0] |= 0x04  # quantized values
        if getattr(self, '_references', None) and len(getattr(self, '_references')) > 0:
            flags[0] |= 0x08  # self-references
        if any(op[0] == 'fractal' for op in getattr(self, 'operation_history', [])):
            flags[0] |= 0x10  # fractal parameters
        if self.dtype in (np.float64, np.complex128):
            flags[0] |= 0x20  # double precision
        block = struct.pack('<I', self.dimensions)
        block += struct.pack('<B', self.rank)
        block += struct.pack('<I', distribution_type)
        block += struct.pack('<f', float(self.sparsity))
        block += struct.pack('<I', 0)  # reserved
        block += bytes(flags)
        assert len(block) == 24
        file_handle.write(block)

    def _read_dimension_block(self, file_handle: BinaryIO) -> Dict[str, Any]:
        """Read 24-byte dimension block"""
        data = file_handle.read(24)
        
        dimensions = struct.unpack('<I', data[0:4])[0]
        rank = struct.unpack('<B', data[4:5])[0]
        # Skip 3 padding bytes (5:8)
        distribution_type = struct.unpack('<I', data[8:12])[0]
        sparsity = struct.unpack('<f', data[12:16])[0]
        # Reserved bytes 16-20 skipped
        flags = data[20:27]  # Last 7 bytes are flags
        
        dist_map = {0: 'normal', 1: 'uniform', 2: 'exponential', 3: 'power_law'}
        distribution = dist_map.get(distribution_type, 'normal')
        
        return {
            'dimensions': dimensions,
            'rank': rank,
            'distribution': distribution,
            'sparsity': sparsity,
            'flags': flags
        }

    def _write_sparsity_block(self, file_handle: BinaryIO) -> None:
        """Write sparsity block"""
        # For now, use uniform sparsity (encoding method 0x00)
        block_data = struct.pack('<IB', 5, 0x00)  # block_size=5, encoding=uniform
        file_handle.write(block_data)

    def _read_sparsity_block(self, file_handle: BinaryIO) -> Dict[str, Any]:
        """Read sparsity block"""
        block_size = struct.unpack('<I', file_handle.read(4))[0]
        encoding_method = struct.unpack('<B', file_handle.read(1))[0]
        
        # For uniform sparsity, no additional data
        remaining_size = block_size - 5
        if remaining_size > 0:
            file_handle.read(remaining_size)
        
        return {'encoding_method': encoding_method}

    def _write_index_block(self, file_handle: BinaryIO) -> None:
        """Write index block (COO)."""
        # If deduplication prepared, use unique indices; else fall back
        if isinstance(self.data, dict):
            if getattr(self, '_dedup_ready', False):
                indices = self._dedup_indices  # already tuples
            else:
                indices = list(self.data.keys())
            num_elems = len(indices)
            data_bytes = b''
            for idx in indices:
                pad = list(idx) + [0]*(5-len(idx))
                data_bytes += struct.pack('<5H', *pad[:5])
            block_size = 8 + 1 + len(data_bytes)  # count + encoding + tuples
            file_handle.write(struct.pack('<Q', block_size))
            file_handle.write(struct.pack('<Q', num_elems))
            file_handle.write(struct.pack('<B', 0x00))
            file_handle.write(data_bytes)
        else:
            block_size = 8 + 1
            file_handle.write(struct.pack('<Q', block_size))
            file_handle.write(struct.pack('<Q', 0))
            file_handle.write(struct.pack('<B', 0x00))
        self._align_to_boundary(file_handle, 8)

    def _read_index_block(self, file_handle: BinaryIO) -> Tuple[List[Tuple], int]:
        """Read index block in COO format"""
        header = file_handle.read(17)
        if len(header) < 17:
            return [], 0xFF
        block_size, num_elements, encoding = struct.unpack('<QQB', header)
        # block_size as written = 8 (num_elems) + 1 (encoding) + data_bytes length
        data_bytes_len = max(0, block_size - 9)
        data_bytes = file_handle.read(data_bytes_len)
        indices: List[Tuple] = []
        if encoding == 0x00 and num_elements > 0 and data_bytes_len >= 10:
            # Each index tuple = 10 bytes (5xuint16)
            expected_len = num_elements * 10
            if expected_len > data_bytes_len:
                expected_len = (data_bytes_len // 10) * 10
            for i in range(0, expected_len, 10):
                idx_data = struct.unpack('<5H', data_bytes[i:i+10])
                indices.append(tuple(idx_data))
        # Skip alignment padding
        pad = (8 - ((8 + block_size) % 8)) % 8
        if pad:
            file_handle.read(pad)
        return indices, encoding

    def _write_value_block(self, file_handle: BinaryIO) -> None:
        """Write value block (with quantization support)."""
        quantized = self.metadata.get('quantized', False)
        # Prepare deduplication for sparse dicts (only if not quantized to avoid scale duplication logic)
        if isinstance(self.data, dict) and not quantized and not getattr(self, '_dedup_checked', False):
            self._prepare_deduplication()
        if isinstance(self.data, dict):
            if getattr(self, '_dedup_ready', False):
                vals = self._dedup_values
            else:
                vals = list(self.data.values())
        else:
            if hasattr(self.data, 'toarray'):
                vals = self.data.toarray().flatten().tolist()
            else:
                vals = np.array(self.data).flatten().tolist()
        payload = b''
        if quantized and self.metadata.get('quant_target') in ('int8', 'int4'):
            target = self.metadata['quant_target']
            scale = float(self.metadata['quant_scale'])
            zp = int(self.metadata['quant_zero_point'])
            if target == 'int8':
                encoding = 0x02
                for v in vals:
                    payload += struct.pack('<b', int(v))
            else:
                encoding = 0x03
                tmp = [int(v) & 0xF for v in vals]
                if len(tmp) % 2 == 1:
                    tmp.append(0)
                for i in range(0, len(tmp), 2):
                    payload += struct.pack('<B', (tmp[i] << 4) | tmp[i+1])
            # prepend scale (float32) & zero point (int8) + padding
            params = struct.pack('<fb', scale, zp) + b'\x00\x00'
            payload = params + payload
        else:
            if any(np.iscomplexobj(v) for v in vals):
                encoding = 0x01
                for v in vals:
                    if np.iscomplexobj(v):
                        payload += struct.pack('<dd', float(v.real), float(v.imag))
                    else:
                        payload += struct.pack('<dd', float(v), 0.0)
            else:
                # Heuristic: attempt codebook (0x05) then mixed precision (0x04); fallback to float32 (0x00)
                float_vals = [float(v) for v in vals]
                # Codebook candidate
                unique_vals = list({fv for fv in float_vals})
                use_codebook = False
                codebook_bytes = b''
                index_bytes = b''
                if 0 < len(unique_vals) <= 256:
                    # Size if codebook used: 1B count + count*4B + len(vals)*1B
                    cb_size = 1 + len(unique_vals)*4 + len(float_vals)
                    # Float32 baseline size: len(vals)*4
                    if cb_size < len(float_vals)*4:
                        use_codebook = True
                        encoding = 0x05
                        # Stable ordering
                        unique_vals.sort()
                        val_to_idx = {v:i for i,v in enumerate(unique_vals)}
                        codebook_bytes = struct.pack('<B', len(unique_vals)) + b''.join(struct.pack('<f', v) for v in unique_vals)
                        index_bytes = bytes([val_to_idx[v] for v in float_vals])
                if use_codebook:
                    payload += codebook_bytes + index_bytes
                else:
                    # Mixed precision candidate: scale/offset + uint16 codes
                    vmin = min(float_vals) if float_vals else 0.0
                    vmax = max(float_vals) if float_vals else 0.0
                    use_mixed = False
                    mixed_bytes = b''
                    if float_vals and vmin != vmax:
                        # Compute scale to map range to 0..65535
                        rng = vmax - vmin
                        scale = rng / 65535.0
                        # Reconstruct after quantization to evaluate error
                        codes = [int(round((fv - vmin)/rng * 65535.0)) for fv in float_vals]
                        recon = [c * scale + vmin for c in codes]
                        # Relative error metric
                        err = 0.0
                        denom = sum(abs(fv) for fv in float_vals) + 1e-12
                        err = sum(abs(a-b) for a,b in zip(float_vals, recon)) / denom
                        # Only use if error < 0.5% and saves space (2 bytes vs 4)
                        if err < 0.005:
                            use_mixed = True
                            encoding = 0x04
                            mixed_bytes = struct.pack('<ff', float(vmin), float(scale)) + b''.join(struct.pack('<H', c) for c in codes)
                    if use_mixed:
                        payload += mixed_bytes
                    else:
                        encoding = 0x00
                        for v in float_vals:
                            payload += struct.pack('<f', v)
        block_size = 1 + len(payload)
        file_handle.write(struct.pack('<Q', block_size))
        file_handle.write(struct.pack('<B', encoding))
        file_handle.write(payload)
        self._align_to_boundary(file_handle, 8)

    def _read_value_block(self, file_handle: BinaryIO) -> Tuple[List, int]:
        """Read value block"""
        header = file_handle.read(9)
        if len(header) < 9:
            return [], 0xFF  # Corrupt
        block_size, encoding = struct.unpack('<QB', header)
        # block_size = 1 (encoding byte) + payload length
        # Sanity checks against unreasonable sizes ( > 64MB )
        if block_size > 64 * 1024 * 1024:
            logger.warning(f"Unreasonable value block size {block_size}, treating as corrupt")
            return [], encoding
        # Determine remaining file size to ensure we don't over-read
        try:
            cur_pos = file_handle.tell()
            # get total size via file descriptor
            total_size = os.fstat(file_handle.fileno()).st_size
            remaining = total_size - cur_pos
        except Exception:
            remaining = None
        payload_len = max(0, block_size - 1)
        if remaining is not None and payload_len > remaining:
            logger.warning(f"Value block payload {payload_len} exceeds remaining file bytes {remaining}; truncating")
            payload_len = max(0, remaining)
        payload = file_handle.read(payload_len)
        values: List[Any] = []
        if payload_len == 0:
            return values, encoding
        mv = memoryview(payload)
        if encoding == 0x00:  # float32
            if payload_len % 4 != 0:
                logger.warning('Value block float32 payload misaligned')
            count = payload_len // 4
            values = list(struct.unpack(f'<{count}f', mv[:count*4]))
        elif encoding == 0x01:  # complex (float64 pairs)
            if payload_len % 16 != 0:
                logger.warning('Value block complex payload misaligned')
            count = payload_len // 16
            for i in range(count):
                real, imag = struct.unpack('<dd', mv[i*16:(i+1)*16])
                values.append(complex(real, imag))
        elif encoding in (0x02, 0x03):  # quantized int8 / int4
            # Expect header: scale(float32) + zp(int8) + 2 pad bytes
            if payload_len < 7:
                logger.warning('Quantized value block too small for header')
                return values, encoding
            scale, zp = struct.unpack('<fb', mv[:5])
            data_stream = mv[7:]  # skip 2 pad bytes
            if encoding == 0x02:  # int8
                count = len(data_stream)
                int_vals = struct.unpack(f'<{count}b', data_stream)
                values = [float((v - zp) * scale) for v in int_vals]
            else:  # int4 packed
                byte_vals = struct.unpack(f'<{len(data_stream)}B', data_stream)
                deq = []
                for b in byte_vals:
                    v1 = (b >> 4) & 0xF
                    v2 = b & 0xF
                    if v1 >= 8: v1 -= 16
                    if v2 >= 8: v2 -= 16
                    deq.extend([v1, v2])
                values = [float((v - zp) * scale) for v in deq]
            if not hasattr(self, 'metadata'):
                self.metadata = {}
            self.metadata['quantized'] = True
            self.metadata['quant_scale'] = scale
            self.metadata['quant_zero_point'] = zp
        elif encoding == 0x04:  # mixed precision (vmin, scale, uint16 codes)
            if payload_len < 8:
                logger.warning('Mixed precision payload too small')
                return values, encoding
            vmin, scale = struct.unpack('<ff', mv[:8])
            codes_len = payload_len - 8
            if codes_len % 2 != 0:
                logger.warning('Mixed precision code section misaligned')
                codes_len -= codes_len % 2
            num_codes = codes_len // 2
            codes = struct.unpack(f'<{num_codes}H', mv[8:8+codes_len])
            values = [c * scale + vmin for c in codes]
        elif encoding == 0x05:  # codebook
            if payload_len < 1:
                logger.warning('Codebook payload too small')
                return values, encoding
            cb_count = mv[0]
            needed = 1 + cb_count*4
            if payload_len < needed:
                logger.warning('Incomplete codebook section')
                return values, encoding
            codebook = [struct.unpack('<f', mv[1+i*4:1+(i+1)*4])[0] for i in range(cb_count)]
            indices = mv[needed:]
            values = [codebook[i] for i in indices]
        else:
            # Unknown encoding; retain raw for potential future handling
            logger.warning(f'Unknown value encoding {encoding:#02x}; skipping payload')
        # Skip alignment padding after payload
        pad = (8 - ((8 + block_size) % 8)) % 8
        if pad:
            file_handle.read(pad)
        return values, encoding

    def _write_reference_block(self, file_handle: BinaryIO) -> None:
        """Write reference block.
        Encodes simple direct references (duplicate values) as type 0x00.
        Layout:
            Block Size (uint32)
            Number of References (uint32)
            Each Reference:
                Type (1B)
                Source Indices (5xuint16)
                Target Indices (5xuint16)
                Param Length (uint16)
                Param Data (bytes)
        """
        # If deduplication prepared we already have reference list
        if not isinstance(self.data, dict) or len(self.data) < 2:
            file_handle.write(struct.pack('<II', 8, 0))
            return
        if getattr(self, '_dedup_ready', False):
            refs = self._dedup_references
        else:
            # fallback to naive duplicates (non-compressing)
            value_map = {}
            for idx, val in self.data.items():
                value_map.setdefault(float(val), []).append(idx)
            refs = []
            for val, idx_list in value_map.items():
                if len(idx_list) > 1:
                    src = idx_list[0]
                    for tgt in idx_list[1:]:
                        refs.append((0x00, src, tgt, b''))
                if len(refs) >= 256:
                    break
        if not refs:
            file_handle.write(struct.pack('<II', 8, 0))
            return
        entries_bytes = b''
        for rtype, src, tgt, params in refs:
            src_pad = list(src) + [0]*(5-len(src))
            tgt_pad = list(tgt) + [0]*(5-len(tgt))
            entries_bytes += struct.pack('<B5H5HH', rtype, *src_pad[:5], *tgt_pad[:5], len(params))
            if params:
                entries_bytes += params
        block_size = 8 + len(entries_bytes)
        file_handle.write(struct.pack('<II', block_size, len(refs)))
        file_handle.write(entries_bytes)
        # 8-byte alignment
        if hasattr(self, '_align_to_boundary'):
            self._align_to_boundary(file_handle, 8)

    def _read_reference_block(self, file_handle: BinaryIO) -> Dict[str, Any]:
        """Read reference block into internal reference list."""
        block_size, num_references = struct.unpack('<II', file_handle.read(8))
        refs = []
        consumed = 0
        while consumed < (block_size - 8) and len(refs) < num_references:
            header = file_handle.read(1 + 10 + 10 + 2)
            if len(header) < 23:
                break
            rtype = header[0]
            src = struct.unpack('<5H', header[1:11])
            tgt = struct.unpack('<5H', header[11:21])
            param_len = struct.unpack('<H', header[21:23])[0]
            params = file_handle.read(param_len) if param_len > 0 else b''
            consumed += 23 + param_len
            refs.append({'type': rtype, 'source': src, 'target': tgt, 'parameters': params})
        # Skip alignment padding
        pad = (8 - (block_size % 8)) % 8
        if pad:
            file_handle.read(pad)
        return {'num_references': num_references, 'references': refs}

    def _write_pattern_block(self, file_handle: BinaryIO) -> None:
        """Write pattern block using simple pattern detector.
        Pattern encoding:
            Block Size (uint32)
            Number of Patterns (uint32)
            For each pattern:
                Size (uint32)  -> number of values in pattern
                Occurrences (uint32)
                Data Length (uint32)
                Data Bytes
                Locations Length (uint32)
                Locations Bytes
        """
        if not isinstance(self.data, dict) or len(self.data) < 8:
            file_handle.write(struct.pack('<II', 8, 0))
            return
        patterns = self._detect_patterns()
        if not patterns:
            file_handle.write(struct.pack('<II', 8, 0))
            return
        body = b''
        count = 0
        for p in patterns:
            data_bytes = p['data']
            loc_bytes = p['locations']
            body += struct.pack('<II', p['length'], p['occurrences'])
            body += struct.pack('<I', len(data_bytes)) + data_bytes
            body += struct.pack('<I', len(loc_bytes)) + loc_bytes
            count += 1
        block_size = 8 + len(body)
        file_handle.write(struct.pack('<II', block_size, count))
        file_handle.write(body)
        # 8-byte alignment
        if hasattr(self, '_align_to_boundary'):
            self._align_to_boundary(file_handle, 8)

    def _read_pattern_block(self, file_handle: BinaryIO) -> Dict[str, Any]:
        """Read pattern block (stores raw patterns in metadata)."""
        block_size, num_patterns = struct.unpack('<II', file_handle.read(8))
        patterns = []
        consumed = 0
        while consumed < (block_size - 8) and len(patterns) < num_patterns:
            header = file_handle.read(8)
            if len(header) < 8:
                break
            size, occ = struct.unpack('<II', header)
            dl = struct.unpack('<I', file_handle.read(4))[0]
            data_bytes = file_handle.read(dl)
            ll = struct.unpack('<I', file_handle.read(4))[0]
            loc_bytes = file_handle.read(ll)
            consumed += 8 + 4 + dl + 4 + ll
            patterns.append({'size': size, 'occurrences': occ, 'data': data_bytes, 'locations': loc_bytes})
        # Skip alignment padding
        pad = (8 - (block_size % 8)) % 8
        if pad:
            file_handle.read(pad)
        return {'num_patterns': num_patterns, 'patterns': patterns}

    def save_rta(self, filepath: str) -> str:
        """
        Save tensor to RTA binary format file
        
        Args:
            filepath: Path to save the RTA file
            
        Returns:
            str: Path to saved file
            
        Raises:
            IOError: If file cannot be written
            ValueError: If tensor data is invalid
        """
        # Validate tensor data before saving
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Tensor data is None or missing")
        
        if not hasattr(self, 'dimensions') or self.dimensions <= 0:
            raise ValueError(f"Invalid dimensions: {getattr(self, 'dimensions', None)}")
        
        if not hasattr(self, 'rank') or self.rank <= 0:
            raise ValueError(f"Invalid rank: {getattr(self, 'rank', None)}")
        
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        
        if not hasattr(self, 'operation_history'):
            self.operation_history = []
        
        try:
            # Ensure parent directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Use a temporary file to ensure atomic writes
            temp_filepath = filepath + '.tmp'
            
            with open(temp_filepath, 'wb') as f:
                # Write magic header
                f.write(b'RTATNSR\0')
                
                # Write format version
                f.write(struct.pack('<HH', 1, 0))  # Major=1, Minor=0
                
                # Write metadata block
                self._write_metadata_block(f)
                
                # Write dimension block (24 bytes spec)
                self._write_dimension_block(f)
                
                # Write sparsity block  
                self._write_sparsity_block(f)
                
                # Prepare deduplication early so index & value blocks are consistent
                if isinstance(self.data, dict) and not getattr(self, '_dedup_checked', False):
                    try:
                        self._prepare_deduplication()
                    except Exception:
                        self._dedup_ready = False
                
                # Write index & value blocks
                self._write_index_block(f)
                self._write_value_block(f)
                
                # Write reference block
                self._write_reference_block(f)
                
                # Write pattern block
                self._write_pattern_block(f)
                
                # Calculate CRC64 by reading the temp file
                f.flush()
                current_pos = f.tell()
                
                # Read file content for CRC calculation
                with open(temp_filepath, 'rb') as crc_file:
                    file_content = crc_file.read()
                crc64 = self._calculate_crc64(file_content)
                
                # Write footer
                f.write(struct.pack('<Q', crc64))
                f.write(b'RTAEND\0\0')
            
            # Atomically move temp file to final location
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
            
            logger.info(f"Tensor saved to RTA format: {filepath}")
            return filepath
            
        except Exception as e:
            # Cleanup temp file if it exists
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            logger.error(f"Error saving tensor to RTA format: {e}")
            raise IOError(f"Failed to save RTA file: {e}") from e

    # ---------------- Quantization & Convenience .rt API -----------------
    def quantize(self, mode: str = 'auto', target: str = 'int8') -> 'RecursiveTensor':
        """Quantize values (store integer codes + scale/zero-point metadata).

        Args:
            mode: 'auto' | 'symmetric' | 'asymmetric'
            target: 'int8' | 'int4'
        Returns: self
        """
        if target not in ('int8', 'int4'):
            raise ValueError("target must be 'int8' or 'int4'")
        if isinstance(self.data, dict):
            vals = np.array(list(self.data.values()), dtype=np.float32)
        else:
            if hasattr(self.data, 'toarray'):
                vals = self.data.toarray().astype(np.float32)
            else:
                vals = np.array(self.data, dtype=np.float32)
        if vals.size == 0:
            return self
        vmin = float(vals.min())
        vmax = float(vals.max())
        if mode == 'auto':
            mode_use = 'symmetric' if vmin < 0 and abs(vmin)/(abs(vmax)+1e-12) > 0.9 else 'asymmetric'
        else:
            mode_use = mode
        if target == 'int8':
            qmin, qmax = -128, 127
        else:
            qmin, qmax = -8, 7
        if mode_use == 'symmetric':
            amax = max(abs(vmin), abs(vmax))
            scale = 1.0 if amax == 0 else amax / qmax
            zp = 0
        else:
            scale = (vmax - vmin)/(qmax - qmin) if vmax != vmin else 1.0
            zp = int(round(qmin - vmin/scale))
            zp = max(qmin, min(qmax, zp))
        def qf(x):
            q = int(round(x/scale + zp))
            return max(qmin, min(qmax, q))
        if isinstance(self.data, dict):
            for k,val in list(self.data.items()):
                self.data[k] = qf(float(val))
        else:
            arr = vals
            qarr = np.vectorize(qf)(arr).astype(np.int8)
            self.data = qarr
            self.dtype = np.int8
        self.metadata['quantized'] = True
        self.metadata['quant_mode'] = mode_use
        self.metadata['quant_target'] = target
        self.metadata['quant_scale'] = scale
        self.metadata['quant_zero_point'] = zp
        return self

    def save_rt(self, filepath: str, quantize: bool | str = False, target: str = 'int8') -> str:
        """Save tensor as .rt file (RTA spec). Optionally quantize.

        Args:
            filepath: destination path (".rt" appended if missing)
            quantize: False | True | 'auto' | 'symmetric' | 'asymmetric'
            target: quantization target ('int8'|'int4')
        """
        if quantize:
            mode = 'auto' if quantize is True else str(quantize)
            self.quantize(mode=mode, target=target)
        if not filepath.endswith('.rt'):
            filepath += '.rt'
        return self.save_rta(filepath)

    def save(self, file_handle: BinaryIO) -> None:
        """
        Save tensor to RTA binary format using file handle for LQT integration
        
        Args:
            file_handle: Open binary file handle to write to
            
        Raises:
            IOError: If file cannot be written
            ValueError: If tensor data is invalid
        """
        # Validate tensor data before saving
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Tensor data is None or missing")
        
        if not hasattr(self, 'dimensions') or self.dimensions <= 0:
            raise ValueError(f"Invalid dimensions: {getattr(self, 'dimensions', None)}")
        
        if not hasattr(self, 'rank') or self.rank <= 0:
            raise ValueError(f"Invalid rank: {getattr(self, 'rank', None)}")
        
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        
        if not hasattr(self, 'operation_history'):
            self.operation_history = []
        
        if not file_handle or not hasattr(file_handle, 'write'):
            raise ValueError("Invalid file handle provided")
        
        try:
            # Write magic header
            file_handle.write(b'RTATNSR\0')
            
            # Write format version
            file_handle.write(struct.pack('<HH', 1, 0))  # Major=1, Minor=0
            
            # Write metadata block
            self._write_metadata_block(file_handle)
            
            # Write dimension block (24 bytes spec)
            self._write_dimension_block(file_handle)
            
            # Write sparsity block  
            self._write_sparsity_block(file_handle)
            
            # Prepare deduplication before writing index/value
            if isinstance(self.data, dict) and not getattr(self, '_dedup_checked', False):
                try:
                    self._prepare_deduplication()
                except Exception:
                    self._dedup_ready = False
            
            # Write index & value blocks
            self._write_index_block(file_handle)
            self._write_value_block(file_handle)
            
            # Write reference block
            self._write_reference_block(file_handle)
            
            # Write pattern block
            self._write_pattern_block(file_handle)
            
            # CRC64 if seekable
            try:
                if file_handle.seekable():
                    pos = file_handle.tell()
                    file_handle.seek(0)
                    data = file_handle.read(pos)
                    crc64 = self._calculate_crc64(data)
                    file_handle.seek(pos)
                else:
                    crc64 = 0
            except Exception:
                crc64 = 0
            file_handle.write(struct.pack('<Q', crc64))
            file_handle.write(b'RTAEND\0\0')
            
            file_handle.flush()
            logger.info("Tensor saved to RTA format via file handle")
            
        except Exception as e:
            logger.error(f"Error saving tensor to RTA format: {e}")
            raise IOError(f"Failed to save RTA to file handle: {e}") from e

    @classmethod
    def load_rta(cls, filepath: str) -> 'RecursiveTensor':
        """
        Load tensor from RTA binary format file
        
        Args:
            filepath: Path to RTA file to load
            
        Returns:
            RecursiveTensor: Loaded tensor object
            
        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid or corrupted
        """
        if not filepath or not os.path.exists(filepath):
            raise IOError(f"File does not exist: {filepath}")
        
        if not os.path.isfile(filepath):
            raise IOError(f"Path is not a file: {filepath}")
        
        if os.path.getsize(filepath) < 32:  # Minimum size for a valid RTA file
            raise ValueError(f"File too small to be valid RTA format: {os.path.getsize(filepath)} bytes")
        
        try:
            with open(filepath, 'rb') as f:
                # Verify magic header
                magic = f.read(8)
                if magic != b'RTATNSR\0':
                    raise ValueError(f"Invalid RTA magic header: {magic}")
                
                # Read format version
                major, minor = struct.unpack('<HH', f.read(4))
                if major != 1:
                    raise ValueError(f"Unsupported RTA format version: {major}.{minor}")
                
                # Create temporary instance for reading blocks
                temp_instance = cls.__new__(cls)
                
                # Read metadata block
                metadata = temp_instance._read_metadata_block(f)
                
                # Read dimension block
                dim_info = temp_instance._read_dimension_block(f)
                # Record quantization flag for value parsing
                if dim_info.get('flags') and (dim_info['flags'][0] & 0x04):
                    temp_instance._quantized_flag = True
                else:
                    temp_instance._quantized_flag = False
                
                # Read sparsity block
                sparsity_info = temp_instance._read_sparsity_block(f)
                
                # Read index block
                indices, index_encoding = temp_instance._read_index_block(f)
                
                # Read value block
                values, value_encoding = temp_instance._read_value_block(f)
                
                # Read reference block
                ref_info = temp_instance._read_reference_block(f)
                
                # Read pattern block
                pattern_info = temp_instance._read_pattern_block(f)
                
                # Read footer and verify CRC64
                stored_crc = struct.unpack('<Q', f.read(8))[0]
                footer_magic = f.read(8)
                if footer_magic != b'RTAEND\0\0':
                    raise ValueError(f"Invalid RTA footer magic: {footer_magic}")
                
                # Verify CRC64 (calculate CRC of everything except the footer)
                current_pos = f.tell()
                f.seek(0)
                content_to_check = f.read(current_pos - 16)  # Exclude footer
                calculated_crc = cls._calculate_crc64(content_to_check)
                
                if stored_crc != calculated_crc:
                    logger.warning(f"CRC64 mismatch: stored={stored_crc:016x}, calculated={calculated_crc:016x}")
                    # Continue loading but warn about potential corruption
                
                # Validate dimension info
                if dim_info['dimensions'] <= 0:
                    raise ValueError(f"Invalid dimensions in file: {dim_info['dimensions']}")
                
                if dim_info['rank'] <= 0 or dim_info['rank'] > 10:
                    raise ValueError(f"Invalid rank in file: {dim_info['rank']}")
                
                if not (0.0 <= dim_info['sparsity'] <= 1.0):
                    raise ValueError(f"Invalid sparsity in file: {dim_info['sparsity']}")
                
                # Create tensor instance
                tensor = cls(
                    dimensions=dim_info['dimensions'],
                    rank=dim_info['rank'],
                    dtype=np.float32,  # Default, will be updated based on data
                    distribution=dim_info['distribution'],
                    sparsity=dim_info['sparsity']
                )
                
                # Restore metadata
                tensor.metadata = metadata
                if 'tensor_uuid' in metadata:
                    tensor.uuid = metadata['tensor_uuid']
                if 'creation_time' in metadata:
                    tensor.creation_time = metadata['creation_time']
                
                # Reconstruct tensor data
                if indices and values:
                    # Sparse tensor
                    if len(indices) != len(values):
                        raise ValueError(f"Index/value count mismatch: {len(indices)} indices, {len(values)} values")
                    
                    tensor.data = {}
                    for idx, val in zip(indices, values):
                        # Truncate indices to actual rank
                        actual_idx = idx[:tensor.rank]
                        tensor.data[actual_idx] = val
                    
                    # Set dtype based on values
                    if values and isinstance(values[0], complex):
                        tensor.dtype = np.complex64
                elif values and not indices:
                    # Dense tensor - reshape values to tensor dimensions
                    expected_shape = tuple([tensor.dimensions] * tensor.rank)
                    expected_size = tensor.dimensions ** tensor.rank
                    
                    if len(values) != expected_size:
                        logger.warning(f"Value count mismatch: got {len(values)}, expected {expected_size}")
                        # Pad or truncate values to match expected size
                        if len(values) < expected_size:
                            values.extend([0] * (expected_size - len(values)))
                        else:
                            values = values[:expected_size]
                    
                    if isinstance(values[0], complex):
                        tensor.dtype = np.complex64
                        tensor.data = np.array(values, dtype=tensor.dtype).reshape(expected_shape)
                    else:
                        tensor.dtype = np.float32
                        tensor.data = np.array(values, dtype=tensor.dtype).reshape(expected_shape)
                else:
                    # Empty tensor
                    tensor.data = {}
                
                # Apply reconstructed references if any
                if 'references' in ref_info and ref_info['references']:
                    tensor._references = ref_info['references']
                    tensor._apply_references()
                    # Infer deduplication if we have direct reference types
                    if not tensor.metadata.get('deduplicated'):
                        if any(r.get('type') == 0x00 for r in ref_info['references']):
                            tensor.metadata['deduplicated'] = True
                
                # Apply reconstructed patterns if any
                if 'patterns' in pattern_info and pattern_info['patterns']:
                    tensor._patterns = pattern_info['patterns']
                    tensor._apply_patterns()
                
                logger.info(f"Successfully loaded RTA tensor from {filepath}")
                return tensor
                
        except Exception as e:
            logger.error(f"Error loading RTA tensor from {filepath}: {e}")
            raise IOError(f"Failed to load RTA file: {e}") from e

    
    @classmethod
    def load_chunked(cls, manifest_path):
        """
        Load tensor from chunked files
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            RecursiveTensor: Loaded tensor
        """
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Create tensor instance
        tensor = cls(
            dimensions=manifest['dimensions'],
            rank=manifest['rank']
        )
        
        # Load chunks
        if manifest['data_format'] == 'sparse':
            is_complex = manifest.get('complex', False)
            
            # Initialize empty data dictionary
            tensor.data = {}
            
            # Load each chunk
            for chunk_path in manifest['chunks']:
                with gzip.open(chunk_path, 'rb') as f:
                    chunk_data = json.loads(f.read().decode('utf-8'))
                
                # Add data from this chunk
                if is_complex:
                    # Complex data has [idx, real, imag] format
                    for item in chunk_data['data']:
                        tensor.data[tuple(item[0])] = complex(item[1], item[2])
                else:
                    # Real data has [idx, val] format
                    for item in chunk_data['data']:
                        tensor.data[tuple(item[0])] = item[1]
        
        # Set UUID from manifest
        tensor.uuid = manifest['tensor_id']
        
        return tensor
    
    def __add__(self, other):
        """
        Add two tensors (element-wise addition)
        
        Args:
            other: RecursiveTensor or scalar
            
        Returns:
            RecursiveTensor: Result of addition
        """
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        
        if isinstance(other, RecursiveTensor):
            # Tensor + Tensor
            if self.rank != other.rank or self.dimensions != other.dimensions:
                raise ValueError("Tensors must have the same rank and dimensions for addition")
                
            if isinstance(self.data, dict) and isinstance(other.data, dict):
                # Both sparse
                result.data = self.data.copy()
                for idx, val in other.data.items():
                    result.data[idx] = result.data.get(idx, 0) + val
            elif isinstance(self.data, dict):
                # self is sparse, other is dense
                result.data = other.data.copy()
                for idx, val in self.data.items():
                    result.data[idx] += val
            elif isinstance(other.data, dict):
                # self is dense, other is sparse
                result.data = self.data.copy()
                for idx, val in other.data.items():
                    result.data[idx] += val
            else:
                # Both dense
                result.data = self.data + other.data
        else:
            # Tensor + scalar
            if isinstance(self.data, dict):
                # Sparse case
                result.data = {idx: val + other for idx, val in self.data.items()}
            else:
                # Dense case
                result.data = self.data + other
        
        # Update metadata
        result.operation_history = self.operation_history.copy()
        result.operation_history.append(('add', id(other) if isinstance(other, RecursiveTensor) else 'scalar'))
        result.metadata = self.metadata.copy()
        result.metadata["operations_count"] += 1
        result.metadata["modified"] = time.time()
        
        if isinstance(other, RecursiveTensor):
            result.metadata["description"] = f"Addition of {self.metadata['description']} and {other.metadata['description']}"
        else:
            result.metadata["description"] = f"Addition of {self.metadata['description']} and scalar {other}"
        
        return result
    
    def __mul__(self, other):
        """
        Multiply tensor by scalar or element-wise multiply with another tensor
        
        Args:
            other: scalar or RecursiveTensor
            
        Returns:
            RecursiveTensor: Result of multiplication
        """
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        
        if isinstance(other, RecursiveTensor):
            if self.rank != other.rank or self.dimensions != other.dimensions:
                raise ValueError("Tensors must have the same rank and dimensions for element-wise multiplication")
                
            if isinstance(self.data, dict) and isinstance(other.data, dict):
                result.data = {}
                for idx in set(self.data.keys()).intersection(other.data.keys()):
                    result.data[idx] = self.data[idx] * other.data[idx]
            elif isinstance(self.data, dict):
                result.data = {}
                for idx, val in self.data.items():
                    if isinstance(other.data, np.ndarray) and all(i < other.data.shape[d] for d, i in enumerate(idx) if d < len(other.data.shape)):
                        result.data[idx] = val * other.data[idx]
            elif isinstance(other.data, dict):
                result.data = {}
                for idx, val in other.data.items():
                    if isinstance(self.data, np.ndarray) and all(i < self.data.shape[d] for d, i in enumerate(idx) if d < len(self.data.shape)):
                        result.data[idx] = self.data[idx] * val
            else:
                result.data = self.data * other.data
        else:
            if isinstance(self.data, dict):
                # Sparse case
                result.data = {idx: val * other for idx, val in self.data.items()}
            else:
                result.data = self.data * other
        
        result.operation_history = self.operation_history.copy()
        result.operation_history.append(('multiply', id(other) if isinstance(other, RecursiveTensor) else 'scalar'))
        result.metadata = self.metadata.copy()
        result.metadata["operations_count"] += 1
        result.metadata["modified"] = time.time()
        
        if isinstance(other, RecursiveTensor):
            result.metadata["description"] = f"Element-wise multiplication of {self.metadata['description']} and {other.metadata['description']}"
        else:
            result.metadata["description"] = f"Scalar multiplication of {self.metadata['description']} by {other}"
        
        return result

    def __getitem__(self, indices):
        """
        Get tensor element or slice
        
        Args:
            indices: tuple - indices to access
            
        Returns:
            Value or sub-tensor
        """
        if isinstance(self.data, dict):
            if isinstance(indices, tuple):
                return self.data.get(indices, 0)
            else:
                return self.data.get((indices,), 0)
        else:
            return self.data[indices]
    
    def __setitem__(self, indices, value):
        """
        Set tensor element or slice
        
        Args:
            indices: tuple - indices to set
            value: new value
        """
        if isinstance(self.data, dict):
            if value == 0:
                if indices in self.data:
                    del self.data[indices]
            else:
                self.data[indices] = value
        else:
            self.data[indices] = value
            
        self.metadata["modified"] = time.time()
    
    def __str__(self):
        """String representation of tensor"""
        if isinstance(self.data, dict):
            nnz = len(self.data)
            density = nnz / (self.dimensions ** self.rank) if self.dimensions > 0 else 0
            return f"RecursiveTensor(dimensions={self.dimensions}, rank={self.rank}, format=sparse, nnz={nnz}, density={density:.2e})"
        else:
            return f"RecursiveTensor(dimensions={self.dimensions}, rank={self.rank}, shape={self.data.shape}, format=dense)"
    
    def __repr__(self):
        """Detailed representation of tensor"""
        return str(self)
        
    def tucker_decomposition(self, ranks=None):
        """
        Perform Tucker decomposition of the tensor.
        
        Args:
            ranks: list - target rank for each mode
            
        Returns:
            tuple: (core_tensor, factor_matrices)
        """
        if ranks is None:
            ranks = [min(self.dimensions, 5)] * self.rank
            
        if isinstance(self.data, dict):
            dense_data = self._sparse_to_dense(self.data)
        else:
            dense_data = self.data
        if len(ranks) != self.rank:
            raise ValueError("Ranks must match the number of modes in the tensor")
        factors = []
        
        for mode in range(self.rank):
            unfolded = np.moveaxis(dense_data, mode, 0)
            unfolded = unfolded.reshape(dense_data.shape[mode], -1)
            
            U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
            
            factors.append(U[:, :ranks[mode]])
        
        core = dense_data.copy()
        for mode, factor in enumerate(factors):
            core = np.tensordot(core, factor, axes=([0], [0]))
        return core, factors

    
    def to_mps(self, max_bond_dimension=None):
        """
        Convert tensor to Matrix Product State representation.
        
        Args:
            max_bond_dimension: int - maximum bond dimension
            
        Returns:
            list: MPS tensors
        """
        if isinstance(self.data, dict):
            dense_data = self._sparse_to_dense(self.data)
        else:
            dense_data = self.data
            
        mps_tensors = []
        
        current = dense_data
        
        for i in range(self.rank - 1):
            shape = current.shape
            current = current.reshape(shape[0], -1)
            
            U, S, V = np.linalg.svd(current, full_matrices=False)
            
            # Truncate if needed
            if max_bond_dimension and len(S) > max_bond_dimension:
                U = U[:, :max_bond_dimension]
                S = S[:max_bond_dimension]
                V = V[:max_bond_dimension, :]
            
            # Create MPS tensor
            mps_tensor = U.reshape(shape[0], -1)
            mps_tensors.append(mps_tensor)
            
            # Update current tensor
            current = np.diag(S) @ V
            current = current.reshape(-1, *shape[1:])
        
        # Add last tensor
        mps_tensors.append(current)
        
        return mps_tensors

    def enable_gradient_tracking(self):
        """
        Enable tracking of gradients for tensor operations.
        Useful for integration with neural networks.
        
        Returns:
            RecursiveTensor: Tensor with gradient tracking
        """
        try:
            import torch
            
            # Create gradient-enabled tensor
            result = RecursiveTensor(self.dimensions, 
                                    rank=self.rank, 
                                    dtype=self.dtype,
                                    distribution=self.distribution,
                                    sparsity=self.sparsity)
            
            if isinstance(self.data, dict):
                # Convert sparse tensor to PyTorch tensor with gradients
                indices = np.array(list(self.data.keys())).T
                values = np.array(list(self.data.values()))
                
                if indices.size > 0:
                    torch_indices = torch.LongTensor(indices)
                    torch_values = torch.FloatTensor(values)
                    
                    # Create sparse torch tensor
                    shape = tuple([self.dimensions] * self.rank)
                    torch_tensor = torch.sparse.FloatTensor(torch_indices, torch_values, shape)
                    torch_tensor = torch_tensor.requires_grad_()
                    
                    result.data = torch_tensor
                else:
                    result.data = {}
            else:
                # Convert dense tensor to PyTorch tensor with gradients
                torch_tensor = torch.tensor(self.data, dtype=torch.float32, requires_grad=True)
                result.data = torch_tensor
            
            # Mark that this tensor has gradient tracking
            result.metadata["has_gradients"] = True
            result.metadata["description"] = f"Gradient-enabled {self.metadata['description']}"
            
            return result
        except ImportError:
            logger.warning("PyTorch not available - gradient tracking requires PyTorch")
            return self

    def persistent_homology(self, max_dimension=1):
        """
        Compute persistent homology of the tensor.
        
        Args:
            max_dimension: int - maximum homology dimension
            
        Returns:
            dict: Persistence diagrams
        """
        try:
            import gudhi as gd
            
            if isinstance(self.data, dict):
                # Use sparse points as filtration
                points = np.array(list(self.data.keys()))
                values = np.array(list(self.data.values()))
            else:
                # Extract points from dense tensor
                points = np.array(np.where(np.abs(self.data) > 1e-6)).T
                values = self.data[tuple(points.T)]
            
            # Create a simplex tree
            st = gd.SimplexTree()
            
            # Add vertices
            for i, point in enumerate(points):
                st.insert([i], filtration=abs(values[i]))
            
            # Add edges based on proximity
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    # Use Euclidean distance for filtration
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist < 2:  # Only connect nearby points
                        st.insert([i, j], filtration=dist)
            
            # Compute persistence
            st.persistence(min_persistence=0.01)
            
            # Get persistence diagrams
            diagrams = {}
            for dim in range(max_dimension + 1):
                pairs = st.persistence_pairs_in_dimension(dim)
                diagrams[dim] = np.array(pairs)
            
            return diagrams
        except ImportError:
            logger.warning("GUDHI not available - persistent homology requires GUDHI package")
            return None

    def to_gpu(self):
        """
        Move tensor to GPU for accelerated computation.
        Requires CuPy or PyTorch.
        
        Returns:
            RecursiveTensor: GPU-accelerated tensor
        """
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        
        try:
            import cupy as cp
            
            if isinstance(self.data, dict):
                # For sparse tensors, we need to handle differently
                # Create COO format for cupy
                indices = list(self.data.keys())
                values = list(self.data.values())
                
                # Create dense tensor on GPU for now (sparse tensor support is limited)
                gpu_tensor = cp.zeros((self.dimensions,) * self.rank, dtype=self.dtype)
                for idx, val in zip(indices, values):
                    gpu_tensor[idx] = val
                    
                result.data = gpu_tensor
            else:
                # Move dense tensor to GPU
                result.data = cp.array(self.data)
                
            result.metadata["on_gpu"] = True
            result.metadata["description"] = f"GPU-accelerated {self.metadata['description']}"
            return result
        
        except ImportError:
            try:
                import torch
                
                if isinstance(self.data, dict):
                    # Create COO tensor for PyTorch
                    indices = np.array(list(self.data.keys())).T
                    values = np.array(list(self.data.values()))
                    
                    if indices.size > 0:
                        torch_indices = torch.LongTensor(indices).cuda()
                        torch_values = torch.FloatTensor(values).cuda()
                        
                        shape = tuple([self.dimensions] * self.rank)
                        result.data = torch.sparse.FloatTensor(torch_indices, torch_values, shape)
                    else:
                        result.data = {}
                else:
                    # Move dense tensor to GPU
                    result.data = torch.tensor(self.data).cuda()
                    
                result.metadata["on_gpu"] = True
                result.metadata["description"] = f"GPU-accelerated {self.metadata['description']}"
                return result
                
            except ImportError:
                logger.warning("Neither CuPy nor PyTorch available - GPU acceleration not possible")
                return self

    def visualize_tensor_network(self, show=True):
        """
        Visualize tensor as a network diagram.
        
        Args:
            show: bool - whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: Network visualization
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes for each dimension
            for i in range(self.rank):
                G.add_node(f"dim_{i}", type="dimension", size=self.dimensions)
            
            # Add central tensor node
            G.add_node("tensor", type="tensor", size=len(self.data) if isinstance(self.data, dict) else self.data.size)
            
            # Connect tensor to dimensions
            for i in range(self.rank):
                G.add_edge("tensor", f"dim_{i}")
            
            # Get positions using spring layout
            pos = nx.spring_layout(G)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Draw dimension nodes as blue circles
            dim_nodes = [n for n in G.nodes if G.nodes[n]['type'] == 'dimension']
            nx.draw_networkx_nodes(G, pos, nodelist=dim_nodes, node_color='skyblue', 
                                  node_size=[G.nodes[n]['size'] * 10 for n in dim_nodes])
            
            # Draw tensor node as red square
            tensor_node = [n for n in G.nodes if G.nodes[n]['type'] == 'tensor']
            nx.draw_networkx_nodes(G, pos, nodelist=tensor_node, node_color='red', 
                                  node_shape='s', node_size=500)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.7)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_weight='bold')
            
            plt.title(f"Tensor Network Representation - Rank {self.rank}")
            plt.axis('off')
            
            if show:
                plt.show()
                
            return plt.gcf()
        
        except ImportError:
            logger.warning("NetworkX not available - tensor network visualization requires NetworkX")
            return None

    def to_hyperbolic_space(self, curvature=-1.0):
        """
        Transform tensor to hyperbolic space representation.
        Useful for hierarchical data modeling.
        
        Args:
            curvature: float - hyperbolic space curvature
            
        Returns:
            RecursiveTensor: Hyperbolic space representation
        """
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=np.complex64,  # Hyperbolic space uses complex numbers
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        
        # Define Poincaré ball model transformation
        def to_poincare(x):
            norm = np.linalg.norm(x)
            if norm >= 1.0:
                # Project to unit ball
                x = x / (norm + 1e-8)
            return x
        
        # Define hyperbolic distance
        def hyperbolic_distance(x, y):
            x_norm = np.linalg.norm(x)
            y_norm = np.linalg.norm(y)
            numerator = 2 * np.linalg.norm(x - y) ** 2
            denominator = (1 - x_norm**2) * (1 - y_norm**2)
            return np.arccosh(1 + numerator / (denominator + 1e-8))
        
        if isinstance(self.data, dict):
            # Transform sparse tensor
            result.data = {}
            
            # Get normalized indices
            indices = np.array(list(self.data.keys()))
            if indices.size > 0:
                # Normalize indices to be in [0,1]
                normalized_indices = indices / self.dimensions
                
                # Transform to Poincaré ball model
                for idx, val in self.data.items():
                    norm_idx = np.array(idx) / self.dimensions
                    poincare_idx = to_poincare(norm_idx)
                    
                    # Store with original indices but hyperbolic value
                    magnitude = abs(val)
                    phase = np.angle(val) if np.iscomplexobj(val) else 0
                    
                    # Compute hyperbolic value using distance from origin
                    hyp_distance = hyperbolic_distance(poincare_idx, np.zeros_like(poincare_idx))
                    
                    # Create complex value with phase
                    hyp_val = magnitude * np.exp(1j * (phase + curvature * hyp_distance))
                    result.data[idx] = hyp_val
            
        else:
            # For dense tensor, we need to create a new tensor with hyperbolic coordinates
            result.data = np.zeros(self.data.shape, dtype=np.complex64)
            
            # Get all indices
            indices = np.array(list(np.ndindex(self.data.shape)))
            
            # Process each index
            for idx in indices:
                norm_idx = np.array(idx) / self.dimensions
                poincare_idx = to_poincare(norm_idx)
                
                val = self.data[idx]
                magnitude = abs(val)
                phase = np.angle(val) if np.iscomplexobj(val) else 0
                
                # Compute hyperbolic value
                hyp_distance = hyperbolic_distance(poincare_idx, np.zeros_like(poincare_idx))
                hyp_val = magnitude * np.exp(1j * (phase + curvature * hyp_distance))
                
                result.data[idx] = hyp_val
        
        result.metadata["hyperbolic"] = True
        result.metadata["curvature"] = curvature
        result.metadata["description"] = f"Hyperbolic transformation of {self.metadata['description']}"
        
        return result

    def apply_temporal_convolution(self, kernel, time_axis=0):
        """
        Apply convolution along temporal dimension.
        
        Args:
            kernel: array - convolution kernel
            time_axis: int - axis representing time
            
        Returns:
            RecursiveTensor: Result of temporal convolution
        """
        result = RecursiveTensor(self.dimensions, 
                               rank=self.rank, 
                               dtype=self.dtype,
                               distribution=self.distribution,
                               sparsity=self.sparsity)
        
        if isinstance(self.data, dict):
            result.data = {}

            grouped_indices = {}
            for idx, val in self.data.items():
                non_time_idx = idx[:time_axis] + idx[time_axis+1:]

                if non_time_idx not in grouped_indices:
                    grouped_indices[non_time_idx] = {}

                time_idx = idx[time_axis]
                grouped_indices[non_time_idx][time_idx] = val

            for non_time_idx, time_values in grouped_indices.items():
                times = np.array(list(time_values.keys()))
                values = np.array(list(time_values.values()))

                sort_idx = np.argsort(times)
                times = times[sort_idx]
                values = values[sort_idx]

                convolved = np.convolve(values, kernel, mode='same')

                for i, t in enumerate(times):
                    full_idx = non_time_idx[:time_axis] + (t,) + non_time_idx[time_axis:]
                    if abs(convolved[i]) > 1e-10:
                        result.data[full_idx] = convolved[i]

        else:
            kernel_shape = [1] * self.rank
            kernel_shape[time_axis] = len(kernel)
            kernel = np.array(kernel)
            kernel = np.reshape(kernel, kernel_shape)

            from scipy.signal import convolve
            result.data = convolve(self.data, kernel, mode='same')

        result.operation_history = self.operation_history.copy()
        result.operation_history.append(('temporal_convolution', time_axis))
        result.metadata = self.metadata.copy()
        result.metadata["description"] = f"Temporal convolution of {self.metadata['description']}"
        return result

    def visualize_3d_tensor_field(self, axes=(0, 1, 2), slice_indices=None, cmap='viridis', title=None, show=True):
        """
        Visualize 3D tensor field as a volumetric plot or isosurface.

        Args:
            axes: tuple - which three axes to use for 3D visualization
            slice_indices: dict - indices for other axes
            cmap: str - matplotlib colormap name
            title: str - plot title
            show: bool - whether to display the plot

        Returns:
            matplotlib.figure.Figure: Figure with 3D tensor field visualization
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize

            if self.rank < 3:
                raise ValueError("Tensor must have rank >= 3 for 3D visualization")

            if slice_indices is None:
                slice_indices = {i: 0 for i in range(self.rank) if i not in axes}

            # Extract 3D slice
            if isinstance(self.data, dict):
                # For sparse tensors, create a 3D grid
                tensor_3d = np.zeros((self.dimensions, self.dimensions, self.dimensions))
                for idx, val in self.data.items():
                    if all(idx[ax] == slice_idx for ax, slice_idx in slice_indices.items()):
                        if all(idx[axis] < self.dimensions for axis in axes):
                            tensor_3d[idx[axes[0]], idx[axes[1]], idx[axes[2]]] = abs(val)
            else:
                slice_idx = tuple(slice(None) if i in axes else slice_indices.get(i, 0)
                                 for i in range(self.rank))
                tensor_3d = np.abs(self.data[slice_idx])

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Create meshgrid for 3D plotting
            x, y, z = np.meshgrid(range(self.dimensions), range(self.dimensions), range(self.dimensions), indexing='ij')

            # Flatten arrays for scatter plot
            x_flat = x.flatten()
            y_flat = y.flatten()
            z_flat = z.flatten()
            values_flat = tensor_3d.flatten()

            # Filter out zero values for cleaner visualization
            mask = values_flat > 1e-6
            x_flat = x_flat[mask]
            y_flat = y_flat[mask]
            z_flat = z_flat[mask]
            values_flat = values_flat[mask]

            if len(values_flat) > 0:
                # Normalize colors
                norm = Normalize(vmin=values_flat.min(), vmax=values_flat.max())
                colors = plt.cm.get_cmap(cmap)(norm(values_flat))

                # Create scatter plot
                scatter = ax.scatter(x_flat, y_flat, z_flat, c=values_flat,
                                   cmap=cmap, alpha=0.6, s=20)

                # Add colorbar
                plt.colorbar(scatter, ax=ax, shrink=0.8, label='Magnitude')

            ax.set_xlabel(f'Axis {axes[0]}')
            ax.set_ylabel(f'Axis {axes[1]}')
            ax.set_zlabel(f'Axis {axes[2]}')
            ax.set_title(title or f'3D Tensor Field Visualization - Axes {axes}')

            if show:
                plt.show()

            return fig

        except ImportError:
            logger.warning("matplotlib or mpl_toolkits not available - 3D visualization requires matplotlib")
            return None

    def visualize_eigenspectrum(self, k=10, title=None, show=True):
        """
        Visualize the eigenspectrum of the tensor.
        
        Args:
            k: int - number of eigenvalues to compute
            title: str - plot title
            show: bool - whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: Figure with eigenspectrum visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            # Compute eigenvalues
            if isinstance(self.data, dict):
                # For sparse tensors, use sparse eigensolver
                vals, _ = self.compute_eigenstates(k=k)
            else:
                # For dense tensors
                if self.rank == 2:
                    vals = np.linalg.eigvals(self.data)
                    # Sort by magnitude
                    idx = np.argsort(np.abs(vals))[::-1]
                    vals = vals[idx[:k]]
                else:
                    # For higher rank, unfold and compute SVD
                    unfolded = self.data.reshape(self.dimensions, -1)
                    U, S, V = np.linalg.svd(unfolded, full_matrices=False)
                    vals = S[:k]
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Magnitude plot
            ax1.plot(np.abs(vals), 'o-', linewidth=2)
            ax1.set_title('Eigenvalue Magnitudes')
            ax1.set_xlabel('Index')
            ax1.set_ylabel('|λ|')
            ax1.grid(True, alpha=0.3)
            
            # Complex plane plot (if complex)
            if np.iscomplexobj(vals):
                ax2.scatter(np.real(vals), np.imag(vals), alpha=0.6)
                ax2.set_title('Eigenvalue Spectrum (Complex Plane)')
                ax2.set_xlabel('Re(λ)')
                ax2.set_ylabel('Im(λ)')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(0, color='black', linewidth=0.5)
                ax2.axvline(0, color='black', linewidth=0.5)
            else:
                ax2.hist(np.real(vals), bins=min(len(vals), 20), alpha=0.7)
                ax2.set_title('Eigenvalue Distribution')
                ax2.set_xlabel('Value')
                ax2.set_ylabel('Count')
                ax2.grid(True, alpha=0.3)
            
            if title:
                fig.suptitle(title)
                
            if show:
                plt.show()
                
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available - eigenspectrum visualization requires matplotlib")
            return None
        except Exception as e:
            logger.warning(f"Failed to visualize eigenspectrum: {e}")
            return None

    def visualize_tensor_evolution(self, evolution_steps=10, cmap='plasma', title=None, show=True):
        """
        Visualize tensor evolution over time or iterations.

        Args:
            evolution_steps: int - number of evolution steps to visualize
            cmap: str - matplotlib colormap name
            title: str - plot title
            show: bool - whether to display the plot

        Returns:
            matplotlib.figure.Figure: Figure with tensor evolution visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            # Generate evolution data (simplified fractal iteration)
            evolution_data = [self.data.copy()]
            current_tensor = self

            for step in range(evolution_steps):
                # Apply simple evolution (could be fractal iteration, contraction, etc.)
                evolved = current_tensor.apply_function(lambda x: x * 0.9 + 0.1 * np.sin(step * 0.1))
                evolution_data.append(evolved.data)
                current_tensor = evolved

            def animate(frame):
                for i, ax in enumerate(axes):
                    ax.clear()

                    if frame < len(evolution_data):
                        data = evolution_data[frame]
                        if isinstance(data, dict):
                            # Convert sparse to dense for visualization
                            dense_data = np.zeros((self.dimensions, self.dimensions))
                            for idx, val in data.items():
                                if len(idx) >= 2 and idx[0] < self.dimensions and idx[1] < self.dimensions:
                                    dense_data[idx[0], idx[1]] = abs(val)
                        else:
                            dense_data = np.abs(data.reshape(self.dimensions, self.dimensions) if data.ndim > 1 else data)

                        im = ax.imshow(dense_data, cmap=cmap, aspect='equal')
                        ax.set_title(f'Step {frame}')
                        ax.axis('off')

                        if i == 0:  # Add colorbar to first subplot
                            plt.colorbar(im, ax=ax, shrink=0.8)

                fig.suptitle(title or f'Tensor Evolution - Frame {frame}/{len(evolution_data)-1}')

            # Create animation
            anim = animation.FuncAnimation(fig, animate, frames=len(evolution_data),
                                         interval=500, repeat=True)

            if show:
                plt.show()

            return fig

        except ImportError:
            logger.warning("matplotlib not available - tensor evolution visualization requires matplotlib")
            return None

    def visualize_phase_space(self, axes=(0, 1), resolution=50, title=None, show=True):
        """
        Visualize tensor in phase space representation.

        Args:
            axes: tuple - axes to use for phase space
            resolution: int - resolution for phase space grid
            title: str - plot title
            show: bool - whether to display the plot

        Returns:
            matplotlib.figure.Figure: Figure with phase space visualization
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Extract data for phase space
            if isinstance(self.data, dict):
                values = np.array(list(self.data.values()))
                indices_list = list(self.data.keys())
                if not indices_list:
                    logger.warning("Empty sparse tensor")
                    return None
                # Ensure all indices have the same length
                index_lengths = [len(idx) for idx in indices_list]
                if len(set(index_lengths)) != 1:
                    logger.warning("Inconsistent index lengths in sparse tensor")
                    return None
                indices = np.array(indices_list)
            else:
                values = self.data.flatten()
                try:
                    indices = np.array(list(np.ndindex(self.data.shape)))
                except Exception as e:
                    logger.warning(f"Failed to create indices for dense tensor: {e}")
                    return None

            if len(values) == 0:
                logger.warning("No values to visualize")
                return None

            # Real vs Imaginary parts (for complex tensors)
            if np.iscomplexobj(values):
                real_parts = np.real(values)
                imag_parts = np.imag(values)

                # Phase space plot
                ax1.scatter(real_parts, imag_parts, alpha=0.6, s=10)
                ax1.set_xlabel('Real Part')
                ax1.set_ylabel('Imaginary Part')
                ax1.set_title('Complex Phase Space')
                ax1.grid(True, alpha=0.3)

                # Magnitude distribution
                magnitudes = np.abs(values)
                ax2.hist(magnitudes, bins=resolution, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Magnitude')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Magnitude Distribution')
                ax2.grid(True, alpha=0.3)
            else:
                # For real tensors, show value distribution and spatial distribution
                ax1.hist(values, bins=resolution, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Value Distribution')
                ax1.grid(True, alpha=0.3)

                # Spatial distribution (first two dimensions)
                if indices.shape[0] > 0 and indices.shape[1] >= 2:
                    try:
                        x_coords = indices[:, axes[0]] if axes[0] < indices.shape[1] else indices[:, 0]
                        y_coords = indices[:, axes[1]] if axes[1] < indices.shape[1] else indices[:, min(1, indices.shape[1]-1)]

                        scatter = ax2.scatter(x_coords, y_coords, c=values, cmap='viridis', alpha=0.6, s=20)
                        ax2.set_xlabel(f'Axis {axes[0]}')
                        ax2.set_ylabel(f'Axis {axes[1]}')
                        ax2.set_title('Spatial Distribution')
                        plt.colorbar(scatter, ax=ax2, shrink=0.8, label='Value')
                    except Exception as e:
                        logger.warning(f"Failed to create spatial distribution plot: {e}")
                        ax2.text(0.5, 0.5, 'Spatial plot failed', ha='center', va='center', transform=ax2.transAxes)
                else:
                    ax2.text(0.5, 0.5, 'Insufficient dimensions\nfor spatial plot', ha='center', va='center', transform=ax2.transAxes)

            fig.suptitle(title or 'Tensor Phase Space Analysis')

            if show:
                plt.show()

            return fig

        except ImportError:
            logger.warning("matplotlib not available - phase space visualization requires matplotlib")
            return None
        except Exception as e:
            logger.warning(f"Failed to visualize phase space: {e}")
            return None

    def visualize_correlation_matrix(self, title=None, show=True):
        """
        Visualize correlation matrix between tensor dimensions.

        Args:
            title: str - plot title
            show: bool - whether to display the plot

        Returns:
            matplotlib.figure.Figure: Figure with correlation matrix visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Convert tensor to matrix form for correlation analysis
            if isinstance(self.data, dict):
                # For sparse tensors, create unfolded matrices
                unfolded_matrices = []
                for mode in range(self.rank):
                    matrix = np.zeros((self.dimensions, self.dimensions**(self.rank-1)))
                    for idx, val in self.data.items():
                        row = idx[mode]
                        col = sum(idx[i] * (self.dimensions**i) for i in range(self.rank) if i != mode)
                        if col < matrix.shape[1]:
                            matrix[row, col] = val
                    unfolded_matrices.append(matrix)
            else:
                # For dense tensors, unfold along each mode
                unfolded_matrices = []
                for mode in range(self.rank):
                    unfolded = np.moveaxis(self.data, mode, 0)
                    shape = unfolded.shape
                    matrix = unfolded.reshape(shape[0], -1)
                    unfolded_matrices.append(matrix)

            # Compute correlations between unfolded matrices
            correlations = np.zeros((self.rank, self.rank))
            for i in range(self.rank):
                for j in range(self.rank):
                    if i != j:
                        # Compute correlation between unfolded matrices
                        mat1 = unfolded_matrices[i]
                        mat2 = unfolded_matrices[j]

                        # Flatten and compute correlation
                        flat1 = mat1.flatten()
                        flat2 = mat2.flatten()

                        if np.std(flat1) > 0 and np.std(flat2) > 0:
                            corr = np.corrcoef(flat1, flat2)[0, 1]
                            correlations[i, j] = corr
                        else:
                            correlations[i, j] = 0
                    else:
                        correlations[i, j] = 1.0

            # Visualize correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0,
                       xticklabels=[f'Mode {i}' for i in range(self.rank)],
                       yticklabels=[f'Mode {i}' for i in range(self.rank)],
                       ax=ax)
            ax.set_title(title or 'Tensor Mode Correlation Matrix')

            if show:
                plt.show()

            return fig

        except ImportError:
            logger.warning("matplotlib and/or seaborn not available - correlation matrix visualization requires matplotlib and seaborn")
            return None

    def create_comprehensive_visualization_dashboard(self, save_path=None):
        """
        Create a comprehensive visualization dashboard for the tensor.

        Args:
            save_path: str - path to save the dashboard (optional)

        Returns:
            dict: Dictionary containing all visualization figures
        """
        dashboard = {}

        # Helper function to safely call visualizers and ensure they return valid results
        def safe_visualize(name, func, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Ensure result is either None or a matplotlib Figure
                if result is not None:
                    # Check if it's a matplotlib Figure
                    try:
                        from matplotlib.figure import Figure
                        if isinstance(result, Figure):
                            return result
                        else:
                            logger.warning(f"Visualizer '{name}' returned unexpected type {type(result)}, expected matplotlib Figure or None")
                            return None
                    except ImportError:
                        # matplotlib not available, result should be None
                        return None
                return result
            except Exception as e:
                logger.warning(f"Visualizer '{name}' failed: {e}")
                return None

        # Create visualizations with error handling
        dashboard['slice'] = safe_visualize('slice', self.visualize_slice, show=False)
        dashboard['eigenspectrum'] = safe_visualize('eigenspectrum', self.visualize_eigenspectrum, show=False)
        dashboard['density'] = safe_visualize('density', self.visualize_density, show=False)
        dashboard['history'] = safe_visualize('history', self.plot_operation_history, show=False)

        if self.rank >= 2:
            dashboard['network'] = safe_visualize('network', self.visualize_tensor_network, show=False)
            dashboard['phase_space'] = safe_visualize('phase_space', self.visualize_phase_space, show=False)
            dashboard['correlation'] = safe_visualize('correlation', self.visualize_correlation_matrix, show=False)

        if self.rank >= 3:
            dashboard['3d_field'] = safe_visualize('3d_field', self.visualize_3d_tensor_field, show=False)

        # Save dashboard if path provided
        if save_path:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                for name, fig in dashboard.items():
                    if isinstance(fig, Figure):
                        try:
                            fig.savefig(f"{save_path}/tensor_{name}.png", dpi=300, bbox_inches='tight')
                            plt.close(fig)
                        except Exception as e:
                            logger.warning(f"Failed to save figure '{name}': {e}")
            except Exception as e:
                logger.warning(f"Failed during dashboard save: {e}")

        return dashboard

    # Add layout property for matplotlib compatibility
    @property
    def layout(self):
        """
        Layout property for matplotlib compatibility.
        Returns basic layout information about the tensor.
        """
        return {
            'dimensions': self.dimensions,
            'rank': self.rank,
            'shape': self.shape,
            'dtype': self.dtype,
            'sparsity': self.sparsity,
            'size': len(self.data) if isinstance(self.data, dict) else self.data.size
        }
    
    def _detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect repeating patterns in tensor data for compression"""
        patterns = []
        
        if not isinstance(self.data, dict) or len(self.data) < 8:
            return patterns
        
        # Convert tensor data to sequence for pattern detection
        items = list(self.data.items())
        
        # Look for repeating subsequences of length 2-4
        for pattern_len in range(2, min(5, len(items) // 2 + 1)):
            pattern_counts = {}
            
            for i in range(len(items) - pattern_len + 1):
                # Extract pattern (indices and values)
                pattern_indices = []
                pattern_values = []
                
                for j in range(pattern_len):
                    idx, val = items[i + j]
                    pattern_indices.append(idx)
                    pattern_values.append(val)
                
                # Create pattern signature
                pattern_sig = tuple(pattern_values)
                
                if pattern_sig not in pattern_counts:
                    pattern_counts[pattern_sig] = []
                pattern_counts[pattern_sig].append((i, pattern_indices))
            
            # Find patterns that occur multiple times
            for pattern_sig, occurrences in pattern_counts.items():
                if len(occurrences) >= 2:  # Pattern occurs at least twice
                    # Encode pattern data
                    pattern_data = b''
                    for val in pattern_sig:
                        if np.iscomplexobj(val):
                            pattern_data += struct.pack('<ff', float(val.real), float(val.imag))
                        else:
                            pattern_data += struct.pack('<f', float(val))
                    
                    # Encode location data
                    locations_data = b''
                    for start_pos, indices in occurrences:
                        locations_data += struct.pack('<I', start_pos)  # Start position
                        for idx in indices:
                            # Pack index as 5 uint16 values
                            padded_idx = list(idx) + [0] * (5 - len(idx))
                            locations_data += struct.pack('<5H', *padded_idx[:5])
                    
                    patterns.append({
                        'data': pattern_data,
                        'locations': locations_data,
                        'occurrences': len(occurrences),
                        'length': pattern_len
                    })
        
        # Sort by compression benefit (occurrences * pattern_length)
        patterns.sort(key=lambda p: p['occurrences'] * p['length'], reverse=True)
        
        # Return top patterns to avoid excessive compression overhead
        return patterns[:10]
    
    def _apply_references(self) -> None:
        """Apply reconstructed references to tensor data"""
        if not hasattr(self, '_references') or not self._references:
            return
        
        for ref in self._references:
            ref_type = ref['type']
            source_idx = tuple(ref['source'][:self.rank])
            target_idx = tuple(ref['target'][:self.rank])
            
            if source_idx in self.data:
                source_val = self.data[source_idx]
                
                if ref_type == 0x00:  # Direct reference
                    self.data[target_idx] = source_val
                elif ref_type == 0x02:  # Fractal reference
                    if len(ref.get('parameters', b'')) >= 4:
                        ratio = struct.unpack('<f', ref['parameters'][:4])[0]
                        self.data[target_idx] = source_val * ratio

    # ---------------- Deduplication Helpers ----------------
    def _prepare_deduplication(self, max_refs: int = 4096) -> None:
        """Detect duplicate values in sparse tensor and prepare reference list.

        Strategy:
        - Identify first occurrence of each float value (rounded to 6 decimals to improve hit rate while stable)
        - Record subsequent occurrences as references
        - Build reduced index/value lists containing only unique value sources
        - Store results in transient attributes used by index/value/reference writers
        """
        self._dedup_checked = True
        if not isinstance(self.data, dict) or len(self.data) < 3:
            self._dedup_ready = False
            return
        value_sources: Dict[float, Tuple] = {}
        references: List[Tuple[int, Tuple, Tuple, bytes]] = []
        unique_indices: List[Tuple] = []
        unique_values: List[float] = []
        for idx, val in self.data.items():
            # Only handle scalar real/complex values; for complex use real+imag tuple key
            if np.iscomplexobj(val):
                key = (round(float(val.real), 6), round(float(val.imag), 6), 'c')
            else:
                key = round(float(val), 6)
            if key not in value_sources:
                value_sources[key] = idx
                unique_indices.append(idx)
                unique_values.append(val)
            else:
                src_idx = value_sources[key]
                references.append((0x00, src_idx, idx, b''))  # direct reference
                if len(references) >= max_refs:
                    break
        # Only enable if we actually saved something
        if references and (len(unique_indices) + len(references) == len(self.data)):
            self._dedup_indices = unique_indices
            self._dedup_values = unique_values
            self._dedup_references = references
            self._dedup_ready = True
            self.metadata['deduplicated'] = True
        else:
            self._dedup_ready = False
    
    def _apply_patterns(self) -> None:
        """Apply reconstructed patterns to tensor data"""
        if not hasattr(self, '_patterns') or not self._patterns:
            return
        
        # Pattern application is complex and would require 
        # reversing the compression algorithm. For now, just
        # store the patterns as metadata for potential use.
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        self.metadata['compressed_patterns'] = len(self._patterns)
    
    def _compress_block_data(self, data: bytes) -> Tuple[bytes, int, int]:
        """Compress block data if beneficial"""
        import zlib
        
        if not self._enable_compression or len(data) < self._compression_threshold:
            return data, 0x00, len(data)  # No compression
        
        try:
            if self._compression_method == 0x01:  # DEFLATE (zlib)
                compressed = zlib.compress(data, level=6)
                if len(compressed) < len(data) * 0.8:  # At least 20% reduction
                    return compressed, 0x01, len(data)
        except Exception:
            pass
        
        return data, 0x00, len(data)  # Fall back to no compression
    
    def _decompress_block_data(self, data: bytes, method: int, uncompressed_size: int) -> bytes:
        """Decompress block data"""
        import zlib
        
        if method == 0x00:  # No compression
            return data
        elif method == 0x01:  # DEFLATE (zlib)
            try:
                decompressed = zlib.decompress(data)
                if len(decompressed) == uncompressed_size:
                    return decompressed
            except Exception:
                pass
        
        # If decompression fails, return original data
        logger.warning(f"Failed to decompress block with method {method}")
        return data
    
    def enable_compression(self, threshold: int = 1024, method: int = 0x01) -> None:
        """Enable block-level compression for RTA format
        
        Args:
            threshold: Minimum block size in bytes to trigger compression
            method: Compression method (0x01=DEFLATE/zlib, 0x02=LZMA, etc.)
        """
        self._enable_compression = True
        self._compression_threshold = threshold
        self._compression_method = method
    
    def disable_compression(self) -> None:
        """Disable block-level compression"""
        self._enable_compression = False


# Example usage
if __name__ == "__main__":
    # Import eigenrecursion components
    try:
        from rcf_integration.eigenrecursive_operations import (
            EigenstateConvergenceEngine, 
            EigenstateConfig,
            EigenrecursiveOperator,
            ConsciousnessEigenoperator
        )
        eigenrecursion_available = True
    except ImportError:
        print("Eigenrecursion modules not available")
        eigenrecursion_available = False

    # Example 1: Create and visualize a recursive tensor
    tensor = RecursiveTensor(
        dimensions=10,
        rank=3,
        dtype=np.float32,
        distribution='uniform',
        sparsity=0.1
    )

    print("Created tensor:", tensor)
    print("Tensor layout:", tensor.layout)

    # Enable compression for RTA output
    tensor.enable_compression(threshold=512, method=0x01)  # DEFLATE compression

    # Perform eigenrecursion computations if available
    if eigenrecursion_available and torch_available:
        try:
            print("Performing eigenrecursion convergence...")
            
            # Configure eigenstate convergence
            config = EigenstateConfig(
                max_iterations=100,
                convergence_threshold=1e-6,
                eigenstate_type='consciousness_eigenstate',
                consciousness_threshold=0.7
            )
            
            # Create convergence engine
            engine = EigenstateConvergenceEngine(config)
            
            # Prepare initial state from tensor
            if isinstance(tensor.data, dict):
                initial_state = torch.tensor(list(tensor.data.values()), dtype=torch.float32)
            else:
                initial_state = torch.tensor(tensor.data.flatten(), dtype=torch.float32)
            
            # Create eigenrecursive operator
            operator = ConsciousnessEigenoperator(
                state_dim=initial_state.numel(),
                config=config
            )
            
            # Converge to eigenstate
            result = engine.converge_to_eigenstate(
                initial_state=initial_state,
                operator=operator
            )
            
            print(f"Eigenrecursion converged: {result.converged}")
            print(f"Final consciousness score: {result.consciousness_score:.4f}")
            print(f"Identity preservation: {result.identity_preservation_score:.4f}")
            
            # Update tensor with converged state
            converged_data = result.final_state.detach().numpy()
            if isinstance(tensor.data, dict):
                # Map back to sparse format (simplified)
                for i, (idx, _) in enumerate(tensor.data.items()):
                    if i < len(converged_data):
                        tensor.data[idx] = converged_data[i]
            else:
                tensor.data = converged_data.reshape(tensor.data.shape)
            
            tensor.metadata["eigenrecursion_performed"] = True
            tensor.metadata["consciousness_score"] = result.consciousness_score
            tensor.metadata["convergence_iterations"] = result.iterations
            
        except Exception as e:
            print(f"Eigenrecursion computation failed: {e}")
    elif not torch_available:
        print("PyTorch not available - skipping eigenrecursion computations")
    else:
        print("Eigenrecursion modules not available - skipping advanced computations")

    # Save tensor to compressed RTA format
    try:
        rta_path = tensor.save_rta("tensor_output_compiled.rta")
        print(f"Tensor saved to compressed RTA format: {rta_path}")
    except Exception as e:
        print(f"RTA save error: {e}")

    # Example 2: Comprehensive visualization
    try:
        dashboard = tensor.create_comprehensive_visualization_dashboard()
        print(f"Created {len(dashboard)} visualizations")
    except Exception as e:
        print(f"Visualization error: {e}")
