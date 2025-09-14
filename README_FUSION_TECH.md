# Fusion Techniques Documentation

## Table of Contents
- [Overview](#overview)
- [Active Fusion Techniques](#active-fusion-techniques)
  - [General_Adapt_and_Fuse](#general_adapt_and_fuse)
  - [Diff_KF_time_measure](#diff_kf_time_measure)
  - [Diff_KF_info_matrix](#diff_kf_info_matrix)
  - [Noon_coop (Non-Cooperative)](#noon_coop-non-cooperative)
- [Implementation Architecture](#implementation-architecture)
- [Usage Patterns](#usage-patterns)
- [Cooperation_type Enumeration](#cooperation_type-enumeration)
- [Performance Considerations](#performance-considerations)

## Overview

Fusion techniques in the CPE769 multiagent adaptive networks system enable agents to share and combine information from their neighbors to improve collective estimation performance. These techniques implement various distributed learning and consensus algorithms for cooperative state estimation.

### Design Philosophy
- **Distributed Processing**: No central coordinator required
- **Neighbor-Based Communication**: Agents only communicate with direct neighbors
- **Weighted Information Fusion**: Flexible weighting schemes for different network topologies
- **Modular Integration**: Compatible with different agent techniques (KF, RLS)

## Active Fusion Techniques

### General_Adapt_and_Fuse

**Status**: ✅ **Primary Fusion Technique** - Recommended for most applications

#### Implementation Details
General_Adapt_and_Fuse implements a covariance-weighted fusion strategy that optimally combines state estimates from neighboring agents.

<augment_code_snippet path="Technique/General_Adapt_and_Fuse.m" mode="EXCERPT">
````matlab
classdef General_Adapt_and_Fuse < Fusion_technique
    properties
        % General adaptive fusion implementation
    end
    
    methods
        function s = apply(obj, varargin)
            % Covariance-weighted fusion of neighbor estimates
            P_inv = zeros(p.Results.dim);
            for ind = 1:length(obj.neighbors)
                P_inv = P_inv + obj.neighbors_weights(ind) * inv(obj.neighbors(ind).agent_technique.get_params('covariance_estimate').covariance_estimate);
            end
            P = pinv(P_inv); 
            s.covariance_estimate = P;
````
</augment_code_snippet>

#### Key Features
- **Optimal Fusion**: Uses inverse covariance weighting for optimal information combination
- **Universal Compatibility**: Works with both KF and RLS agent techniques
- **Flexible Weighting**: Supports arbitrary neighbor weight configurations
- **Robust Implementation**: Handles singular covariance matrices using pseudo-inverse

#### Mathematical Foundation
- **Information Matrix Fusion**: Combines information matrices (inverse covariances) from neighbors
- **Weighted Averaging**: Final estimate is weighted average based on estimation uncertainties
- **Optimal Properties**: Minimizes mean square error under Gaussian assumptions
- **Theoretical Foundation**: Based on Merched's diffusion adaptation paper

#### Usage Example
<augment_code_snippet path="test_MAS_sim.m" mode="EXCERPT">
````matlab
% Create General_Adapt_and_Fuse fusion technique
fusion_tech = General_Adapt_and_Fuse('neighbors', neighbor_agents, ...
                                    'neighbors_weights', weights);
````
</augment_code_snippet>

### Diff_KF_time_measure

**Status**: ✅ **Specialized for KF** - Diffusion Kalman Filter with time measure fusion

#### Implementation Details
Implements the diffusion Kalman filter algorithm with time measure fusion, specifically designed for Kalman filter agents.

<augment_code_snippet path="Technique/Diff_KF_time_measure.m" mode="EXCERPT">
````matlab
classdef Diff_KF_time_measure < Fusion_technique
    properties
        phi_k       % Incremental step state estimate
        P_k         % Incremental step covariance estimate
    end
    
    methods
        function s = apply(obj, varargin)
            % Diffusion step (Fusion step)
            x = zeros(p.Results.dim, 1);
            for ind = 1:length(obj.neighbors)
                x(:,1) = x(:,1) + obj.neighbors_weights(ind) * obj.neighbors(ind).fusion_technique.phi_k;
            end
````
</augment_code_snippet>

#### Key Features
- **Two-Step Process**: Incremental step followed by diffusion step
- **KF-Specific**: Designed specifically for Kalman filter agents
- **Distributed Consensus**: Implements distributed averaging of intermediate estimates
- **Theoretical Foundation**: Based on Sayed's diffusion adaptation theory

#### Algorithm Steps
1. **Incremental Step**: Each agent performs local Kalman update
2. **Diffusion Step**: Agents combine intermediate estimates from neighbors
3. **State Propagation**: Results used for next time step

### Diff_KF_info_matrix

**Status**: ✅ **Advanced KF Fusion** - Diffusion Kalman Filter with information matrix fusion

#### Implementation Details
Implements diffusion Kalman filtering using information matrix (inverse covariance) fusion for enhanced numerical stability.

<augment_code_snippet path="Technique/Diff_KF_info_matrix.m" mode="EXCERPT">
````matlab
classdef Diff_KF_info_matrix < Fusion_technique
    properties
        phi_k       % Incremental step state estimate
        P_k         % Incremental step covariance estimate
    end
    
    methods
        function apply_incremental_step(obj, varargin)
            % Incremental step using information matrix fusion
            S_k_inv = zeros(p.Results.y_dim);
            q_k = zeros(p.Results.y_dim, 1);
            
            for ind = 1:length(obj.neighbors)
                S_k_inv = S_k_inv + obj.neighbors(ind).agent_technique.H' * pinv(obj.neighbors(ind).agent_technique.R) * obj.neighbors(ind).agent_technique.H;
                q_k = q_k + obj.neighbors(ind).agent_technique.H' * pinv(obj.neighbors(ind).agent_technique.R) * obj.neighbors(ind).agent_technique.y;
            end
````
</augment_code_snippet>

#### Key Features
- **Information Matrix Approach**: Uses information matrices for numerical stability
- **Two-Phase Algorithm**: Incremental step followed by diffusion step
- **Enhanced Robustness**: Better numerical properties than covariance-based methods
- **KF Integration**: Seamlessly integrates with Kalman filter agents

#### Mathematical Foundation
- **Information Form**: Works in information space (inverse covariance domain)
- **Distributed Information Fusion**: Combines information contributions from all neighbors
- **Consensus Achievement**: Achieves network-wide consensus through local interactions
- **Theoretical Foundation**: Based on Sayed's diffusion adaptation theory

### Noon_coop (Non-Cooperative)

**Status**: ✅ **Baseline Reference** - Non-cooperative mode for performance comparison

#### Implementation Details
Implements a dummy fusion technique that performs no information sharing, serving as a baseline for performance comparison.

<augment_code_snippet path="Technique/Noon_coop.m" mode="EXCERPT">
````matlab
classdef Noon_coop < Fusion_technique
    methods
        function s = apply(obj, varargin)
            % Non-cooperative filtering: Do nothing (dummy)
            s = struct();
        end
    end
````
</augment_code_snippet>

#### Key Features
- **No Information Sharing**: Agents operate independently
- **Baseline Performance**: Provides reference for cooperative methods
- **Minimal Overhead**: No computational or communication overhead
- **Compatibility**: Works with all agent techniques

## Implementation Architecture

### Abstract Base Class
All fusion techniques inherit from the `Fusion_technique` abstract class:

<augment_code_snippet path="Technique/Fusion_technique.m" mode="EXCERPT">
````matlab
classdef Fusion_technique < handle
    properties
        neighbors           % Array of neighbor agents
        neighbors_weights   % Weights for neighbor information
    end
    methods (Abstract)
        apply(obj, varargin)  % Main fusion operation
    end
````
</augment_code_snippet>

### Key Properties
- **`neighbors`**: Array of Agent objects representing network neighbors
- **`neighbors_weights`**: Numerical weights for combining neighbor information
- **Handle Class**: Enables reference semantics for efficient memory usage

### Key Methods
- **`apply()`**: Main fusion operation that combines neighbor information
- **`reevaluate_weights()`**: Updates neighbor weights based on network changes

## Usage Patterns

### Standard Workflow
1. **Define Network Topology**: Specify neighbor relationships for each agent
2. **Create Fusion Technique**: Instantiate appropriate fusion class with neighbors and weights
3. **Assign to Agent**: Set the fusion technique for each agent
4. **Social Learning**: Call agent's `social_learning_step()` to perform fusion

### Neighbor Configuration
```matlab
% Define neighbors for agent 'a'
neighbors_idx = neighbor_lists{a};
neighbor_agents = cell(length(neighbors_idx), 1);
for i = 1:length(neighbors_idx)
    neighbor_agents{i} = agents{neighbors_idx(i)};
end
neighbor_agents = [neighbor_agents{:}]';  % Convert to array

% Equal weighting
weights = ones(1, length(neighbors_idx)) / length(neighbors_idx);
```

### Fusion Technique Selection
<augment_code_snippet path="test_MAS_sim.m" mode="EXCERPT">
````matlab
% Create fusion technique based on selection
switch str_fusion_tech
    case 'General_Adapt_and_Fuse'
        fusion_tech = General_Adapt_and_Fuse('neighbors', neighbor_agents, ...
                                    'neighbors_weights', weights);
    case 'Diff_KF_time_measure'
        if isa(agents{a}.agent_technique, 'KF')
            fusion_tech = Diff_KF_time_measure('neighbors', neighbor_agents, ...
                                    'neighbors_weights', weights);
        end
    case 'Diff_KF_info_matrix'
        if isa(agents{a}.agent_technique, 'KF')
            fusion_tech = Diff_KF_info_matrix('neighbors', neighbor_agents, ...
                                    'neighbors_weights', weights);
        end
    otherwise
        fusion_tech = Noon_coop();  % Non-cooperative baseline
end
````
</augment_code_snippet>

## Cooperation_type Enumeration

### Current Status
The `Cooperation_type` enumeration was originally intended to be implemented as an enum class following OOP best practices for more general and extensible code. However, this approach is **currently deprecated** in favor of direct fusion technique instantiation.

### Original Design Intent
- **Enum-Based Selection**: Use enumeration values to select cooperation strategies
- **Factory Pattern**: Automatic instantiation of appropriate fusion techniques
- **Extensibility**: Easy addition of new cooperation types

### Current Implementation Approach
The current implementation directly instantiates fusion technique objects, providing:
- **Direct Control**: Explicit control over fusion technique parameters
- **Flexibility**: Custom neighbor configurations and weighting schemes
- **Simplicity**: Reduced abstraction layers for easier debugging

### Limitations of Current Approach
- **Code Duplication**: Similar instantiation code repeated across applications
- **Less Extensible**: Adding new fusion techniques requires code changes in multiple places
- **Manual Configuration**: Requires explicit neighbor and weight management

## Performance Considerations

### Computational Complexity
- **General_Adapt_and_Fuse**: O(d³) per agent due to matrix inversions (d = state dimension)
- **Diff_KF_time_measure**: O(d³) per agent for state vector operations (also has matrix inversions)
- **Diff_KF_info_matrix**: O(d³) per agent due to information matrix operations
- **Noon_coop**: O(1) - no fusion overhead

### Communication Requirements
- **All Fusion Techniques**: Require neighbor state estimates and covariances
- **Network Topology**: Communication limited to direct neighbors only
- **Synchronization**: Assumes synchronous updates across the network

### Convergence Properties
- **General_Adapt_and_Fuse**: Fast convergence with optimal weighting
- **Diffusion Methods**: Theoretical convergence guarantees under connectivity assumptions
- **Network Effects**: Performance depends on network topology and connectivity

---

**Note**: For agent technique documentation, see [README_AGENT_TECH.md](README_AGENT_TECH.md).

*Recommended: Use General_Adapt_and_Fuse for general applications, diffusion methods for specialized KF scenarios.*
