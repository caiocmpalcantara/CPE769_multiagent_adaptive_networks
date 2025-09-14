# Agent Techniques Documentation

## Table of Contents
- [Overview](#overview)
- [Currently Supported Techniques](#currently-supported-techniques)
  - [KF (Kalman Filter)](#kf-kalman-filter)
  - [RLS (Recursive Least Squares)](#rls-recursive-least-squares)
- [Deprecated Techniques](#deprecated-techniques)
  - [LMS (Least Mean Squares)](#lms-least-mean-squares)
  - [Wiener Filter](#wiener-filter)
- [Implementation Architecture](#implementation-architecture)
- [Usage Patterns](#usage-patterns)
- [Testing and Validation](#testing-and-validation)

## Overview

Agent techniques in the CPE769 multiagent adaptive networks system implement various adaptive filtering algorithms for distributed state estimation. Each technique follows the **Agent_technique2** pattern (as mentioned in memories) with standardized interfaces for seamless integration with the Agent class.

### Design Philosophy
- **Unified Interface**: All techniques implement the abstract `Agent_technique` class
- **State Estimation Focus**: Algorithms estimate state vectors rather than parameters
- **Kalman-Compatible**: Techniques follow Kalman filtering paradigms where applicable
- **Modular Design**: Easy to switch between different algorithms at runtime

## Currently Supported Techniques

### KF (Kalman Filter)

**Status**: ✅ **Fully Supported** - Primary recommendation for state estimation

#### Implementation Details
The KF class implements a linear Kalman filter for optimal state estimation under Gaussian noise assumptions.

<augment_code_snippet path="Technique/KF.m" mode="EXCERPT">
````matlab
classdef KF < Kalman
    % Class: KF - Represent an agent technique that implements Linear KF
    % This class is derived from Kalman which is derived from Agent_technique
    properties
        H           % Observation matrix
        R           % Measurement noise covariance
        sigma       % Noise standard deviation
    end
````
</augment_code_snippet>

#### Key Features
- **Linear State-Space Model**: Supports linear system dynamics and observation models
- **Optimal Estimation**: Provides minimum mean square error estimates under Gaussian assumptions
- **Covariance Tracking**: Maintains state estimation uncertainty through covariance matrices
- **System Model Integration**: Works with `Linear_State` system models

#### Constructor Parameters
```matlab
kf = KF('x_dim', x_dim, 'y_dim', y_dim, ...
        'H_matrix', H_matrix, 'R_matrix', R, ...
        'Pa_init', {'delta', 0.1}, ...
        'xa_init', {'initial_state', zeros(x_dim, 1)}, ...
        'system_model', model_sys);
```

#### Usage Example
<augment_code_snippet path="test_MAS_sim.m" mode="EXCERPT">
````matlab
% Create KF technique
agent_technique = KF('x_dim', x_dim, 'y_dim', y_dim, ...
                    'H_matrix', H_matrix_init, 'R_matrix', R(a), ...
                    'Pa_init', {'delta', 0.1}, ...
                    'xa_init', {'initial_state', zeros(x_dim, 1)}, ...
                    'system_model', model_sys);
````
</augment_code_snippet>

#### Mathematical Model
- **State Equation**: `x(k+1) = A*x(k) + w(k)`
- **Observation Equation**: `y(k) = H*x(k) + v(k)`
- **Noise Assumptions**: `w(k) ~ N(0,Q)`, `v(k) ~ N(0,R)`

### RLS (Recursive Least Squares)

**Status**: ✅ **Fully Supported** - Alternative to Kalman filtering

#### Implementation Details
The RLS class implements recursive least squares for state estimation, refactored to follow the Agent_technique2 pattern.

<augment_code_snippet path="Technique/RLS.m" mode="EXCERPT">
````matlab
classdef RLS < Agent_technique
    % Recursive Least Squares (RLS) Algorithm - State Estimation Version
    % Mathematical model: y = H_matrix * x + noise (estimate x, H_matrix known)
    properties (Access = public)
        P                   % state error covariance matrix
        g                   % gain vector (similar to Kalman gain)
        alpha               % innovation vector
        H_matrix            % known observation matrix
        x_hat               % state estimate
        lambda              % forgetting factor (0 < lambda <= 1)
    end
````
</augment_code_snippet>

#### Key Features
- **Exponential Forgetting**: Uses forgetting factor λ for time-varying environments
- **Computational Efficiency**: Lower computational complexity than Kalman filter
- **Known Observation Matrix**: Assumes H_matrix is known and fixed
- **State Vector Estimation**: Estimates state x given observations y = H*x + noise

#### Constructor Parameters
```matlab
start_vals = struct('delta', 0.1, 'initial_state', zeros(x_dim, 1));
rls = RLS('x_dim', x_dim, 'y_dim', y_dim, ...
          'H_matrix', H_matrix, ...
          'lambda', 0.98, ...
          'start_vals', start_vals);
```

#### Usage Example
<augment_code_snippet path="test_MAS_sim.m" mode="EXCERPT">
````matlab
% Create RLS technique
start_vals = struct('delta', 0.1, 'initial_state', zeros(x_dim, 1));
agent_technique = RLS('x_dim', x_dim, 'y_dim', y_dim, ...
                      'H_matrix', H_matrix_init, ...
                      'lambda', 0.98, ...
                      'start_vals', start_vals);
````
</augment_code_snippet>

#### Mathematical Model
- **Observation Model**: `y(k) = H*x + v(k)`
- **Cost Function**: Minimizes weighted sum of squared errors with exponential forgetting
- **Update Equations**: Recursive updates for state estimate and covariance matrix

## Deprecated Techniques

### LMS (Least Mean Squares)

**Status**: ⚠️ **DEPRECATED** - Not recommended for new implementations

#### Deprecation Reason
The LMS implementation was designed for parameter estimation rather than state estimation, making it incompatible with the current Kalman-focused architecture.

<augment_code_snippet path="Technique/Lms.m" mode="EXCERPT">
````matlab
classdef Lms < Agent_technique
    properties
        H                   % matrix de pesos atual
        mu                  % passo adaptativo
        iteracts            % número de iterações
        epsilon             % termo de regularização
    end
````
</augment_code_snippet>

#### Legacy Usage (Not Recommended)
- Originally used for adaptive parameter estimation
- Required observation and state buffers
- Implemented steepest descent optimization

### Wiener Filter

**Status**: ⚠️ **DEPRECATED** - Not recommended for new implementations

#### Deprecation Reason
The Wiener filter implementation was designed for batch processing with windowed data, which doesn't align with the real-time state estimation requirements of the current system.

<augment_code_snippet path="Technique/Wiener.m" mode="EXCERPT">
````matlab
classdef Wiener < Agent_technique
    properties
        R_hat               % estimado a partir do buffer de estados
        p_hat               % estimado a partir do buffer de estados e de observações
        H                   % matrix de pesos atual
        n_win               % tamanho da janela de observação
    end
````
</augment_code_snippet>

#### Legacy Usage (Not Recommended)
- Used windowed estimation approach
- Required buffer management for observations and states
- Implemented batch correlation matrix estimation

## Implementation Architecture

### Abstract Base Class
All agent techniques inherit from the `Agent_technique` abstract class:

<augment_code_snippet path="Technique/Agent_technique.m" mode="EXCERPT">
````matlab
classdef Agent_technique < handle
    properties
        x_dim               % dimensão do estado
        y_dim               % dimensão da observação
        xp_hat              % State prior estimate
        xa_hat              % State posterior estimate
    end
    methods (Abstract)
        [varargout] = apply(obj, varargin)
        obj = reset(obj);
        obj = update_params(obj, varargin);
        obj = get_params(obj, varargin);
    end
````
</augment_code_snippet>

### Key Methods
- **`apply()`**: Main filtering operation, processes new measurements
- **`reset()`**: Reinitializes the filter state
- **`update_params()`**: Updates filter parameters dynamically
- **`get_params()`**: Retrieves current filter state and parameters

## Usage Patterns

### Standard Workflow
1. **Create System Model**: Define state transition and observation models
2. **Initialize Technique**: Create KF or RLS instance with appropriate parameters
3. **Create Agent**: Instantiate Agent with the chosen technique
4. **Process Measurements**: Call agent's `self_learning_step()` for each observation
5. **Retrieve Estimates**: Access state estimates through agent properties

### Parameter Configuration
- **State Dimension**: Set `x_dim` based on problem requirements
- **Observation Dimension**: Set `y_dim` based on sensor configuration
- **Noise Parameters**: Configure R matrix (KF) or lambda (RLS) based on noise characteristics
- **Initial Conditions**: Set initial state estimates and covariances

## Testing and Validation

### Recommended Test Scripts
- **`test_kalman.m`**: Comprehensive KF testing with various scenarios
- **`test_rls_vs_kalman.m`**: Comparative analysis between RLS and KF
- **`test_agent.m`**: Agent-level testing with KF integration
- **`test_MAS_sim.m`**: Full multiagent system testing

### Running KF Tests
```matlab
% Select test case (1, 2, 3, or 4)
sim = 4;
% Test Kalman Filter implementation
run('test_kalman.m');
```

### Running RLS Tests
```matlab
% Select test case (1, 2, 3, or 4)
sim = 4;
% Test RLS implementation
run('test_rls.m');
```
### Running Both Tests
```matlab
% Test RLS implementation and compare with KF
run('test_rls_vs_kalman.m');
```

### Validation Criteria
- **Convergence**: Estimates should converge to true state values
- **Consistency**: Covariance matrices should reflect actual estimation errors
- **Stability**: Filters should remain stable under various noise conditions
- **Performance**: Compare MSE performance between different techniques

---

**Note**: For fusion technique documentation, see [README_FUSION_TECH.md](README_FUSION_TECH.md).

*Always use KF or RLS for new implementations. LMS and Wiener are maintained for legacy compatibility only.*
