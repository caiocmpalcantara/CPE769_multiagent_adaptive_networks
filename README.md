# CPE769 Multiagent Adaptive Networks

## Table of Contents
- [Project Overview](#project-overview)
- [Codebase Organization](#codebase-organization)
- [Architecture and Design Patterns](#architecture-and-design-patterns)
- [Usage Guide](#usage-guide)
- [Running test_MAS_sim.m](#running-test_mas_simm)
- [Expected Outputs](#expected-outputs)
- [Technical Documentation](#technical-documentation)
- [Getting Started](#getting-started)

## Project Overview

This project implements a comprehensive **multiagent adaptive networks system** for distributed learning and state estimation. The system enables multiple autonomous agents to collaboratively learn and estimate unknown parameters or states through various adaptive filtering techniques and fusion strategies.

### Main Objectives
- **Distributed State Estimation**: Agents cooperatively estimate unknown states using local observations
- **Adaptive Filtering**: Implementation of multiple adaptive algorithms (Kalman Filter, RLS, LMS, Wiener)
- **Social Learning**: Agents share information through various fusion techniques to improve collective performance
- **Network Topology Support**: Flexible network configurations with different connectivity patterns
- **Performance Analysis**: Comprehensive simulation framework for comparing different techniques and topologies

### Key Features
- Multiple agent techniques: **KF (Kalman Filter)**, **RLS (Recursive Least Squares)**, LMS, Wiener
- Advanced fusion strategies: **General_Adapt_and_Fuse**, **Diff_KF_time_measure**, **Diff_KF_info_matrix**
- Configurable network topologies with neighbor-based communication
- Monte Carlo simulation support for statistical analysis
- Real-time performance metrics and visualization

## Codebase Organization

```
CPE769_multiagent_adaptive_networks/
├── Agent/                          # Agent implementations
│   ├── Agent.m                     # Core learning agent class
│   └── Cooperation_type.m          # Cooperation strategy enumeration
├── Technique/                      # Adaptive filtering and fusion techniques
│   ├── Agent_technique.m           # Abstract base for adaptive algorithms
│   ├── KF.m                       # Kalman Filter implementation
│   ├── RLS.m                      # Recursive Least Squares implementation
│   ├── Lms.m                      # Least Mean Squares implementation (deprecated)
│   ├── Wiener.m                   # Wiener filtering implementation (deprecated)
│   ├── Fusion_technique.m         # Abstract base for fusion strategies
│   ├── General_Adapt_and_Fuse.m   # General adaptive fusion technique
│   ├── Diff_KF_time_measure.m     # Diffusion KF with time measure fusion
│   ├── Diff_KF_info_matrix.m      # Diffusion KF with information matrix fusion
│   ├── Noon_coop.m                # Non-cooperative baseline
│   └── Kalman_inc/                # Kalman filter components
│       ├── Linear_State.m         # Linear state model
│       ├── Linear_Obs.m           # Linear observation model
│       └── System_Model.m         # Abstract system model
├── Simulation/                     # Simulation framework
│   ├── Simulation.m               # Abstract simulation base
│   ├── Static_sim.m               # Stationary environment simulation
│   └── Noise.m                    # Noise generation utilities
├── Utils/                         # Utility functions
│   ├── DEBUG.m                    # Debug output utility
│   ├── print_graph.m              # Network topology visualization
│   └── [other utilities]
├── test_MAS_sim.m                 # Main simulation test script
└── [test files and results]
```

### Key Files Description
- **`test_MAS_sim.m`**: Main simulation script for multiagent social learning
- **`Agent.m`**: Core agent class managing individual learning and social fusion
- **`KF.m`** & **`RLS.m`**: Primary agent techniques (currently supported)
- **`General_Adapt_and_Fuse.m`**: Main fusion technique for cooperative learning
- **`Linear_State.m`**: System model for Kalman filtering applications

## Architecture and Design Patterns

### Object-Oriented Design Principles
The codebase follows **SOLID principles** and implements several key design patterns:

#### 1. **Strategy Pattern**
- **`Agent_technique`** abstract class defines common interface for all adaptive filtering algorithms
- Concrete implementations: `KF`, `RLS`, `Lms`, `Wiener`
- Enables runtime algorithm switching and polymorphic behavior

#### 2. **Template Method Pattern**
- **`Fusion_technique`** abstract class defines fusion protocol structure
- Concrete implementations: `General_Adapt_and_Fuse`, `Diff_KF_time_measure`, `Diff_KF_info_matrix`
- Standardizes cooperation strategies while allowing customization

#### 3. **Handle Class Pattern (MATLAB)**
- All major classes inherit from `handle` for reference semantics
- Enables efficient memory management and object sharing between agents
- Critical for maintaining state consistency across the network

#### 4. **Composition Pattern**
- **`Agent`** class composes `Agent_technique` and `Fusion_technique` objects
- Flexible combination of learning algorithms and cooperation strategies
- Supports dependency injection for testing and configuration

### Best OOP Practices Implemented
- **Encapsulation**: Private properties with controlled access methods
- **Abstraction**: Clear separation between interface and implementation
- **Polymorphism**: Unified interfaces for different algorithm types
- **Single Responsibility**: Each class has a focused, well-defined purpose

## Usage Guide

### Prerequisites
- MATLAB R2019b or later
- Signal Processing Toolbox (recommended)
- Statistics and Machine Learning Toolbox (recommended)

### Basic Setup
```matlab
% Add necessary paths
addpath("./Technique/")
addpath("./Agent/")
addpath("./Technique/Kalman_inc/")
addpath("./Utils/")
```

## Running test_MAS_sim.m

The main simulation script `test_MAS_sim.m` requires **three mandatory variables** to be defined in the MATLAB Command Window before execution:

### Required Variables

#### 1. `net_topology` (Network Topology)
Defines the communication network structure between agents.

**Available Options:**
- **`'caio_net_topology'`**: 6-agent network with moderate connectivity
  - Agent connections: [1,2,3], [1,2], [1,3,4], [3,4,5,6], [4,5], [4,6]
- **`'merched_net_topology'`**: 20-agent network with complex connectivity
  - Larger, more realistic network topology for advanced simulations

**Example:**
```matlab
net_topology = 'caio_net_topology';  % or 'merched_net_topology'
```

#### 2. `tech` (Agent Technique)
Specifies the adaptive filtering algorithm used by all agents.

**Available Options:**
- **`'KF'`**: Kalman Filter (recommended, fully supported)
- **`'RLS'`**: Recursive Least Squares (fully supported)

**Example:**
```matlab
tech = 'KF';  % or 'RLS'
```

#### 3. `str_fusion_tech` (Fusion Technique)
Determines how agents share and fuse information from neighbors.

**Available Options:**
- **`'General_Adapt_and_Fuse'`**: General adaptive fusion (works with both KF and RLS)
- **`'Diff_KF_time_measure'`**: Diffusion KF with time measure fusion (KF only)
- **`'Diff_KF_info_matrix'`**: Diffusion KF with information matrix fusion (KF only)
- **`''`** (empty string): Non-cooperative mode (no information sharing)

**Example:**
```matlab
str_fusion_tech = 'General_Adapt_and_Fuse';
```

### Complete Execution Example
```matlab
% Define required variables
net_topology = 'caio_net_topology';
tech = 'KF';
str_fusion_tech = 'General_Adapt_and_Fuse';

% Run the simulation
run('test_MAS_sim.m'); % or simply type 'test_MAS_sim' in the command window
```

### Execution Flow

1. **Initialization Phase**
   - Verifies required classes are available
   - Sets up simulation parameters (state dimension=3, observation dimension=1, N=500 time steps, M>1 realizations for Monte Carlo)
   - Creates network topology and visualizes it

2. **Agent Creation Phase**
   - Creates agents with specified technique (KF or RLS)
   - Configures observation models and noise parameters
   - Sets up fusion techniques and neighbor connections

3. **Simulation Loop**
   - For each time step:
     - **Self-learning step**: Each agent processes its local observation
     - **Social learning step**: Agents fuse information from neighbors
   - Stores individual and fused estimates for analysis

4. **Results Analysis**
   - Computes Mean Square Deviation (MSD) for performance evaluation
   - Generates multiple visualization plots
   - Saves results to log file

## Expected Outputs

### Console Output
- Class verification status
- Simulation parameters summary
- Agent creation confirmation
- Real-time progress updates
- Final performance summary

### Generated Figures
- **Figure 10**: Network topology graph
- **Figure 1**: True dynamic vs. observations over time per agent
- **Figure 2**: Individual agent estimates (before fusion)
- **Figure 3**: Fused estimates (after social learning)
- **Figure 4**: Estimation errors over time
- **Figure 5**: Mean Square Deviation (MSD) analysis
- **Figure 8**: Final MSD performance

### Log File
- Complete simulation log saved to `log.txt`
- Includes all console output and debug information

## Technical Documentation

For detailed technical information, refer to:
- **[README_AGENT_TECH.md](README_AGENT_TECH.md)**: Agent techniques implementation details
- **[README_FUSION_TECH.md](README_FUSION_TECH.md)**: Fusion techniques implementation details

## Getting Started

### Quick Start Example
```matlab
% 1. Set up variables
net_topology = 'caio_net_topology';
tech = 'KF';
str_fusion_tech = 'General_Adapt_and_Fuse';

% 2. Run simulation
run('test_MAS_sim.m'); % or simply type 'test_MAS_sim' in the command window

% 3. Analyze results in generated figures and log.txt
```

### Recommended Configurations
- **Beginners**: `caio_net_topology` + `KF` + `General_Adapt_and_Fuse`
- **Advanced**: `merched_net_topology` + `KF` + `Diff_KF_info_matrix`
- **Comparison**: Run multiple configurations to compare performance

---

*For questions or issues, refer to the technical documentation or examine the test files for additional examples. Any bugs or feature requests are welcome! email: caio.alcantara@alumni.usp.br*
