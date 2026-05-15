# DRO: Direct Regret Optimization for Bayesian Optimization

Direct Regret Optimization (DRO) is a novel approach for Bayesian Optimization that focuses on explicitly minimizing the final simple regret, rather than using conventional acquisition functions like GP-UCB.

## Method Overview

DRO introduces a fundamentally different perspective on Bayesian Optimization by:

1. Using an **ensemble of Gaussian Processes** with varying hyperparameters to capture model uncertainty
2. Simulating multiple **potential optimization trajectories** within these models
3. Employing **Bayesian early stopping** to make simulations computationally feasible
4. Training a **Decision Transformer** to predict actions that minimize final simple regret
5. Using the transformer's predictions to guide the actual optimization process

This approach combines dense offline training (via simulations) with sparse online learning (through real function evaluations) to achieve improved sample efficiency and non-myopic exploration.

## Installation

The following instructions has been tested on various Apple-silicon platforms and AWS CPU Linux instances. Require Python 3.9.7.

```bash
# (optional) create conda environment
conda create --name DRO python=3.9.7

# Install dependencies (need pip available)
pip install -r requirements.txt  --no-deps
```

## Usage

### Basic Usage

```bash
# here method specify the algorithm test_function specify the benchmark
python main.py method=dro test_function=Ackley

# here we use more efficient dro for easy task lunarlander
python main.py method=dro_simple test_function=LunarLander

# here we use more thorough rollout for ackley 20d
python main.py method=dro_long test_function=Ackley test_function.input_dim=20
```

### Visualize with existing results

After storing the pickle experiment results into corresponding folders, e.g. ackley_dim_study, run corresponding the visualization script generates the desired figures.

### Configuration

The framework uses Hydra for configuration management. Key parameters can be set in `config.yaml` or overridden via command line:

```bash
# Run with default configuration
python main.py

# Override specific parameters
python main.py bo.max_iterations=50 gp.num_models=10
```

## Configuration Options

### GP Model Configuration

- `kernel`: Kernel function type ("rbf", "matern", "rq")
- `noise_constraint`: Minimum noise level
- `lengthscale_min/max`: Range of lengthscales for the GP ensemble
- `num_models`: Number of GPs in the ensemble

### Acquisition Function (for simulation)

- `function`: Acquisition function type ("ei", "ucb", "pi")
- `kappa`: Exploration parameter for UCB
- `xi`: Exploration parameter for EI and PI
- `early_stop_threshold`: Threshold for Bayesian early stopping

### Decision Transformer

- `hidden_size`: Size of transformer hidden layers
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `dropout`: Dropout rate
- `lr`: Learning rate
- `weight_decay`: Weight decay for regularization
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `max_seq_length`: Maximum sequence length for transformer

### Simulation Settings

- `num_rollouts`: Number of simulated trajectories
- `max_rollout_length`: Maximum length of each trajectory
- `early_stop`: Whether to use Bayesian early stopping

### Bayesian Optimization Settings

- `max_iterations`: Maximum number of BO iterations
- `input_dim`: Dimensionality of the search space
- `domain_min/max`: Domain boundaries
- `initial_points`: Number of initial random points
- `objective`: "maximize" or "minimize"

## Algorithm

DRO works in the following steps:

1. **Initialize** with a set of random points
2. For each iteration:
   - **Update GP ensemble** with all observed data
   - **Simulate trajectories** using conventional acquisition functions
   - **Train decision transformer** to predict actions that minimize final regret
   - **Propose next candidate** using the trained transformer
   - **Evaluate** the objective function at the proposed point
3. Return the best observed point after all iterations

## Key Components

### GP Ensemble

Multiple Gaussian Process models with different hyperparameters are maintained to capture epistemic uncertainty.

### Within-Model Sampling

Simulated trajectories are generated using standard acquisition functions within each GP model.

### Bayesian Early Stopping

Trajectories are terminated when the expected improvement falls below a threshold to maintain computational efficiency.

### Decision Transformer

A transformer architecture that learns to predict points that will lead to low final regret, trained on simulated trajectories.

