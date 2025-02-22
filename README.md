# Dynamic Pricing Agent with REINFORCE

This repository contains an implementation of a dynamic pricing agent using the REINFORCE algorithm (a policy gradient method) in a simulated environment with seasonality and competitor pricing.  The agent learns to set optimal prices to maximize revenue.

## Overview

This project features a custom OpenAI Gym environment (`RL-based-threshold-pricing`) that simulates a more realistic dynamic pricing scenario than the previous example.  The environment incorporates:

*   **Seasonality:** Customer willingness-to-pay (WTP) fluctuates according to a sinusoidal pattern, simulating seasonal demand changes.
*   **Competitor Pricing:** A competitor also sets a dynamic price, influencing the agent's probability of making a sale.  The competitor's price also varies with seasonality and includes some random noise.
*   **Continuous Action Space:** The agent chooses a price from a continuous range (0 to 20).
*   **Continuous State Space:** The state includes the normalized time, a seasonality factor and the competitor's price.

The agent uses a neural network (implemented with PyTorch) to represent its policy, which is a Gaussian distribution over prices.  The REINFORCE algorithm is used to train the policy to maximize cumulative rewards.

## Environment (`DynamicPricingEnv`)

The `DynamicPricingEnv` class defines the environment:

*   **Action Space:** `Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32)`.  The agent's action is a continuous price between 0 and 20.
*   **Observation Space:** `Box(low=[0.0, -1.0, 0.0], high=[1.0, 1.0, 20.0], shape=(3,), dtype=np.float32)`. The state is a 3-dimensional vector:
    *   `normalized_time`:  The current time step normalized to the range [0, 1].
    *   `seasonal_factor`:  A value between -1 and 1 representing the seasonal influence on WTP.
    *   `competitor_price`: The current price set by the competitor.
*   **`reset()`:** Resets the environment to the initial state (time step 0).  Calculates the initial seasonal factor and competitor price.
*   **`step(action)`:**
    *   Takes the agent's chosen price (a continuous value).
    *   Calculates the customer's WTP, considering the seasonal factor.
    *   Determines the probability of a sale based on the agent's price, the customer's WTP, and the competitor's price.  A lower price relative to the competitor increases the sale probability.
    *   Calculates the reward (the agent's price if a sale occurs, 0 otherwise).
    *   Updates the environment's state (advances time, calculates the new seasonal factor and competitor price).
    *   Returns the next state, reward, `done` flag (True when the episode reaches `max_steps`), and an empty info dictionary.
*   **`_get_seasonality(step)`:** Calculates the seasonal factor using a sine function.
*   **`_get_competitor_price(step)`:**  Calculates the competitor's price, which includes a base price, a seasonal component, and random noise.

## Policy Network (`PolicyNetwork`)

The `PolicyNetwork` class (a PyTorch `nn.Module`) defines the agent's policy:

*   **Input:**  A state vector (3-dimensional).
*   **Output:**  A `torch.distributions.Normal` object representing a Gaussian distribution over prices.  The network outputs the mean and (log) standard deviation of this distribution.
*   **Architecture:**  A simple feedforward network with two hidden layers (ReLU activations) and two output heads: one for the mean and one for the log standard deviation. The log standard deviation is a learnable parameter.
*    **Forward Pass:** The `forward` method takes a state tensor as input and returns the normal distribution.

## REINFORCE Algorithm

The `train()` function implements the REINFORCE algorithm:

1.  **Initialization:**  Creates the environment, policy network, and optimizer (Adam).
2.  **Training Loop (Episodes):**
    *   **Rollout:** For each episode:
        *   Reset the environment.
        *   Collect a trajectory (sequence of states, actions, log probabilities, and rewards) by interacting with the environment. The policy network is used to sample actions from the Gaussian distribution defined by its output.
        *   Calculate the discounted cumulative rewards (returns) for each time step.
    *   **Policy Update:**
        *   Calculate the policy loss.  The loss is the negative sum of the log probabilities of the taken actions, weighted by the corresponding discounted returns.
        *   Perform backpropagation to compute the gradients of the loss with respect to the policy network's parameters.
        *   Update the policy network's parameters using the optimizer.
    *   **Logging:** Store and print the total reward for each episode.
3.  **Visualization:** After training, plot the total reward per episode.

## Getting Started

1.  **Prerequisites:**
    *   Python 3.6+
    *   NumPy
    *   Gym (`pip install gym`)
    *   PyTorch (`pip install torch`)
    *   Matplotlib
    *   Unittest

2.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    Replace `<repository_url>` and `<repository_directory>` with your repository details.

3.  **Run the script:**

    ```bash
    python RL-based-threshold-pricing.py
    ```

    This will train the REINFORCE agent and display a plot of the total reward per episode.  You can run the unit tests by uncommenting the `unittest.main()` lines.

## Key Concepts and Improvements

*   **Dynamic Pricing:** This code demonstrates a more complex dynamic pricing scenario, including seasonality and competitor pricing.
*   **REINFORCE (Policy Gradient):**  A policy gradient method that directly optimizes the policy (represented by the neural network) by estimating the gradient of the expected cumulative reward.
*   **Gaussian Policy:**  The policy is a Gaussian distribution, allowing for continuous actions.
*   **Competitor Model:**  The competitor's pricing strategy adds another layer of complexity to the environment.
*   **Seasonality:** The sinusoidal seasonality factor adds a realistic dynamic to the customer's WTP.

*   **Possible Improvements:**
    *   **More Realistic Models:** Use more sophisticated models for customer WTP and competitor behavior (e.g., incorporating historical data, price elasticity, or different customer segments).
    *   **Advanced Policy Gradient Methods:**  Experiment with more advanced policy gradient algorithms like A2C, PPO, or DDPG, which often offer better sample efficiency and stability than REINFORCE.
    *   **Hyperparameter Tuning:**  Systematically tune hyperparameters (learning rate, discount factor, network architecture, etc.) using techniques like grid search or Bayesian optimization.
    *   **Exploration Strategies:** Implement more sophisticated exploration strategies (e.g., entropy regularization) to encourage the agent to explore a wider range of actions.
    *   **Recurrent Neural Networks (RNNs):** Consider using RNNs (like LSTMs) in the policy network to handle potential temporal dependencies in the state (e.g., if past competitor prices are important).
    * **Batching** Improve training by processing episodes in batches.

This comprehensive README explains the code, its functionality, how to run it, and potential improvements. It's suitable for a GitHub repository. The unit tests included provide a basic level of code verification.
