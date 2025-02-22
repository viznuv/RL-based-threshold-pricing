# Threshold Pricing Agent with Q-Learning

This repository contains a simple implementation of a threshold pricing agent using the Q-learning algorithm.  The agent learns to set optimal threshold prices in a simulated environment to maximize rewards.

## Overview

The core of this project is a custom OpenAI Gym environment (`ThresholdPricingEnv`) that simulates a single-product pricing scenario.  A customer has a "willingness to pay" (WTP), drawn from a uniform distribution. The agent sets a threshold price.  If the customer's WTP is greater than or equal to the threshold price, the agent receives the threshold price as a reward; otherwise, the reward is zero. The agent's goal is to learn the threshold prices that maximize its cumulative reward over time, given a fluctuating market demand (represented by the environment's state).

The agent uses Q-learning to learn the optimal pricing strategy.  The state represents market demand, and the actions are discrete threshold prices (1 to 10).

## Environment (`ThresholdPricingEnv`)

The `ThresholdPricingEnv` class defines the environment:

*   **Action Space:**  Discrete(10).  Actions 0 through 9 correspond to threshold prices 1 through 10.
*   **Observation Space:** Box(low=0, high=100, shape=(1,), dtype=np.float32). Represents the market demand (a single integer value).  In this implementation, the demand is randomly generated within the range [10, 50] at each step.
*   **`reset()`:** Initializes the environment's state (market demand) to a random value between 10 and 50. Returns the initial state.
*   **`step(action)`:**  Takes an action (threshold price index), simulates a customer's WTP, calculates the reward, updates the environment's state (new random demand), and returns the next state, reward, a `done` flag (always `False` in this continuous environment), and an empty info dictionary.
*    **Reward:** If customer WTP >= threshold price then reward is threshold price else, no reward.

## Q-Learning Implementation

The script implements a basic Q-learning algorithm:

1.  **Initialization:** A Q-table is initialized to zeros.  The Q-table has dimensions (number of possible states) x (number of possible actions). The state is discretized for use as an index into the Q-table.
2.  **Hyperparameters:**
    *   `alpha`: Learning rate (0.1)
    *   `gamma`: Discount factor (0.95)
    *   `epsilon`: Exploration rate (0.1)
    *   `episodes`: Number of training episodes (500)
    *   `steps_per_episode`: Maximum steps per episode (100)
3.  **Training Loop:**
    *   The environment is reset at the beginning of each episode.
    *   For each step in the episode:
        *   An action is selected using an epsilon-greedy policy (explore with probability `epsilon`, otherwise choose the action with the highest Q-value).
        *   The `step()` function of the environment is called with the chosen action.
        *   The Q-value for the current state-action pair is updated using the Q-learning update rule:
            ```
            Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
            ```
            where `s` is the current state, `a` is the current action, `s'` is the next state, and `a'` is the action that maximizes the Q-value in the next state.
        *   The current state is updated to the next state.
        *   The total reward for the episode is accumulated.
    *   The total reward per episode is stored for plotting.
    *   Progress is printed every 50 episodes.
4.  **Visualization:** After training, a plot of the total reward per episode is displayed, showing the learning progress.

## Getting Started

1.  **Prerequisites:**
    *   Python 3.6+
    *   NumPy
    *   Gym (`pip install gym`)
    *   Matplotlib

2.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    Replace `<repository_url>` and `<repository_directory>` with your repository details.

3.  **Run the script:**

    ```bash
    python threshold_pricing.py
    ```

    This will train the Q-learning agent and display a plot of the total reward per episode.

## Key Concepts and Improvements

*   **Threshold Pricing:**  This code demonstrates a simplified version of threshold pricing.  Real-world applications might involve more complex demand models, competitor pricing, and dynamic pricing strategies.
*   **Q-Learning:** The core reinforcement learning algorithm. The agent learns a Q-function that estimates the expected cumulative reward for each state-action pair.
*   **Exploration vs. Exploitation:** The epsilon-greedy strategy balances exploration (trying random actions) and exploitation (choosing the action believed to be best).
*   **State Discretization:** The continuous state space (market demand) is discretized into integer values to be used as indices in the Q-table.  This is a common technique when using Q-learning with continuous states.
*   **Continuous vs. Episodic:** The environment is set up as continuous (no `done` signal).  You could modify it to be episodic (e.g., ending after a certain number of steps or when a certain condition is met).

*   **Possible Improvements:**
    *   **More Realistic Demand Model:** Instead of uniform random demand, use a more sophisticated model (e.g., based on historical data, seasonality, or external factors).
    *   **Competitor Pricing:** Introduce competitor agents or a model of competitor pricing.
    *   **Dynamic Pricing:** Explore more advanced dynamic pricing strategies beyond simple threshold pricing.
    *   **Function Approximation:** For larger state spaces, use function approximation (e.g., neural networks) instead of a Q-table.  This would allow for handling truly continuous states without discretization. Libraries like TensorFlow or PyTorch would be useful here.
    *   **Policy Gradient Methods:** Consider alternative reinforcement learning algorithms, such as policy gradient methods (e.g., REINFORCE, A2C, PPO).
    *    **Parameter Tuning:** Tune hyperparameters.
    *   **Add Tests:** Add unit tests for the environment and Q-learning algorithm.
    *   **Varying Customer WTP distribution:** The customer WTP is currently drawn from U(1,10), you can change that.

This README provides a comprehensive overview of the code, explains how to run it, and suggests potential improvements.  It is suitable for use on GitHub to document the project.
