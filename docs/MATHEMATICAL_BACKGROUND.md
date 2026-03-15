# Mathematical Foundations of Alignment

## 1. Direct Preference Optimization (DPO)
Unlike traditional RLHF which requires training a separate Reward Model (RM) and optimizing via PPO, DPO optimizes the policy model directly on preference data.

Given a dataset of preferences $D = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}$, where $y_w$ is the chosen response and $y_l$ is the rejected response.

The DPO loss objective is derived by re-parameterizing the reward model in terms of the optimal policy:

$$ L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right] $$

Where:
*   $\pi_\theta$ is the active policy (the model being trained).
*   $\pi_{ref}$ is the frozen reference policy.
*   $\beta$ controls the deviation from the reference policy (KL-divergence penalty).
*   $\sigma$ is the sigmoid function.

This mathematically guarantees that increasing the probability of the chosen response relative to the rejected response implicitly maximizes the reward.
