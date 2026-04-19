# Role: Implementing Horizon-Aware IQL (Time-to-Go)

**Objective:**
You are an expert AI assistant specializing in Causal Inference and Offline Reinforcement Learning. Your current task is to upgrade the Implicit Q-Learning (IQL) planner to be a **Horizon-Aware Goal-Conditioned RL** model. Currently, the IQL networks (Actor, Critic, Value) only receive the CT-encoded latent state $Z_t$. To enable dynamic, time-aware treatment planning (e.g., interventions based on the deadline), the networks must explicitly receive both the **Target (e.g., future tumor volume)** and the **Time-to-Go ($\Delta t$)** as part of their state observation space.

**Mathematical Formulation:**
* Old State: $S_t = Z_t$
* New State: $S_t = \text{Concat}(Z_t, \text{Target}, \Delta t_{\text{norm}})$
* New Next State: $S_{t+1} = \text{Concat}(Z_{t+1}, \text{Target}, \Delta t_{\text{next\_norm}})$
* where $\Delta t_{\text{next\_norm}} = \Delta t_{\text{norm}} - \frac{1}{\text{MAX\_TAU}}$

Please follow this multi-phase execution plan to safely refactor the codebase.

---

### Phase 1: Modify Dataset Builder (`iql_dataset_builder.py` / Transition Generator)
**Goal:** Implement Hindsight Experience Replay (HER) logic to sample future targets, calculate $\Delta t$, and construct augmented state vectors.

1.  **Define Configuration:** Introduce a normalization constant `max_tau` (e.g., `12.0`) from the config (`exp.max_tau` or fallback to 12.0). Neural networks require scaled inputs.
2.  **Target Sampling Logic (Windowed HER):** When building transitions `(s, a, r, s', done)` for a valid time step `t` within a patient's trajectory:
    * Define the maximum observable horizon: `max_future_step = min(t + max_tau, T_valid)`
    * Ensure there is at least one future step available. If `t >= T_valid - 1`, skip or handle as terminal.
    * Randomly sample a future time step `t_target` such that `t < t_target <= max_future_step`.
    * Extract the future outcome `Y_target` at `t_target`.
    * *Self-Correction Check:* This guarantees that `delta_t` will NEVER exceed `max_tau`, naturally bounding `delta_t_norm` between `(0, 1.0]`.
3.  **Time-to-Go Calculation:**
    * Calculate current remaining time: `delta_t = t_target - t`
    * Normalize it: `delta_t_norm = delta_t / max_tau`
    * Calculate next step's remaining time: `delta_t_next_norm = (delta_t - 1) / max_tau`
4.  **State Augmentation (Concatenation):**
    * `state_t` = `torch.cat([Z_t, Y_target, delta_t_norm], dim=-1)`
    * `state_next` = `torch.cat([Z_{t+1}, Y_target, delta_t_next_norm], dim=-1)`
5.  **Reward Function Update:** Ensure the reward is correctly calculated using the *same* target: e.g., $r_t = - \text{MSE}(Y_{t+1}, Y_{target})$.

---

### Phase 2: Modify Model Architecture (`train_iql_planner.py` / `iql_planner.py`)
**Goal:** Expand the input dimensionality of all MLPs in the IQL algorithm to accept the augmented state.

1.  **Calculate New State Dimension:**
    * Let $D_z$ be the dimension of the CT history encoder output ($Z_t$).
    * Let $D_{target}$ be the dimension of the output target (usually same as $Y_{dim}$).
    * Let $1$ be the dimension of $\Delta t$.
    * `new_state_dim = z_dim + output_dim + 1`
2.  **Update Networks:** Update the `__init__` methods of the `Actor`, `QNetwork` (Critic), and `VNetwork` (Value) to use `new_state_dim` instead of `z_dim`.
    * *Self-Correction Check:* Ensure `state_dim` is passed correctly during the instantiation of these models in the main script. Do not hardcode dimensions; infer them from configs.

---

### Phase 3: Modify Inference / Rollout (`eval_iql_planner.py`)
**Goal:** Implement the countdown mechanism during the autoregressive evaluation phase.

1.  **Initialize Target and $\Delta t$:** For a given evaluation trajectory where the user requests a simulation of length `eval_tau` towards a specific `eval_target`:
    * The total steps to run is `eval_tau`.
2.  **Autoregressive Loop Updates:** Inside the `for step in range(eval_tau):` loop:
    * Calculate remaining steps: `steps_left = eval_tau - step`
    * Normalize it: `delta_t_norm = torch.tensor([[steps_left / max_tau]], dtype=torch.float32, device=device)`
    * Make sure dimensions match the batch size: `delta_t_norm = delta_t_norm.repeat(batch_size, 1)`
    * Construct the current augmented state for the Actor:
        `augmented_state = torch.cat([current_Z, eval_target, delta_t_norm], dim=-1)`
    * Pass `augmented_state` to the Actor to get `action`.
    * Pass `current_Z` and `action` to the CT World Model to get `next_Z` and `predicted_Y`.
    * Continue loop with `current_Z = next_Z`.

---

### ⚠️ Critical Constraints for Cursor
* **Tensor Shapes:** Pay strict attention to tensor dimensions (e.g., `.unsqueeze(-1)`) when concatenating a scalar like `delta_t` to batched vectors like $Z_t$ (shape `[Batch, Z_dim]`). `delta_t` must be reshaped to `[Batch, 1]`.
* **Do Not Break CT Modularity:** The CT encoding function and `ct_model.step()` MUST NOT be modified to expect `delta_t`. The CT is the world model; physical physics don't care about the Actor's deadline. Only the IQL state representations are augmented.
* **Normalization Consistency:** Ensure the exact same `max_tau` constant is used in `iql_dataset_builder.py` and `eval_iql_planner.py`.