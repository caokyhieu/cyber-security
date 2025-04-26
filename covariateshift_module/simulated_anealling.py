import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt

class SimulatedAnnealingJAX:
    def __init__(self, obj_func, bounds, perturb_func, 
                 max_iter=1000, initial_temp=100.0, alpha=0.99, temp_min=1e-3):
        self.obj_func = obj_func
        self.bounds = bounds
        self.perturb_func = perturb_func
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.temp_min = temp_min

    def optimize(self, key):
        # Split key for initial position generation
        key, subkey = jax.random.split(key)
        current_position = jax.random.uniform(
            subkey, minval=self.bounds[0], maxval=self.bounds[1],
            shape=self.bounds[0].shape
        )
        current_value = self.obj_func(current_position)
        best_position, best_value = current_position, current_value
        temperature = self.initial_temp

        # Preallocate arrays to track progress
        positions = jnp.zeros((self.max_iter + 1,) + self.bounds[0].shape)
        values = jnp.zeros(self.max_iter + 1)
        positions = positions.at[0].set(current_position)
        values = values.at[0].set(current_value)

        # Initial state carry for lax.scan
        carry_init = (
            current_position,
            current_value,
            best_position,
            best_value,
            temperature,
            key,
            positions,
            values,
        )

        # Run optimization loop with lax.scan
        carry_final, _ = lax.scan(
            self._step_function, 
            carry_init, 
            jnp.arange(self.max_iter)
        )

        # Extract final results
        (_, _, best_position, best_value, _, _, positions, values) = carry_final
        return best_position, best_value, positions, values

    def _step_function(self, carry, iteration):
        (current_pos, current_val, best_pos, best_val, 
         temp, key, positions, values) = carry

        # Split keys for perturbation and acceptance
        key, perturb_key, accept_key = jax.random.split(key, 3)

        # Generate candidate solution
        candidate_pos = self.perturb_func(
            current_pos, self.bounds, iteration, self.max_iter, perturb_key
        )
        candidate_val = self.obj_func(candidate_pos)

        # Calculate acceptance probability
        delta = candidate_val - current_val
        accept_prob = jnp.exp(-delta / temp)
        accept = (candidate_val < current_val) | (jax.random.uniform(accept_key) < accept_prob)

        # Update current state
        new_current_pos = jnp.where(accept, candidate_pos, current_pos)
        new_current_val = jnp.where(accept, candidate_val, current_val)

        # Update best solution
        new_best_pos = jnp.where(new_current_val < best_val, new_current_pos, best_pos)
        new_best_val = jnp.minimum(new_current_val, best_val)

        # Cool down temperature
        new_temp = temp * self.alpha

        # Update tracking arrays
        positions = positions.at[iteration + 1].set(new_current_pos)
        values = values.at[iteration + 1].set(new_current_val)

        # New carry state
        new_carry = (
            new_current_pos,
            new_current_val,
            new_best_pos,
            new_best_val,
            new_temp,
            key,
            positions,
            values,
        )
        return new_carry, None

    def plot_progress(self, positions, values):
        """Plot optimization progress."""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(positions, marker='o', linestyle='-', alpha=0.6)
        plt.xlabel("Iteration")
        plt.ylabel("Position")
        plt.title("Position Progress")

        plt.subplot(1, 2, 2)
        plt.plot(values, marker='o', linestyle='-', color='red', alpha=0.6)
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title("Objective Value Progress")

        plt.tight_layout()
        plt.show()


# JAX-compatible Rosenbrock function
def rosenbrock(x):
    return (1 - x)**2 + 100 * (x**2 - 1)**2

# Adaptive perturbation function with JAX
def adaptive_perturb(x, bounds, iteration, max_iter, key):
    scale = jnp.exp(-iteration / (max_iter / 10))
    step = jax.random.uniform(key, minval=-scale, maxval=scale,shape=bounds[0].shape)
    return jnp.clip(x + step, bounds[0], bounds[1])