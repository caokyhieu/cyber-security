import random
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

def gaussian_mutation(key, offspring, mutation_rate, mutation_std):
    mutation_mask = random.uniform(key, offspring.shape) < mutation_rate
    mutation_values = random.normal(key, offspring.shape) * mutation_std
    mutated_offspring = jnp.where(mutation_mask, offspring + mutation_values, offspring)
    return mutated_offspring

def uniform_mutation(key, offspring, mutation_rate, mutation_range):
    mutation_mask = random.uniform(key, offspring.shape) < mutation_rate
    mutation_values = random.uniform(key, offspring.shape, minval=-mutation_range, maxval=mutation_range)
    mutated_offspring = jnp.where(mutation_mask, offspring + mutation_values, offspring)
    return mutated_offspring

def boundary_mutation(key, offspring, mutation_rate, lower_bound, upper_bound):
    mutation_mask = random.uniform(key, offspring.shape) < mutation_rate
    mutation_values = random.uniform(key, offspring.shape, minval=lower_bound, maxval=upper_bound)
    mutated_offspring = jnp.where(mutation_mask, mutation_values, offspring)
    return mutated_offspring

def non_uniform_mutation(key, offspring, mutation_rate, mutation_std):
    mutation_mask = random.uniform(key, offspring.shape) < mutation_rate
    generation = 0  # Replace with actual generation number
    mutation_values = random.normal(key, offspring.shape) * (mutation_std / (generation + 1))
    mutated_offspring = jnp.where(mutation_mask, offspring + mutation_values, offspring)
    return mutated_offspring

class GeneticAlgorithm:

    def __init__(self, population_size, solution_dim, num_generations,
                 mutation_rate, num_parents, fitness_func,
                 rng_key=123, tournament_size=5,early_stopping=10,
                 mutation_std=0.5):
        self.population_size = population_size
        self.solution_dim = solution_dim
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.num_parents = num_parents
        self.fitness_func = fitness_func
        self.key = random.PRNGKey(rng_key)
        self.tournament_size = tournament_size
        self.early_stopping = early_stopping
        self.mutation_std = mutation_std

    def adapt_mutation(self, fitness_scores, generation):
        # Adjust mutation parameters based on population performance
        if generation > 0:
            # Calculate the average fitness
            average_fitness = jnp.mean(fitness_scores)
            
            # Update mutation rate and std deviation based on fitness
            self.mutation_rate *= (1.0 + jnp.sign(average_fitness - 0.5) * 0.1)
            self.mutation_std *= (1.0 + jnp.sign(average_fitness - 0.5) * 0.1)
            # You can adjust the factors 0.1 as needed
            
            # Limit mutation rate and std deviation to reasonable values
            self.mutation_rate = jnp.clip(self.mutation_rate, 0.01, 0.5)
            self.mutation_std = jnp.clip(self.mutation_std, 0.01, 0.5)

    def fitness(self, individual):
        return self.fitness_func(individual)

    def run(self, verbose=True):
        # Initialize the population with random individuals
        population = random.normal(self.key, (self.population_size, self.solution_dim))
        ## init best fitness score a big number
        best_fitness_val = 1e6
        def select_parents(fitness_scores):
            tournaments = random.choice(self.key,self.population_size, (self.num_parents, self.tournament_size))
            # print(fitness_scores[tournaments].shape)
            tournament_index = jnp.argmin(fitness_scores[tournaments], axis=1) + jnp.arange(0,self.num_parents * self.tournament_size,self.tournament_size)
            selected_indices = jnp.take(tournaments,tournament_index )
            # print(jnp.take(population, selected_indices, axis=0).shape)
            return population[selected_indices]
        # Main genetic algorithm loop
        for generation in range(self.num_generations):
            # Calculate the fitness of each individual in the population
            fitness_scores = jnp.array([self.fitness(individual) for individual in population])

            # Select parents using tournament selection
            parents = select_parents(fitness_scores)

            # Create the next generation using crossover (two-point crossover)
            crossover_points = random.randint(self.key, (self.population_size - self.num_parents, 2), 1, self.solution_dim - 1)
            offspring = []
            for i in range(self.population_size - self.num_parents):
                parent1, parent2 = random.choice(self.key, parents, shape=(2,), replace=False)
                crossover_point1, crossover_point2 = crossover_points[i]
                if self.solution_dim>2:
                    child = jnp.hstack((parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
                else:
                    child = jnp.hstack((parent1[:crossover_point1], parent2[crossover_point1:]))
                offspring.append(child)

            offspring = jnp.array(offspring)
            # Apply mutation to the offspring
            mutated_offspring = gaussian_mutation(self.key, offspring, self.mutation_rate, self.mutation_std)

            # Replace the old population with the new generation
            population = population.at[:self.num_parents, :].set(parents)
            population = population.at[self.num_parents:, :].set(mutated_offspring)

            # Adapt mutation parameters based on population performance
            self.adapt_mutation(fitness_scores, generation)

            ## check early stopping
            best_fitness = min(fitness_scores)
            if best_fitness < best_fitness_val:
                best_fitness_val = best_fitness
                early_stopping = 0
            else:
                early_stopping += 1
            
            if early_stopping > self.early_stopping:
                break
            
            # Print the best fitness in this generation
            if verbose:
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # Print the best solution found
        best_fitness_scores = jnp.array([self.fitness(individual) for individual in population])
        best_solution = population[jnp.argmin(best_fitness_scores)]
        if verbose:
            print("Best Solution:", best_solution)
        return best_solution, best_fitness, generation
    


