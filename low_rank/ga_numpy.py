import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

class NumpyGeneticAlgorithm:

    def __init__(self, population_size, solution_dim, num_generations,
                 mutation_rate, num_parents, fitness_func,
                 rng_seed=123, tournament_size=5, early_stopping=10):
        self.population_size = population_size
        self.solution_dim = solution_dim
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.num_parents = num_parents
        self.fitness_func = fitness_func
        self.rng = default_rng(rng_seed)
        self.tournament_size = tournament_size
        self.early_stopping = early_stopping

    def fitness(self, individual):
        return self.fitness_func(individual)

    def run(self, verbose=True):
        # Initialize the population with random individuals
        population = self.rng.standard_normal((self.population_size, self.solution_dim))
        best_fitness_val = 1e6

        def select_parents(fitness_scores):
            tournaments = self.rng.choice(self.population_size, (self.num_parents, self.tournament_size))
            tournament_index = np.argmin(fitness_scores[tournaments], axis=1) + np.arange(0, self.num_parents * self.tournament_size, self.tournament_size)
            selected_indices = np.take(tournaments, tournament_index)
            return population[selected_indices]

        # Main genetic algorithm loop
        for generation in tqdm(range(self.num_generations)):
            # Calculate the fitness of each individual in the population
            fitness_scores = np.array([self.fitness(individual) for individual in population])

            # Select parents using tournament selection
            parents = select_parents(fitness_scores)

            # Create the next generation using crossover (two-point crossover)
            crossover_points = self.rng.integers(1, self.solution_dim , size=(self.population_size - self.num_parents, 2))

            offspring = []
            for i in range(self.population_size - self.num_parents):
                parent1, parent2 = self.rng.choice(parents, size=(2,), replace=False)
                crossover_point1, crossover_point2 = min(crossover_points[i]),max(crossover_points[i])
                if self.solution_dim > 2:
                    child = np.hstack((parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
                else:
                    child = np.hstack((parent1[:crossover_point1], parent2[crossover_point1:]))
                offspring.append(child)

            offspring = np.array(offspring)

            # Apply mutation to the offspring
            mutation_mask = self.rng.random((self.population_size - self.num_parents, self.solution_dim)) < self.mutation_rate
            mutated_offspring = np.where(mutation_mask, self.rng.standard_normal((self.population_size - self.num_parents, self.solution_dim)) + offspring, offspring)

            # Replace the old population with the new generation
            population[:self.num_parents, :] = parents
            population[self.num_parents:, :] = mutated_offspring

            # Check early stopping
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
        best_fitness_scores = np.array([self.fitness(individual) for individual in population])
        best_solution = population[np.argmin(best_fitness_scores)]
        if verbose:
            print("Best Solution:", best_solution)
        return best_solution, best_fitness, generation