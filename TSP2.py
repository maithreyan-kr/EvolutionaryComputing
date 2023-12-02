import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import time

# function to calculate distance between two points
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# objective function for TSP
def total_distance(path, cities):
    dist = 0
    for i in range(len(path) - 1):
        dist += distance(cities[path[i]], cities[path[i + 1]])
    dist += distance(cities[path[-1]], cities[path[0]])  # Return to the starting city
    return dist,

# crossover function using Partially Mapped Crossover(PMX)
def custom_crossover(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(0, size - 1)
    cxpoint2 = random.randint(0, size - 1)
    while cxpoint2 == cxpoint1:
        cxpoint2 = random.randint(0, size - 1)
    if cxpoint1 > cxpoint2:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # applying PMX crossover
    for i in range(cxpoint1, cxpoint2):
        temp1, temp2 = ind1[i], ind2[i]
        ind1[i], ind2[i] = temp2, temp1
        ind1[indices[ind1.index(temp2)]], ind2[indices[ind2.index(temp1)]] = temp2, temp1

    return ind1, ind2

# Custom mutation function using Swap mutation
def custom_mutation(ind):
    size = len(ind)
    mutpoint1 = random.randint(0, size - 1)
    mutpoint2 = random.randint(0, size - 1)
    while mutpoint2 == mutpoint1:
        mutpoint2 = random.randint(0, size - 1)
    ind[mutpoint1], ind[mutpoint2] = ind[mutpoint2], ind[mutpoint1]
    return ind,

# TSP problem class for DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# read TSP file
def read_tsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cities = []
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            city_id = int(parts[0])
            x_coord = float(parts[1])
            y_coord = float(parts[2])
            cities.append((x_coord, y_coord))

    return cities

# CHC genetic algorithm function
def chc_genetic_algorithm(cities, population_size, generations, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    start_time = time.time()
    population = toolbox.population(n=population_size)
    CXPB, MUTPB = 1.2, 0.43

    # evaluate entire population
    fitnesses = list(map(lambda ind: total_distance(ind, cities), population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    avg_objectives = []
    max_objectives = []

    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)

        # evaluate offspring
        fitnesses = list(map(lambda ind: total_distance(ind, cities), offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # select next generation individuals
        population = tools.selBest(population + offspring, population_size)

        # record statistics
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        avg_objectives.append(mean)
        max_objectives.append(min(fits))

        print(f"Generation {gen + 1}: Avg {mean}, Max {min(fits)}")

    best_ind = tools.selBest(population, 1)[0]
    best_path = list(best_ind)
    
    end_time = time.time()  # record end time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")

    return best_path, avg_objectives, max_objectives

if __name__ == "__main__":
    tsp_file_path = '/Users/cex/Downloads/eil76.tsp'
    chromosome_length = 24 
    
    # read TSP file
    cities = read_tsp_file(tsp_file_path)
    num_cities = len(cities)

    population_size = 200
    generations = 375
    num_runs = 30  

    # DEAP initialization
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_cities), num_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selBest)

    avg_objectives_list = []
    max_objectives_list = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        best_path, avg_objectives, max_objectives = chc_genetic_algorithm(cities, population_size, generations, random_seed=run)

        best_fitness = total_distance(best_path, cities)
        print(f"Best Result: Path {best_path}, Fitness {best_fitness}")

        avg_objectives_list.append(avg_objectives)
        max_objectives_list.append(max_objectives)

    # Plot average objective values over multiple runs
    avg_objectives_array = np.array(avg_objectives_list)
    avg_objectives_mean = np.mean(avg_objectives_array, axis=0)
    avg_objectives_std = np.std(avg_objectives_array, axis=0)

    plt.errorbar(range(1, generations + 1), avg_objectives_mean, yerr=avg_objectives_std, label='Average Objective')
    plt.xlabel('Generation')
    plt.ylabel('Average Objective Value')
    plt.legend()
    plt.show()

    # Plot maximum average objective values over multiple runs
    max_objectives_array = np.array(max_objectives_list)
    max_objectives_mean = np.mean(max_objectives_array, axis=0)
    max_objectives_std = np.std(max_objectives_array, axis=0)

    plt.errorbar(range(1, generations + 1), max_objectives_mean, yerr=max_objectives_std, label='Max Objective')
    plt.xlabel('Generation')
    plt.ylabel('Max Average Objective Value')
    plt.legend()
    plt.show()
