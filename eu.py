import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING, filename="algo.log", filemode="w", format= "%(message)s")

def load_locations():
    city = np.array([
        [5, 2, 4, 8, 9, 0, 3, 3, 8, 7],
        [5, 5, 3, 4, 4, 6, 4, 1, 9, 1],
        [4, 1, 2, 1, 3, 8, 7, 8, 9, 1],
        [1, 7, 1, 6, 9, 3, 1, 9, 6, 9],
        [4, 7, 4, 9, 9, 8, 6, 5, 4, 2],
        [7, 5, 8, 2, 5, 2, 3, 9, 8, 2],
        [1, 4, 0, 6, 8, 4, 0, 1, 2, 1],
        [1, 5, 2, 1, 2, 8, 3, 3, 6, 2],
        [4, 5, 9, 6, 3, 9, 7, 6, 5, 10],
        [0, 6, 2, 8, 7, 1, 2, 1, 5, 3]
    ])
    return city

def coord_of(pos, city):
    return np.unravel_index(pos, city.shape)

def response_time_of(fire_station, city):
    r = cost_of(fire_station, city)
    return 1.7 + 3.4 * r

def cost_of(proposed, city):
    cost = 0
    for pos in range(city.size):
        if pos != proposed:
            fire_freq = city.flat[pos]
            cost += distance_of(pos, proposed, fire_freq, city)
            logging.info(f"nonproposed:{pos}, proposed: {proposed}, fire_freq: {fire_freq},  cost: {cost}")
    return cost

def distance_of(non_proposed, proposed, fire_freq, city):
    (xn, yn) = coord_of(non_proposed, city)
    logging.info(f"coord: {xn,yn}")
    (xfs, yfs) = coord_of(proposed, city)
    w = fire_freq
    return w * np.sqrt((xn - xfs)**2 + (yn - yfs)**2)

def init_population(population_size, city):
    return np.random.permutation(city.size)

def truncation_selection(population, proportion, city):
    sorted_population = sorted(population, key=lambda loc: cost_of(loc, city))
    N = int(len(population) * proportion)
    return sorted_population[:N], N

def crossover(parents, city, N):
    children = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            parent1, parent2 = parents[i], parents[i + 1]
            logging.warning(f"parent1 : {parent1}, parent2: {parent2}")
            x1, y1 = coord_of(parent1, city)
            x2, y2 = coord_of(parent2, city)
            child = [x1 if np.random.rand() > 0.5 else y1,
                     x2 if np.random.rand() > 0.5 else y2]
            xc, yc = child
            children.append(np.ravel_multi_index((xc, yc), city.shape))
            logging.warning(f"child: {children}")
    return np.array(children)

def mutate(children, mutation_rate, parents, population, N, city):
    for i in range(len(children)):
        if np.random.rand() < mutation_rate:
            xc, yc = coord_of(children[i], city)
            mutated = np.ravel_multi_index((yc, xc), city.shape)
            children[i] = mutated
    new_population = np.concatenate((children, parents, population[N:]))
    sorted_population = sorted(new_population, key=lambda loc: cost_of(loc, city))
    return np.array(sorted_population[:len(population)])

def emergency_response_unit_locator():
    city = load_locations()
    population = init_population(1000, city)
    print(population)
    init_value = population[0]

    gens = [0]
    costs = [cost_of(init_value, city)]
    x, y = coord_of(init_value, city)
    coords = [f'({x}, {y})']
    time = [f'{response_time_of(init_value, city):.2f} min']
    xs = [x]
    ys = [y]

    print(f'INITIAL COORDS: ({x}, {y})')
    print(f'INITIAL COST: {costs[0]:.2f}')

    for generation in range(1, 99):
        parents, N = truncation_selection(population, 0.90, city)
        logging.warning(f"parents:{parents}, {N}")
        children = crossover(parents, city, N)
        logging.warning(f"children:{children}, {N}")
        population = mutate(children, 0.2, parents, population, N, city)

        loc_of_best = population[0]
        cost_of_best = cost_of(loc_of_best, city)
        x, y = coord_of(loc_of_best, city)
        response_time_of_best = response_time_of(loc_of_best, city)

        print('=', end='', flush=True)

        gens.append(generation)
        costs.append(cost_of_best)
        coords.append(f'({x}, {y})')
        time.append(f'{response_time_of_best:.2f} min')

        xs.append(x)
        ys.append(y)

        plt.subplot(2, 1, 1)
        plt.scatter(xs, ys, c='blue', marker='o')
        plt.grid(True)
        plt.axis([0, 10, 0, 10])
        plt.xticks(np.arange(11))
        plt.yticks(np.arange(11))

        plt.subplot(2, 1, 2)
        plt.scatter(gens, costs, c='blue', marker='o')
        plt.plot(gens, costs, 'r')
        plt.xlabel('Generations')
        plt.ylabel('Cost Value')
        plt.title('Cost Value Per Generation (100 Generations)')
        plt.grid(True)

        plt.pause(0.1)

    print(']')
    df = pd.DataFrame({
        'Generations': gens,
        'ProposedCoordinates': coords,
        'CostValue': costs,
        'ResponseTime': time
    })
    print(df.to_string())

    plt.show()

if __name__ == "__main__":
    emergency_response_unit_locator()
