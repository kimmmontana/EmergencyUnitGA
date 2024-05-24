import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import logging

logging.basicConfig(level=logging.INFO, filename="algo.log", filemode="w", format= "%(message)s")

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

def init_population(city):
    return np.random.permutation(city.size)

def tournament_selection(population, city,  tournament_size):
    
    tournament = random.sample(list(population),tournament_size)
    best_individual = min(tournament, key=lambda loc: cost_of(loc, city))

    return best_individual

def crossover(parent1, parent2, city):
    children = []

    x1, y1 = coord_of(parent1, city)
    x2, y2 = coord_of(parent2, city)
    child = [x1 if np.random.rand() > 0.5 else y1,
            x2 if np.random.rand() > 0.5 else y2]
    xc, yc = child
    return np.ravel_multi_index((xc, yc), city.shape)
    '''''
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
    '''

def mutate(children, mutation_rate, city):
    if np.random.rand() < mutation_rate:
            print("mutated")
            xc, yc = coord_of(children, city)
            children = np.ravel_multi_index((yc, xc), city.shape)
    return children
    '''''
    for i in range(len(children)):
        if np.random.rand() < mutation_rate:
            xc, yc = coord_of(children[i], city)
            mutated = np.ravel_multi_index((yc, xc), city.shape)
            children[i] = mutated
    new_population = np.concatenate((children, parents, population[N:]))
    sorted_population = sorted(new_population, key=lambda loc: cost_of(loc, city))
    return np.array(sorted_population[:len(population)])
    '''

def emergency_response_unit_locator():
    gens = [0]
    city = load_locations()
    population = init_population(city)
    print(population)
    init_value = population[0]
    
    
    costs = [cost_of(init_value, city)]
    x, y = coord_of(init_value, city)
    coords = [f'({x}, {y})']
    time = [f'{response_time_of(init_value, city):.2f} min']
    xs = [x]
    ys = [y]

    print(f'INITIAL COORDS: ({x}, {y})')
    print(f'INITIAL COST: {costs[0]:.2f}')
    
    parent1 = tournament_selection(population, city, 3)

    for generation in range(1, 100):
        parent2 = tournament_selection(population, city, 3)
        children = crossover(parent1, parent2, city)
        mutated_indiv = mutate(children,0.2,city)
        best_individual = min(parent1,parent2,children,mutated_indiv, key=lambda loc: cost_of(loc, city))
        parent1 = best_individual

        loc_of_best = parent1
        cost_of_best = cost_of(parent1, city)
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

        plt.pause(0.001)
        
    print(']')
    df = pd.DataFrame({
        'Generations': gens,
        'ProposedCoordinates': coords,
        'CostValue': costs,
        'ResponseTime': time
    })
    print(df.to_string())

    #plt.show()

if __name__ == "__main__":
    emergency_response_unit_locator()
