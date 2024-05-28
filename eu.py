import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import logging

#logging.basicConfig(level=logging.INFO, filename="algo.log", filemode="w", format= "%(message)s")

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
    child1 = [x1,y2]
    child2= [x2,y1]
    children.append(np.ravel_multi_index((child1[0],child1[1]), city.shape))
    children.append(np.ravel_multi_index((child2[0],child2[1]), city.shape))

    return children

def mutate(children, mutation_rate, city):
    mutated_children = []
    if np.random.rand() < mutation_rate:
            for i in range(0,2):
                xc, yc = coord_of(children[i], city)
                mutated_children.append(np.ravel_multi_index((yc, xc), city.shape))

    return mutated_children

def emergency_response_unit_locator():
    gens = []
    city = load_locations()
    population = init_population(city)
    print(population)

    costs = []
    coords = []
    time = []
    xs = []
    ys = []
    
    
    parent1 = tournament_selection(population, city, 3)

    for generation in range(0, 100):

        parent2 = tournament_selection(population, city, 3)

        #eliminate the bad parent
        # if generation != 0:
        #     while cost_of(parent2,city) > cost_of(parent1,city)*(1+0.5):
        #         parent2 = tournament_selection(population, city, 3)
        

        children = crossover(parent1, parent2, city)
        mutated_children = mutate(children,0.5,city)
        to_compare = [parent1, children[0], children[1]] + mutated_children
        best_individual = min(to_compare, key=lambda loc: cost_of(loc, city))
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
        
   

    
    plt.scatter(gens, costs, c='blue', marker='o')
    plt.plot(gens, costs, 'r')
    plt.xlabel('Generations')
    plt.ylabel('Cost Value')
    plt.title('Cost Value Per Generation (100 Generations)')
    plt.grid(True)

        
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
