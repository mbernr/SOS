import random as rn
import numpy as np
from utils import *
from deap import algorithms, base, creator, tools
import time

instance_path = "instances/knapsack/large_scale/knapPI_1_100_1000_1"
instance_opt_sol_path = "instances/knapsack/large_scale-optimum/knapPI_1_100_1000_1"

items = parse_knapsack(instance_path)
optimal_sol = parse_knapsack_optimal_solution(instance_opt_sol_path)

#----------VARIABLES--------------------
NBR_ITEMS = items.number_items
IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = items.capacity

NGEN = 1000 #number of generations
#MU = 500 #numer of individuals to select for next generation
#LAMBDA = 200 #number of children to produce on each generation
#CXPB = 0.7 #probability next generation is produced by crossover
#MUTPB = 0.3 #probability next generation is produced by mutation
#tournament_size = 3
verb = False
#----------------------------------------

# To assure reproductibility, the RNG seed is set prior to the items
# dict initialization.
rn.seed(69)

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", set, fitness=creator.Fitness)
print("num_generations, population_size, tournament_size, prb_crossover, prb_mutation, best_found, optimal, performance, run_time")

for path in [("instances/knapsack/low-dimensional/f2_l-d_kp_20_878","instances/knapsack/low-dimensional-optimum/f2_l-d_kp_20_878"),("instances/knapsack/large_scale/knapPI_1_100_1000_1","instances/knapsack/large_scale-optimum/knapPI_1_100_1000_1"),("instances/knapsack/large_scale/knapPI_3_1000_1000_1","instances/knapsack/large_scale-optimum/knapPI_3_1000_1000_1")]:

    instance_path = path[0]
    instance_opt_sol_path = path[1]
    print ("Now running the following instance: ", instance_path)

    for MU in [50, 100, 250, 500]:
        for tournament_size in [3, 10, 20]:
            for prb in [(0.2,0.7),(0.5,0.5),(0.7,0.2),(0.9,0.1)]:

                CXPB = prb[0]
                MUTPB = prb[1]

                toolbox = base.Toolbox()

                # Attribute generator
                toolbox.register("attr_item", rn.randrange, NBR_ITEMS)

                # Structure initializers
                toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_item, IND_INIT_SIZE)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)

                def evalKnapsack(individual):
                    weight = 0.0
                    value = 0.0
                    for item in individual:
                        weight += items.weights_items[item]
                        value += items.values_items[item]
                    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
                        return 0,             # Ensure overweighted bags are dominated
                    return value,

                def cxSet(ind1, ind2):
                    """Apply a crossover operation on input sets. The first child is the
                    intersection of the two sets, the second child is the difference of the
                    two sets.
                    """
                    temp = set(ind1)                # Used in order to keep type
                    ind1 &= ind2                    # Intersection (inplace)
                    ind2 ^= temp                    # Symmetric Difference (inplace)
                    return ind1, ind2

                def mutSet(individual):
                    """Mutation that pops or add an element."""
                    if rn.random() < 0.5:
                        if len(individual) > 0:     # We cannot pop from an empty set
                            individual.remove(rn.choice(sorted(tuple(individual))))
                    else:
                        #print("before: ", individual)
                        individual.add(rn.randrange(NBR_ITEMS))
                        #print("after: ", individual)
                    return individual,

                def eval_sol(sol):
                    val = 0
                    for individual in sol:
                        for item in individual:
                            val += items.values_items[item]
                    return val
                toolbox.register("evaluate", evalKnapsack)
                toolbox.register("mate", cxSet)
                toolbox.register("mutate", mutSet)
                toolbox.register("select", tools.selTournament, tournsize=tournament_size)



                pop = toolbox.population(n=MU)

                hof = tools.ParetoFront()

                stats = tools.Statistics(lambda ind: ind.fitness.values)
                stats.register("avg", np.mean, axis=0)
                stats.register("std", np.std, axis=0)
                stats.register("min", np.min, axis=0)
                stats.register("max", np.max, axis=0)


                start_time = time.time()

                algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats,
                                    halloffame=hof, verbose=verb)

                end_time = time.time()
                run_time = round(end_time - start_time, 2)

                performance = round((eval_sol(hof) / optimal_sol), 4)

                print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}s".format(
                    NGEN,
                    MU,
                    tournament_size,
                    CXPB,
                    MUTPB,
                    eval_sol(hof),
                    optimal_sol,
                    performance,
                    run_time
                ))


#-----------_JUNKYARD -------------------
#old selection method
#toolbox.register("select", tools.selNSGA2)

#old genetic algorithm call
#algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
#                          halloffame=hof)



#print("Selected items: ", hof)
#print("Solution found: ", eval_sol(hof))
#print("optimal solution was: ", optimal_sol)
#print("Runtime: ", run_time)