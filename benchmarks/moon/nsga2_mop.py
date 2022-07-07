#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json
import os
import matplotlib.pyplot as plt

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

plt.rcParams['figure.figsize'] = (14, 8)

os.system("rm pops_mop/*")



# ***********************************************************
# IO functions
# ***********************************************************
def writeVars(ind):
	with open("./pops_mop/pop_vars_eval.txt", 'w+') as f:
		line = f'{ind[0]}\t{ind[1]}\n'
		f.write(line)

	os.system("./moon_mop pops_mop")


def writePop(x):
	with open("./pops_mop/pop_vars_eval.txt", 'w+') as f:
		for ind in x:
			line = f'{ind[0]}\t{ind[1]}\n'
			f.write(line)

	os.system("./moon_mop pops_mop")


def readLine(path, i):
	values = [0 for _ in range(i)]
	with open(path, 'r') as f:
		lines = f.readlines()
		for j in range(i):
			values[j] = float(lines[-1].split('\t')[j])

	return values


# ***********************************************************
# Evaluate functions
# ***********************************************************
def evaluate(x):
	writeVars(x)
	fobj = readLine("pops_mop/pop_objs_eval.txt", 3)
	fcons = readLine("pops_mop/pop_cons_eval.txt", 2)
	y1 = fobj[0] + c1(fcons)
	y2 = fobj[1] + c1(fcons)
	y3 = fobj[2] + c1(fcons)
	return y1, y2, y3



# ***********************************************************
# Constraints functions
# ***********************************************************
def c1(x):
	penality = 0
	if ( (x[0] > 0) and (x[0] < 0.05) ):
		penality += 0
	else:
		penality += 0.3

	if( (x[1] > 0) and (x[1] < 0.3) ):
		penality += 0
	else:
		penality += 0.3

	return penality


def uniform(low, up, size=None):
	try:
		return [random.uniform(a, b) for a, b in zip(low, up)]
	except TypeError:
		return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
# ***********************************************************
# Evaluate functions
# ***********************************************************
NDIM = 2
CXPB = 0.9
BOUND_LOW, BOUND_UP = 0.0, 1.0

NGEN = int(input("Numero de geracoes: "))
NPOP = int(input("Numero de individuos: "))
NINT = int(input("Numero de tentativas: "))

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)



# ***********************************************************
# Evaluate functions
# ***********************************************************
def main(seed=None):
	random.seed(seed)
	fits = []
	posx = [[] for i in range(NPOP)]
	posy = [[] for i in range(NPOP)]
	iterFits = []
	iterBests = []
	hof = [[] for _ in range(NINT+1)]


	stats = tools.Statistics(lambda ind: ind.fitness.values)
	# stats.register("avg", numpy.mean, axis=0)
	# stats.register("std", numpy.std, axis=0)
	stats.register("min", numpy.min, axis=0)
	stats.register("max", numpy.max, axis=0)

	for j in range(1, NINT+1):
		print(f"N:{j}")
		pop = toolbox.population(n=NPOP)
		hof[j] = tools.HallOfFame(NPOP)
		genFits = []
		genBests = []

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in pop if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# This is just to assign the crowding distance to the individuals
		# no actual selection is done
		pop = toolbox.select(pop, len(pop))

		logbook = tools.Logbook()
		logbook.header = "gen", "evals", "std", "min", "avg", "max"
		record = stats.compile(pop)
		logbook.record(gen=0, evals=len(invalid_ind), **record)
		#print(logbook.stream)

		hof[j].update(pop)
		genFits.append(hypervolume(hof[j], [1, 0, 1]))

		# Begin the generational process
		for gen in range(1, NGEN):
			print(f"Gen: {gen}")
			# Vary the population
			offspring = tools.selTournamentDCD(pop, len(pop))
			offspring = [toolbox.clone(ind) for ind in offspring]

			for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
				if random.random() <= CXPB:
					toolbox.mate(ind1, ind2)

				toolbox.mutate(ind1)
				toolbox.mutate(ind2)
				del ind1.fitness.values, ind2.fitness.values

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit

			hof[j].update(pop)
			genBests.append(hof[j][0])

			for i in range(NPOP):
				posx[i].append(pop[i][0])
				posy[i].append(pop[i][1])

			# Select the next generation population
			pop = toolbox.select(pop + offspring, NPOP)
			record = stats.compile(pop)
			logbook.record(gen=gen, evals=len(invalid_ind), **record)
			# print(logbook.stream)
			print(f"BEST:pos[{hof[j][0][0], hof[j][0][1]}]\t FOBJS:[{hof[j][0].fitness.values[0], hof[j][0].fitness.values[1], hof[j][0].fitness.values[2]}]")

		iterBests.append(hof[j][0])
		iterFits.append(hof[j][0].fitness.values[0])

		print(f"{hof[j]}")
		print(f"pos:[{hof[j][0][0], hof[j][0][1]}]")

		#for j in range(NPOP):
			#ax1.plot(gen, )

		print(f"Final population hypervolume is {hypervolume(pop, [1, 0, 1])}")

	return pop, logbook

if __name__ == "__main__":
	# with open("pareto_front/zdt1_front.json") as optimal_front_data:
	#     optimal_front = json.load(optimal_front_data)
	# Use 500 of the 1000 points in the json file
	# optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

	pop, stats = main()
	# pop.sort(key=lambda x: x.fitness.values)

	# print(stats)
	# print("Convergence: ", convergence(pop, optimal_front))
	# print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

	# import matplotlib.pyplot as plt
	# import numpy

	# front = numpy.array([ind.fitness.values for ind in pop])
	# optimal_front = numpy.array(optimal_front)
	# plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
	# plt.scatter(front[:,0], front[:,1], c="b")
	# plt.axis("tight")
	# plt.show()
