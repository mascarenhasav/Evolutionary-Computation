from numpy.lib.npyio import savetxt
import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from pyswarm import pso

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

plt.rcParams['figure.figsize'] = (16, 6)

os.system("rm pops/*")

#****************************************************
# Functions for PSO with DEAP
def generate(size, pmin, pmax, smin, smax):
	part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
	part.speed = [random.uniform(smin, smax) for _ in range(size)]
	part.smin = smin
	part.smax = smax
	return part

def updateParticle(part, best, phi1, phi2, omega):
	w  = (random.uniform(0, omega) for _ in range(len(part)))
	s = 0
	#c1 = (phi1 for _ in range(len(part)))
	#c2 = (phi2 for _ in range(len(part)))
	c1 = (random.uniform(0, phi1) for _ in range(len(part)))
	c2 = (random.uniform(0, phi2) for _ in range(len(part)))
	vPart  = map(operator.mul, c1, map(operator.sub, part.best, part))
	vSwarm = map(operator.mul, c2, map(operator.sub, best, part))
	part.speed = list(map(operator.add, map(operator.mul, w, part.speed), map(operator.add, vPart, vSwarm)))
	#print(part.speed)
	for i, speed in enumerate(part.speed):
		if abs(speed) < part.smin:
			#part.speed[i] = math.copysign(part.smin, speed)
			part.speed[i] = part.smin
		elif abs(speed) > part.smax:
			#part.speed[i] = math.copysign(part.smax, speed)
			part.speed[i] = part.smax
		v = part[i] + part.speed[i]
		if( (v >= 0) and (v <= 1) ):
			s += 1

	if(s>=2):
		part[:] = list(map(operator.add, part, part.speed))
	#print(part.speed)

def writeVars(ind):
	with open("./pops/pop_vars_eval.txt", 'w+') as f:
		line = f'{ind[0]}\t{ind[1]}\n'
		f.write(line)
	os.system("./moon_sop pops")

def writePop(x):
	with open("./pops/pop_vars_eval.txt", 'w+') as f:
		for ind in x:
			line = f'{ind[0]}\t{ind[1]}\n'
			f.write(line)
	os.system("./moon_sop pops")


def readLine(path, i):
	values = [0 for _ in range(i)]
	with open(path, 'r') as f:
		lines = f.readlines()
		#print(lines)
		for j in range(i):
			values[j] = float(lines[-1].split('\t')[j])

		#print(values)
		return values

def evaluate(x):
	writeVars(x)
	fobj = readLine("pops/pop_objs_eval.txt", 1)
	fcons = readLine("pops/pop_cons_eval.txt", 2)

	#y = -fobj[0] + 0.7*(fcons[0] + fcons[1])
	#y = fobj[0] + c1(fcons)
	y = fobj[0]

	print(f"pos[{x[0], x[1]}]\t fobj:{fobj} fcons:{fcons}")
	return y,

#***************************************
#constraints functions
def c1(x):
	penality = 0
	if ( (x[0] > 0) and (x[0] < 0.05) ):
		penality += 0
	else:
		penality += 0.3
	if ( (x[1] > 0) and (x[1] < 0.3) ):
		penality += 0
	else:
		penality += 0.3
	return penality



#***************************************
# parameters
size = 2

pmin = 0
pmax = 1
smin = 0
smax = 0.5

phi1 = 1
phi2 = 1
omega = 0.5

GEN = int(input("Numero de geracoes: "))
npop = int(input("Numero de individuos: "))

N = 1  # number of tentatives


#***************************************
# initialization of the operators in deap
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #-1 means minimize
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
    smin=None, smax=None, best=None)
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=size, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=phi1, phi2=phi2, omega=omega)

def main():
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	logbook = tools.Logbook()
	logbook.header = ["gen", "evals"] + stats.fields

	best = None
	bestG = None
	fit = 0
	gen = [i for i in range(GEN)]

	posx = [[] for i in range(npop)]
	posy = [[] for i in range(npop)]
	partFit = [[] for i in range(npop)]

	fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)

	for j in range(N):
		pop = toolbox.population(n=npop);
		hof = tools.HallOfFame(1)
		fits = []

		for i, part in enumerate(pop):
			part.fitness.values = evaluate(part)
			if not part.best:
				part.best = creator.Particle(part)
				part.best.fitness.values = part.fitness.values
			if not best:
				best = creator.Particle(part)
				best.fitness.values = part.fitness.values
			partFit[i].append(pop[i].fitness.values[0])

		record = stats.compile(pop)
		logbook.record(gen=0, evals=len(pop), **record)
		#print(logbook.stream)
		hof.update(pop)
		fits.append(hof[0].fitness.values[0])

		for g in range(1, GEN):
			print(f"gen:{g}")
			for k, part in enumerate(pop):
				print(f"part:{k}")
				#print(pop[i].fitness.values)
				toolbox.update(part, best)
				part.fitness.values = evaluate(part)
				if part.best.fitness > part.fitness:
					pop[i].best = creator.Particle(pop[i])
					pop[k].best.fitness.values = part.fitness.values
				if best.fitness > part.fitness:
					best = creator.Particle(pop[i])
					best.fitness.values = part.fitness.values


				'''
				print(f'part: {part}\t{part.fitness.values}')
				print(f'pbest:{part.best}\t{part.best.fitness.values}')
				print(f'best: {best}\t{best.fitness.values}')
				print(f'---------------------------------------------')
				'''

				partFit[k].append(pop[k].fitness.values[0])

			hof.update(pop)
			record = stats.compile(pop)
			logbook.record(gen=g, evals=len(pop), **record)
			fits.append(hof[0].fitness.values[0])
			for i in range(npop):
				posx[i].append(pop[i][0])
				posy[i].append(pop[i][1])
				print(f"part:{i}\tpos[{pop[i][0], pop[i][1]}]\t fobj:{pop[i].fitness.values[0]}")

			#print(logbook.stream)
	for j in range(npop):
		#ax1.plot(gen, agentFit[j], label=f'agent = {j+1} min:{min(agentFit[j])}')
		ax1.plot(gen, partFit[j])


	plt.grid(which='major', color='dimgrey', linewidth=0.8)
	plt.grid(which='minor', color='dimgrey', linestyle=':', linewidth=0.5)
	print(f"\nBest:{hof[0].fitness.values[0]}\tpos:{hof[0][0]}, long:{hof[0][1]}")
	writePop(pop)
	ax1.set_ylabel(f"Erro", fontsize=15)
	ax1.set_xlabel("Generations", fontsize=15)
	ax1.set_xlim(0, GEN-1)
	ax1.set_title(f"Erro x Generations   population={npop}", fontsize=15)
	#ax1.legend()
	for text in ax1.legend().get_texts():
		text.set_color('black')
	for g in range(npop):
		ax2.scatter(posx[g], posy[g])
	ax2.set_xlim(0, 1)
	ax2.set_ylim(0, 1)
	ax2.set_title(f"PSO with {npop} individuals and {GEN} generations\npos:{hof[0]} min:{hof[0].fitness.values[0]}", fontsize=15)
	ax2.set_ylabel("Latitude", fontsize=15)
	ax2.set_xlabel("Longitude", fontsize=15)
	#ax2.legend()
	for text in ax2.legend().get_texts():
		text.set_color('black')
	plt.tight_layout()
	plt.savefig("pso_f.png")

	return pop, logbook, best


if __name__ == "__main__":
	main()
