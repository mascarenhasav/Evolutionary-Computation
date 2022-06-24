import random
import array
import numpy as np
import os
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14, 8)
#plt.style.use('dark_background')
#plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['savefig.facecolor'] = 'black'

os.system("rm pops/*")

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

def evaluate(x, index):
	writeVars(x)
	fobj = readLine("pops/pop_objs_eval.txt", 1)
	fcons = readLine("pops/pop_cons_eval.txt", 2)

	#y = -fobj[0] + 0.7*(fcons[0] + fcons[1])
	y = fobj[0] + c1(fcons)
	#y = fobj[0]

#	print(f"pos[{x[0], x[1]}]\t fobj:{fobj} fcons:{fcons}")
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
# Differential evolution parameters
size = 2
CR = 0.6
F = 0.7

GEN = int(input("Numero de geracoes: "))
npop = int(input("Numero de individuos: "))
print("1-rand/1, 2-best/1, 3-rand/2, 4-best/2")
ms = int(input("Mutantion strategy: "))
N = int(input("Numero de tentativas: "))  # number of tentatives

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRandom, k=5)
#toolbox.register("evaluate", evaluate(index))

def main():
	fits = []
	gen = [i for i in range(GEN)]
	posx = [[] for i in range(npop)]
	posy = [[] for i in range(npop)]
	agentFits = [[] for _ in range(npop)]
	iterFits = []
	iterBests = []
	hof = [[] for _ in range(N+1)]

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	logbook = tools.Logbook()
	logbook.header = "gen", "evals", "std", "min", "avg", "max"

	fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)

	for j in range(1, N+1):
		print(f"N:{j}")
		pop = toolbox.population(n=npop);
		hof[j] = tools.HallOfFame(1)
		genFits = []
		genBests = []

		for i, ind in enumerate(pop):
			ind.fitness.values = evaluate(ind, i)
			agentFits[i].append(pop[i].fitness.values[0])

		record = stats.compile(pop)
		logbook.record(gen=0, evals=len(pop), **record)
		#print(logbook.stream)
		hof[j].update(pop)
		genFits.append(hof[j][0].fitness.values[0])

		for g in range(1, GEN):
			print(f"gen:{g}")
			for k, agent in enumerate(pop):
				a,b,c,d,e = toolbox.select(pop)
				y = toolbox.clone(agent)
				index = random.randrange(size)
				for i, value in enumerate(agent):
					if i == index or random.random() < CR:
						if(ms == 1):
							diff = a[i] + F*(b[i]-c[i])
						elif(ms == 2):
							diff = hof[j][0][i] + F*(b[i]-c[i])
						elif(ms == 3):
							diff = a[i] + F*(b[i]-c[i]) + F*(d[i]-e[i])
						elif(ms == 4):
							diff = hof[j][0][i] + F*(b[i]-c[i]) + F*(d[i]-e[i])
						if((diff>=0) and (diff<=1) ):
							y[i] = diff
							y.fitness.values = evaluate(y, k)
							if y.fitness > agent.fitness:
								pop[k] = y

				agentFits[k].append(pop[k].fitness.values[0])

			hof[j].update(pop)
			record = stats.compile(pop)
			logbook.record(gen=g, evals=len(pop), **record)
			genBests.append(hof[j][0])
			for i in range(npop):
				posx[i].append(pop[i][0])
				posy[i].append(pop[i][1])
				#print(f"agent:{i}\tpos[{pop[i][0], pop[i][1]}]\t fobj:{pop[i].fitness.values[0]}")

			#print(logbook.stream)
			print(f"Best:\tpos[{hof[j][0][0], hof[j][0][1]}]\t fobj:{hof[j][0].fitness.values[0]}")

		iterBests.append(hof[j][0])
		iterFits.append(hof[j][0].fitness.values[0])
		#hof = tools.HallOfFame(1)
		#print(f"\nBest N:{iterFits[-1]}")
		for j in range(npop):
			#ax1.plot(gen, agentFit[j], label=f'agent = {j+1} min:{min(agentFit[j])}')
			ax1.plot(gen, agentFits[j])
			agentFits[j] = []


	plt.grid(which='major', color='dimgrey', linewidth=0.8)
	plt.grid(which='minor', color='dimgrey', linestyle=':', linewidth=0.5)
	std = np.std(iterFits)
	min = np.min(iterFits)
	max = np.max(iterFits)
	median = np.median(iterFits)
	mean = np.mean(iterFits)
	print(f"--------------------------------------------------------------------")
	print(f"    min     |    median   |     mean    |     max     |     std    |")
	print(f"--------------------------------------------------------------------")
	print(f" {min:.6f}  |  {median:.6f}  |  {mean:.6f}  |  {max:.6f}  |  {std:.6f}  |")
	print(f"--------------------------------------------------------------------")
	writePop(pop)
	ax1.set_ylabel(f"Erro", fontsize=15)
	ax1.set_xlabel("Generations", fontsize=15)
	ax1.set_xlim(0, GEN)
	ax1.set_title(f"Erro x Generations   population={npop}", fontsize=15)
	#ax1.legend()
	for text in ax1.legend().get_texts():
		text.set_color('black')
	for g in range(npop):
		ax2.scatter(posx[g], posy[g])
	ax2.set_xlim(0, 1)
	ax2.set_ylim(0, 1)
	#ax2.set_title(f"DE with {npop} individuals and {GEN} generations\npos:{hof[0]} min:{hof[0].fitness.values[0]}", fontsize=15)
	ax2.set_ylabel("Latitude", fontsize=15)
	ax2.set_xlabel("Longitude", fontsize=15)
	#ax2.legend()
	for text in ax2.legend().get_texts():
		text.set_color('black')
	plt.tight_layout()
	plt.savefig("de.png")


if __name__ == "__main__":
	main()
