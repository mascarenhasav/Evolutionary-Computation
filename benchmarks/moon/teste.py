import os

os.system("rm ./pops/pop_objs_eval.txt")
os.system("rm ./pops/pop_cons_eval.txt")
os.system("./moon_sop pops")
with open("pops/pop_objs_eval.txt", 'r') as f:
	lastLine = f.readlines()
	print(len(lastLine))
	for i in range(len(lastLine)):
		f1 = float(lastLine[i].split('\t')[0])
		print(f"row[{i}]\tf1:{f1}")

with open("pops/pop_cons_eval.txt", 'r') as f:
	lastLine = f.readlines()
	print(len(lastLine))
	for i in range(len(lastLine)):
		c1 = float(lastLine[i].split('\t')[0])
		c2 = float(lastLine[i].split('\t')[1])
		print(f"row[{i}]\tc1:{c1} c2:{c2}")
