import os
import random

os.chdir("./points")

for file in os.listdir('.'):
	f = open(file, "r+")
	lines = [line.rstrip() for line in f]
	slice = random.sample(lines, 2048)
	f.close()
	f = open(file, "w")
	for line in slice:
		f.write(line+"\n")
	f.close()
