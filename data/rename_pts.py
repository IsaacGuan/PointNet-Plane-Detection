import os

os.chdir("./points")

files = os.listdir('.')
files.sort()

i = 1

for file in files:
	if file[-2: ] == 'py':
		continue
	#new_name = "table" + str(i) + ".pts"
	new_name = "chair" + str(i) + ".pts"
	i += 1
	os.rename(file, new_name)
