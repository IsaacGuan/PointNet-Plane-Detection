import os

def number(x):
	return(int(x[5:-4]))

filelist = open("filelist", "w+")

files = os.listdir("./points")

for file in sorted(files, key = number):
	file_name = file[:-4]
	filelist.write("%s\n" % file_name)
