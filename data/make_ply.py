import os
from shutil import copyfile

os.chdir("./points")

for file in os.listdir('.'):
	copyfile(file, "../ply/" + file[:-4] + ".ply")

os.chdir("../ply")

for file in os.listdir('.'):
	f = open(file, "r+")
	lines = [line.rstrip() for line in f]
	f.seek(0)
	head = "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex " + str(len(lines)) + "\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n"
	f.write(head)
	for line in lines:
		f.write(line+"\n")
	f.close()
