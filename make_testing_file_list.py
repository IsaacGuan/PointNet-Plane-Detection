import os

def number(x):
	return(int(x[5:-4]))

testing_filelist = open("testing_ply_file_list", "w+")

files = os.listdir("./data/points")

for file in sorted(files, key = number):
	file_name = file[:-4]
	testing_filelist.write("points/%s.pts points_label/%s.seg\n" % (file_name, file_name))
