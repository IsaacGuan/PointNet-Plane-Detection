filelist = [line.rstrip() for line in open("filelist", 'r')]

labels = open("labels", "w+")

for i in range(0, len(filelist)):
	labels.write("%d\n" % 0)
