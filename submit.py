import subprocess

#for _ in range(32,65,8):
#	subprocess.call(['qsub', 'jobscript','-F',str(_)])

for length in [61]:
	for i in range(300):
		subprocess.call(['qsub', 'jobscript','-F',str(length)])
