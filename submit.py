import subprocess

#for _ in range(32,65,8):
#	subprocess.call(['qsub', 'jobscript','-F',str(_)])

for length in [33]:
	for i in range(50):
		subprocess.call(['qsub', 'jobscript','-F',str(length)])
