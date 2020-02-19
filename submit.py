import subprocess

#300 walks with L=60
for length in [60]:
	for i in range(300):
		subprocess.call(['qsub', 'jobscript','-F',str(length)])
