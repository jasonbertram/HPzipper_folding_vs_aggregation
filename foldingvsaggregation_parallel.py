import numpy as np
import pathos.multiprocessing as mp
import time
from function_definitions import *

L=32-1
H=0.5
sample_size=1000
alpha=0.3
print "alpha= ",alpha
print "L= ",L

sequence=generate_initial_sequence(L,H)

#for parallelization
def Fpar(x):
	return F(x,alpha,sample_size)

pool=mp.Pool(16)
start=time.time()

sequence_history=[]
possible_mutation_history=[]
mutation_history=[]
structure_history=[]
local_peaking_times=[]
for i in range(100):
        print "Mutation number: ", i
        sequence_history.append(sequence)
        Fvalues=np.array(pool.map(Fpar, [mutate(sequence,_) for _ in range(L)]+[sequence]))
	original_fitness=np.sum(Fvalues[-1,:-1])
	mutant_fitnesses=np.sum(Fvalues[:-1,:-1],1)
	mutation_effects=(mutant_fitnesses-original_fitness)
	possible_mutation_history.append(mutation_effects)
        if np.max(mutation_effects)<0:
            mutation_position=np.random.randint(L)
	    local_peaking_times.append(i)
	else:
	    mutation_position=np.argmax(mutation_effects)
        mutation_history.append(mutation_effects[mutation_position])
        print 'Mutation effect: ', mutation_history[-1]
	structure_history.append(Fvalues[mutation_position])
        sequence=mutate(sequence,mutation_position)

print (time.time()-start)/60.

import cPickle
with open("/N/dc2/scratch/jxb/HPzipper/output"+str(alpha)+'_'+str(sample_size)+'_'+str(L),'w') as fout:
    cPickle.dump(alpha,fout)
    cPickle.dump(sequence_history,fout)
    cPickle.dump(possible_mutation_history,fout)
    cPickle.dump(mutation_history,fout)
    cPickle.dump(structure_history,fout)
    cPickle.dump(local_peaking_times,fout)


"""
sequence=np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])


Farray=[]
for pos in range(L):
	working_sequence=mutate(sequence,pos) 
        Fvalues=np.array(pool.map(Fpar, [mutate(working_sequence,_) for _ in range(L)if _!=pos]))
	Farray.append(Fvalues)

Farray=np.array(Farray)
"""
