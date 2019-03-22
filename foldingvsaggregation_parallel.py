import numpy as np
import pathos.multiprocessing as mp
import time
from function_definitions import *

L=32-1
H=0.5
sample_size=100
alpha=0.35
print "alpha= ",alpha

#sequence=generate_initial_sequence(L,H)
sequence=np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

#for parallelization
def Fpar(x):
	return F(x,sample_size)

pool=mp.Pool(16)
start=time.time()

Farriay=[]
for pos in range(L):
	working_sequence=mutate(sequence,pos) 
        Fvalues=np.array(pool.map(Fpar, [mutate(working_sequence,_) for _ in range(L)if _!=pos]))
	Farray.append(Fvalues)

Farray=np.array(Farray)

import pickle
with open('/N/dc2/scratch/jxb/HPzipper/outputnbd','w') as fout:
    pickle.dump(Farray,fout)


"""
sequence_history=[]
beneficial_history=[]
mutation_history=[]
fitness_history=[]
previous_mutation=-1
for i in range(400):
        print "Mutation number: ", i
        sequence_history.append(sequence)
        Fvalues=np.array(pool.map(Fpar, [mutate(sequence,_) for _ in range(L)]+[sequence]))
	mutation_effects=(Fvalues[:-1,:-1]-Fvalues[-1,:-1])/Fvalues[-1,:-1]
	beneficial_mutations=np.argsort(np.sum(np.array([1,-alpha])*mutation_effects,1))
	beneficial_history.append(mutation_effects[beneficial_mutations])
        #beneficial_mutations=[position for position,_ in enumerate(mutation_effects) if (_[0]>=0 and _[1]<=alpha*_[0]) or (_[0]<0 and _[1]<=_[0]/alpha)]
        if len(beneficial_mutations)>0:
            #pick random beneficial mutation and mutate
            #mutation_position=np.random.randint(len(beneficial_mutations))
	    mutation_position=-1
            mutation_history.append(mutation_effects[beneficial_mutations[mutation_position]])
            print 'Mutation effect: ', mutation_history[-1]
	    fitness_history.append(Fvalues[beneficial_mutations[mutation_position]])
            sequence=mutate(sequence,beneficial_mutations[mutation_position])
	    previous_mutation=beneficial_mutations[mutation_position]

print (time.time()-start)/60.

import pickle
with open("/N/dc2/scratch/jxb/HPzipper/output"+str(alpha),'w') as fout:
    pickle.dump(alpha,fout)
    pickle.dump(sequence_history,fout)
    pickle.dump(beneficial_history,fout)
    pickle.dump(mutation_history,fout)
    pickle.dump(fitness_history,fout)
"""
