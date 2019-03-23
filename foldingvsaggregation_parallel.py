import numpy as np
import pathos.multiprocessing as mp
import time
from function_definitions import *

L=32-1
H=0.5
sample_size=100
alpha=0.35
print "alpha= ",alpha

sequence=generate_initial_sequence(L,H)

#for parallelization
def Fpar(x):
	return F(x,sample_size)

pool=mp.Pool(16)
start=time.time()

sequence_history=[]
possible_mutation_history=[]
mutation_history=[]
fitness_history=[]
possible_fitness_history=[]
for i in range(1000):
        print "Mutation number: ", i
        sequence_history.append(sequence)
        Fvalues=np.array(pool.map(Fpar, [mutate(sequence,_) for _ in range(L)]+[sequence]))
	mutation_effects=(Fvalues[:-1,:-1]-Fvalues[-1,:-1])/Fvalues[-1,:-1]
	possible_mutations=np.argsort(np.sum(np.array([1,-alpha])*mutation_effects,1))
	possible_fitness_history=np.sum(np.array([1,-alpha])*mutation_effects,1)[possible_mutations]
	possible_mutation_history.append(mutation_effects[possible_mutations])
        #possible_mutations=[position for position,_ in enumerate(mutation_effects) if (_[0]>=0 and _[1]<=alpha*_[0]) or (_[0]<0 and _[1]<=_[0]/alpha)]
        if len([_ for _ in possible_fitness_history if _>=0.])==0:
            mutation_position=np.random.randint(len(possible_mutations))
	else:
	    mutation_position=-1
            mutation_history.append(mutation_effects[possible_mutations[mutation_position]])
            print 'Mutation effect: ', mutation_history[-1]
	    fitness_history.append(Fvalues[possible_mutations[mutation_position]])
            sequence=mutate(sequence,possible_mutations[mutation_position])

print (time.time()-start)/60.

import cPickle
with open("/N/dc2/scratch/jxb/HPzipper/output"+str(alpha),'w') as fout:
    cPickle.dump(alpha,fout)
    cPickle.dump(sequence_history,fout)
    cPickle.dump(possible_mutation_history,fout)
    cPickle.dump(mutation_history,fout)
    cPickle.dump(fitness_history,fout)


"""
sequence=np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])


Farray=[]
for pos in range(L):
	working_sequence=mutate(sequence,pos) 
        Fvalues=np.array(pool.map(Fpar, [mutate(working_sequence,_) for _ in range(L)if _!=pos]))
	Farray.append(Fvalues)

Farray=np.array(Farray)
"""
