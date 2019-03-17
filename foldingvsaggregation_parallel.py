import numpy as np
import pathos.multiprocessing as mp
import time
from function_definitions import *

L=64-1
H=0.5
sample_size=100
alpha=0.25

sequence=generate_initial_sequence(L,H)

#for parallelization
def Fpar(x):
	return F(x,sample_size)

pool=mp.Pool(16)
start=time.time()

sequence_history=[]
number_beneficial_history=[]
mutation_history=[]
fitness_history=[]
for i in range(1000):
        print "Mutation number: ", i
        sequence_history.append(sequence)
        Fvalues=np.array(pool.map(Fpar, [mutate(sequence,_) for _ in range(L)]+[sequence]))
	mutation_effects=(Fvalues[:-1,:-1]-Fvalues[-1,:-1])/Fvalues[-1,:-1]
        beneficial_mutations=[position for position,_ in enumerate(mutation_effects) if (_[0]>=0 and _[1]<=alpha*_[0]) or (_[0]<0 and _[1]<=_[0]/alpha)]
        number_beneficial_history.append(len(beneficial_mutations))
        if len(beneficial_mutations)>0:
            #pick random beneficial mutation and mutate
            mutation_position=np.random.randint(len(beneficial_mutations))
            print 'Number beneficial:', number_beneficial_history[-1]
            print 'H or P mutated:', sequence[beneficial_mutations[mutation_position]]
            mutation_history.append(mutation_effects[beneficial_mutations[mutation_position]])
            print 'Mutation effect: ', mutation_history[-1]
	    fitness_history.append(Fvalues[beneficial_mutations[mutation_position]])
            sequence=mutate(sequence,beneficial_mutations[mutation_position])

print (time.time()-start)/60.

import pickle
with open("/N/dc2/scratch/jxb/HPzipper/output03",'w') as fout:
    pickle.dump(sequence_history,fout)
    pickle.dump(number_beneficial_history,fout)
    pickle.dump(mutation_history,fout)
    pickle.dump(fitness_history,fout)

