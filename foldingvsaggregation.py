import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
from function_definitions import *

pool=multiprocessing.Pool(6)

L=48
H=0.5
sample_size=100

sequence=generate_initial_sequence(L,H)

start=time.time()

sequence_history=[]
number_beneficial_history=[]
mutation_history=[]
for i in range(100):
        print "Mutation number: ", i
        sequence_history.append(sequence)
    	Finitial=F(sequence,sample_size)
        #mutation_effects=np.array(pool.map(lambda x: (F(mutate(sequence,x),sample_size)-Finitial)/Finitial, range(L)))
        print pool.map(lambda x: (F(mutate(sequence,x),sample_size)-Finitial)/Finitial, range(L))
        beneficial_mutations=[position for position,_ in enumerate(mutation_effects) if _[0]>=0 and _[1]<=0]# and _[1]<=0]
        if len(beneficial_mutations)>0:
            #pick random beneficial mutation and mutate
            mutation_position=np.random.randint(len(beneficial_mutations))
            number_beneficial_history.append(len(beneficial_mutations))
            print 'Number beneficial:', number_beneficial_history[-1]
            print 'H or P mutated:', sequence[beneficial_mutations[mutation_position]]
            mutation_history.append(mutation_effects[beneficial_mutations[mutation_position]])
            print 'Mutation effect: ', mutation_history[-1]
            sequence=mutate(sequence,beneficial_mutations[mutation_position])
        else:
            #weakly_deleterious_mutations=[position for position,_ in enumerate(mutation_effects) if _[0]>=-0.01 and _[1]<=0]
            #if len(weakly_deleterious_mutations)>0:
            #    mutation_position=np.random.randint(len(weakly_deleterious_mutations))
            #    print len(weakly_deleterious_mutations),"deleterious"
            #    sequence=mutate(sequence,weakly_deleterious_mutations[mutation_position])
            #else:
            break

print (time.time()-start)/60.
"""
import pickle
with open("/N/dc2/scratch/jxb/HPzipper",'w') as fout:
    pickle.dump(sequence_history,fout)
    pickle.dump(number_beneficial_history,fout)
    pickle.dump(mutation_history,fout)
"""
