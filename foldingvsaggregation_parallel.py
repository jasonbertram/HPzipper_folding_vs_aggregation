import time
from zip_functions import *
from dask import compute,delayed
import dask.multiprocessing
import cPickle
import sys

#uses dask to compute fitness of all possible mutations in parallel
#at most use all the cores on one node, couldn't get multinode dask to work reliably
#run multiple calls of this script across different nodes to run multiple paths in parallel


L=int(sys.argv[1])
#number of zipped conformations used to estimate fitness
sample_size=1000
alpha=1.0
#odds of H vs P residues in initial sequence
odds=5
print "alpha=",alpha
print "L=",L

start=time.time()

#Initial sequence used to generate Fig 4 in Bertram and Masel, Genetics, 2020
#sequence_history=[np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1,
#       1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
#       1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1])]


sequence_history=[generate_initial_sequence_connected(L,odds)]
mutation_history=[]
structure_history=[F(sequence_history[-1],alpha,sample_size)]
for i in range(200):
    print "Mutation number: ", i
    values=[delayed(F,pure=True,traverse=False)(x,alpha,sample_size) for x in [mutate(sequence_history[-1],_) for _ in range(1,L-1)]+[sequence_history[-1]]]
    Fvalues=np.array(compute(*values, scheduler='processes'))
    original_fitness=np.sum(Fvalues[-1,:-1])
    mutant_fitnesses=np.sum(Fvalues[:-1,:-1],1)
    mutation_effects=(mutant_fitnesses-original_fitness)        
    if np.max(mutation_effects)<0:
        
        #Improve estimate of current fitness
                
        values=[delayed(F,pure=True,traverse=False)(sequence_history[-1],alpha,sample_size) for _ in range(L/10)]
        Fcurrentvalues=np.array(compute(*values, scheduler='processes'))
        current_fitnesses=np.sum(Fcurrentvalues[:,:-1],1)
        original_fitness=np.mean(current_fitnesses)
        mutation_effects=(mutant_fitnesses-original_fitness)
        
        if np.max(mutation_effects)<=0:
            break              
    
    mutation_history.append(mutation_effects)
    beneficials=[pos for pos,_ in enumerate(mutation_effects) if _>=0]
    mutation_position=np.random.choice(beneficials)
    print 'Mutation effect: ', mutation_effects[mutation_position]
    structure_history.append(Fvalues[mutation_position])
    sequence_history.append(mutate(sequence_history[-1],mutation_position+1))

print (time.time()-start)/60

with open("/N/dc2/scratch/jxb/HPzipper/output"+str(alpha)+'_'+str(sample_size)+'_'+str(L)+'_'+str(odds)+'_'+str(int(time.time()))[-6:],'w') as fout:
    cPickle.dump(L,fout)
    cPickle.dump(sequence_history,fout)
    cPickle.dump(structure_history,fout)
    cPickle.dump(mutation_history,fout)
