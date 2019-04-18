import time
from function_definitions import *
from dask import compute,delayed
import dask.multiprocessing

#from dask.distributed import Client
#from dask_mpi import initialize

#initialize()
#client=Client()

L=32-1
H=0.7
sample_size=1000
alpha=0.6
print "alpha=",alpha
print "L=",L

#for parallelization
def Fpar(x):
	return F(x,alpha,sample_size)

start=time.time()

sequence_history=[generate_initial_sequence(L,H)]
possible_mutation_history=[]
structure_history=[]
for i in range(50):
        print "Mutation number: ", i
        values=[delayed(Fpar)(x) for x in [mutate(sequence_history[-1],_) for _ in range(L)]+[sequence_history[-1]]]
        Fvalues=np.array(compute(*values, scheduler='processes'))
        original_fitness=np.sum(Fvalues[-1,:])
        mutant_fitnesses=np.sum(Fvalues[:-1,:],1)
        mutation_effects=(mutant_fitnesses-original_fitness)
        possible_mutation_history.append(mutation_effects)
        if np.max(mutation_effects)<0:
            Fvalues2ndorder=[]
            for pos in range(L):
                working_sequence=mutate(sequence_history[-1],pos)
                values=[delayed(Fpar)(x) for x in [mutate(working_sequence,_) for _ in range(L) if _!=pos]]
                Fvalues2ndorder.append(np.array(compute(*values, scheduler='processes')))

            Fvalues2ndorder=np.array(Fvalues2ndorder)
            mutant_fitnesses2ndorder=np.sum(Fvalues2ndorder[:,:,:],2)
            possible_valleys=np.array([[first,second,fval] for first,_ in enumerate(mutant_fitnesses2ndorder) for second,fval in enumerate(_) if np.max(_)>original_fitness and fval>original_fitness])
            
            if len(possible_valleys)==0:
                print "stuck"
                break
            
            valley_crossing=map(int,possible_valleys[np.argsort(possible_valleys[:,2])[-1]][:-1])
            
            structure_history.append(Fvalues[valley_crossing[0]])
            structure_history.append(Fvalues2ndorder[valley_crossing[0],valley_crossing[1]])
            
            sequence_history.append(mutate(sequence_history[-1],valley_crossing[0]))
            sequence_history.append(mutate(sequence_history[-1],valley_crossing[1]+int(first<second)))
            
            print "valley crossing!", valley_crossing
            print "Delta F=", np.sum(structure_history[-1][:-1])-original_fitness
            
        else:
            mutation_position=np.argmax(mutation_effects)
            print 'Mutation effect: ', mutation_effects[mutation_position]
            structure_history.append(Fvalues[mutation_position])
            sequence_history.append(mutate(sequence_history[-1],mutation_position))


print (time.time()-start)/60

import cPickle
with open("/N/dc2/scratch/jxb/HPzipper/output"+str(alpha)+'_'+str(sample_size)+'_'+str(L)+str(int(time.time()))[-4:],'w') as fout:
    cPickle.dump(alpha,fout)
    cPickle.dump(sequence_history,fout)
    cPickle.dump(possible_mutation_history,fout)
    #cPickle.dump(mutation_history,fout)
    cPickle.dump(structure_history,fout)
    #cPickle.dump(local_peaking_times,fout)

"""

sequence=np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1])


Farray=[]
for pos in range(L):
	working_sequence=mutate(sequence,pos) 
        Fvalues=np.array(pool.map(Fpar, [mutate(working_sequence,_) for _ in range(L) if _!=pos]))
	Farray.append(Fvalues)

Farray=np.array(Farray)


print (time.time()-start)/60

import cPickle
with open("/N/dc2/scratch/jxb/HPzipper/outputnbd"+str(alpha)+'_'+str(sample_size)+'_'+str(L),'w') as fout:
    cPickle.dump(Farray,fout)
"""
