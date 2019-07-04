import time
from zip_functions import *
from dask import compute,delayed
import dask.multiprocessing
import cPickle
import sys

#from dask.distributed import Client
#from dask_mpi import initialize

#initialize()
#client=Client()

#L=int(sys.argv[1])-1
L=60
sample_size=1000
num_sequences=1
alpha=1.
print "alpha=",alpha
print "L=",L

start=time.time()

#sequence_history=[generate_initial_sequence_connected(L)]
sequence_history=[np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1,
       1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1])]
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
        #===========================
        #Valley Crossing
        #===========================
        """
        
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
        """
        #===========================
        #Improved estimate of current fitness
        #===========================
        
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
    #mutation_position=np.argmax(mutation_effects)
    print 'Mutation effect: ', mutation_effects[mutation_position]
    structure_history.append(Fvalues[mutation_position])
    sequence_history.append(mutate(sequence_history[-1],mutation_position+1))

print (time.time()-start)/60

with open("/N/dc2/scratch/jxb/HPzipper/output"+str(alpha)+'_'+str(sample_size)+'_'+str(L)+'_4_'+str(int(time.time()))[-6:],'w') as fout:
    cPickle.dump(L,fout)
    cPickle.dump(sequence_history,fout)
    cPickle.dump(structure_history,fout)
    cPickle.dump(mutation_history,fout)
