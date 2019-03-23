import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt
import numpy as np
from function_definitions import *
import sys
import cPickle

with open(str(sys.argv[1]),'r') as fin:
	alpha=cPickle.load(fin)
        sequence_history=cPickle.load(fin)
        possible_mutation_history=cPickle.load(fin)
        mutation_history=np.array(cPickle.load(fin))
        structure_history=np.array(cPickle.load(fin))
        local_peaking_times=np.array(cPickle.load(fin))

plt.matshow(np.array(sequence_history))
plt.savefig("/N/dc2/scratch/jxb/HPzipper/sequence.pdf")

plt.figure()
plt.plot([np.sum((sequence_history[0]-_)**2) for _ in sequence_history])
plt.ylabel(r"Hamming Distance")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/hamming.pdf")

#plt.figure()
#ordered_fitness=np.transpose(np.sum(possible_mutation_history,2))
#plt.plot(ordered_fitness)
#plt.plot(ordered_fitness,'.')
#plt.ylabel(r"Number of beneficial mutations (out of $L$=63)")
#plt.xlabel(r"Substitution")
#plt.savefig("/N/dc2/scratch/jxb/HPzipper/ben_mutations.pdf")

plt.figure()
#plt.plot(mutation_history[:,0])
plt.plot(mutation_history)
plt.ylabel(r"Mutation effects on foldability (+) and aggregation potential (-)")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/mut_effects.pdf")

plt.figure()
plt.plot(np.array(map(np.sum,sequence_history))/float(len(sequence_history[0])))
plt.ylabel(r"Hydrophobicity")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/hydrophobicity.pdf")

plt.figure()
plt.plot(map(lambda x: dispersion_all_phases(x,6),sequence_history))
plt.ylabel(r"Hydrophobic dispersion")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/dispersion.pdf")

plt.figure()
plt.plot(fitness_history[:,:-1])
plt.plot(np.sum(np.array([1,-alpha])*mutation_history,1),linewidth=2)
plt.ylabel(r"F,A")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/fitness.pdf")

plt.figure()
plt.plot(fitness_history[:,-1])
plt.ylabel(r"Percent order")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/order.pdf")

"""
start=time.time()
mutation_effects=mutations(sequence,sample_size)
plt.figure()
plt.plot(mutation_effects[:,0],mutation_effects[:,1],'.')
plt.plot([np.min(mutation_effects[:,0]),np.max(mutation_effects[:,0])],[0,0],'k',linewidth=2)
plt.plot([0,0],[np.min(mutation_effects[:,1]),np.max(mutation_effects[:,1])],'k',linewidth=2)
plt.xlabel(r"$\Delta$ Foldability")
plt.ylabel(r"$\Delta$ Aggregation")
print time.time()-start
"""
