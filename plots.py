import matplotlib
matplotlib.use('PS')

import matplotlib.pyplot as plt

import numpy as np
from function_definitions import *
import pickle

with open("/N/dc2/scratch/jxb/HPzipper/output",'r') as fin:
        sequence_history=pickle.load(fin)
        number_beneficial_history=pickle.load(fin)
        mutation_history=np.array(pickle.load(fin))
        fitness_history=np.array(pickle.load(fin))

plt.matshow(np.array(sequence_history))
plt.savefig("/N/dc2/scratch/jxb/HPzipper/sequence.ps")

plt.figure()
plt.plot(number_beneficial_history)
plt.ylabel(r"Number of beneficial mutations (out of $L$=47)")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/ben_mutations.ps")

plt.figure()
plt.plot(mutation_history[:,0])
plt.plot(mutation_history[:,1])
plt.ylabel(r"Mutation effects on foldability (+) and aggregation potential (-)")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/mut_effects.ps")

plt.figure()
plt.plot(np.array(map(np.sum,sequence_history))/float(len(sequence_history[0])))
plt.ylabel(r"Hydrophobicity")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/hydrophobicity.ps")

plt.figure()
plt.plot(map(lambda x: dispersion_all_phases(x,6),sequence_history))
plt.ylabel(r"Hydrophobic dispersion")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/dispersion.ps")

plt.figure()
plt.plot(fitness_history[:,:-1])
plt.ylabel(r"F,A")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/fitness.ps")


plt.figure()
plt.plot(fitness_history[:,-1])
plt.ylabel(r"Percent order")
plt.xlabel(r"Substitution")
plt.savefig("/N/dc2/scratch/jxb/HPzipper/order.ps")

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
