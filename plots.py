import numpy as np
import matplotlib.pyplot as plt
from function_definitions import *
import pickle

with open("/N/dc2/scratch/jxb/L50S1000",'r') as fin:
        sequence_history=pickle.load(fin)
        number_beneficial_history=pickle.load(fin)
        mutation_history=np.array(pickle.load(fin))

plt.matshow(np.array(sequence_history))
plt.show()

plt.figure()
plt.plot(number_beneficial_history)
plt.ylabel(r"Number of beneficial mutations (out of $L$=50)")
plt.xlabel(r"Substitution")
plt.show()

plt.figure()
plt.plot(mutation_history[:,0])
plt.plot(mutation_history[:,1])
plt.ylabel(r"Mutation effects on foldability (+) and aggregation potential (-)")
plt.xlabel(r"Substitution")
plt.show()

plt.figure()
plt.plot(np.array(map(np.sum,sequence_history))/len(sequence_history[0]))
plt.ylabel(r"Hydrophobicity")
plt.xlabel(r"Substitution")
plt.show()

plt.figure()
plt.plot(map(lambda x: dispersion_all_phases(x,6),sequence_history))
plt.ylabel(r"Hydrophobic dispersion")
plt.xlabel(r"Substitution")
plt.show()

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
