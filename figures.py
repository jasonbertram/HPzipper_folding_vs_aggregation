import matplotlib.pyplot as plt
import numpy as np
from zip_functions import *
from analysis_functions import *
import cPickle

#=================================================
#Initial conditions

initial_sequences_test=np.array([generate_initial_sequence_connected(60) for _ in range(300)])
plt.hist(map(hydrophobicity,initial_sequences_test),bins=np.arange(20.5,54.5)/60.)
#foldicity=np.array(map(lambda x: chance_complete(x,1000),initial_sequences))
#args=[pos for pos,_ in enumerate(fold_degen_all) if _>0]
#fitnesses=np.array(map(lambda x: F(x,1.,1000),initial_sequences))

fig1, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=[8,8])

ax1.hist(map(hydrophobicity,initial_sequences),20,density=True)
ax1.set_xlabel('Hydrophobicity')
ax1.set_ylabel('Frequency')

ax2.hist(map(mean_runlength_normalized,initial_sequences),20,density=True)
ax2.set_xlabel('Mean run length (normed)')
ax2.set_ylabel('Frequency')

ax3.hist(map(lambda x: dispersion_all_phases(x,6),initial_sequences),20,density=True)
ax3.set_xlabel('Hydrophobic clustering')
ax3.set_ylabel('Frequency')

#=================================================

import os

L_all=[]
sequence_all=[]
structure_all=[]

for filename in os.listdir('.'):
    if filename[:15]=='output1.0_1000_':
        with open(filename,'r') as fin:
            L_all.append(cPickle.load(fin))
            sequence_all.append(cPickle.load(fin))
            structure_all.append(cPickle.load(fin))

L_all=np.array(L_all)
structure_all=np.array(structure_all)
sequence_all=np.array(sequence_all)

initial_sequences=np.array([_[0] for _ in sequence_all])
initial_structures=np.array([_[0] for _ in structure_all])
final_sequences=np.array([_[-1] for _ in sequence_all])
final_structures=np.array([_[-1] for _ in structure_all])

#chance_complete_initial=np.array(map(lambda x: chance_complete(x,1000),initial_sequences))
#chance_complete_final=np.array(map(lambda x: chance_complete(x,1000),final_sequences))

with open ('fold_degeneracy_properties','r') as fin:
    chance_complete_initial=cPickle.load(fin)
    chance_complete_final=cPickle.load(fin)
    fold_degen_final=cPickle.load(fin)


incomplete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _==0.]
not_incomplete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _>0.]
not_complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _<1.]
complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _==1.]

#fold_degen_final=np.array(map(lambda x: num_folds(x,100),final_sequences[complete_pos]))
protein_like_pos=[_ for pos,_ in enumerate(complete_pos) if fold_degen_final[pos]==1]

#with open ('fold_degeneracy_properties','w') as fout:
#    cPickle.dump(chance_complete_initial,fout)
#    cPickle.dump(chance_complete_final,fout)
#    cPickle.dump(fold_degen_final,fout)

#plt.plot(map(hydrophobicity,initial_sequences[not_complete_pos]),map(hydrophobicity,final_sequences[not_complete_pos]),'.')
#plt.plot(map(hydrophobicity,initial_sequences[not_incomplete_pos]),map(hydrophobicity,final_sequences[not_incomplete_pos]),'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.scatter(map(hydrophobicity,initial_sequences[complete_pos]),map(hydrophobicity,final_sequences[complete_pos]),c='k',s=16)#./fold_degen_final**2)
plt.plot([0,1],[0,1],'k')
plt.xlim([0.2,1.])
plt.ylim([0.2,1.])
plt.xlabel('Initial Hydrophobicity')
plt.ylabel('Final Hydrophobicity')

plt.figure()
plt.plot(-initial_structures[not_complete_pos,1],-final_structures[not_complete_pos,1],'.')
#plt.plot(-initial_structures[not_incomplete_pos,1],-final_structures[not_incomplete_pos,1],'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.plot(-initial_structures[complete_pos,1],-final_structures[complete_pos,1],'xk')
plt.plot([0,80],[0,80],'k')
plt.xlabel('Initial Aggregation Potential')
plt.ylabel('Final Aggregation Potential')

plt.figure()
plt.plot(np.sum(initial_structures[not_complete_pos,:-1],1),np.sum(final_structures[not_complete_pos,:-1],1),'.')
#plt.plot(-initial_structures[not_incomplete_pos,1],-final_structures[not_incomplete_pos,1],'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.plot(np.sum(initial_structures[complete_pos,:-1],1),np.sum(final_structures[complete_pos,:-1],1),'xk')
#plt.plot(np.sum(initial_structures[protein_like_pos,:-1],1),np.sum(final_structures[protein_like_pos,:-1],1),'xb')
plt.plot([-75,0],[-75,0],'k')
plt.xlabel('Initial F')
plt.ylabel('Final F')

plt.figure()
plt.plot(map(hydrophobicity,initial_sequences[not_complete_pos]),-initial_structures[not_complete_pos,1],'.')
#plt.plot(-initial_structures[not_incomplete_pos,1],-final_structures[not_incomplete_pos,1],'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.plot(map(hydrophobicity,initial_sequences[complete_pos]),-initial_structures[complete_pos,1],'xk')
#plt.plot([0,80],[0,80],'k')
plt.xlabel('Initial Hydrophobicity')
plt.ylabel('Initial Aggregation Potential')


plt.figure()
plt.plot(map(lambda x: dispersion_all_phases(x,6),initial_sequences[not_complete_pos]),-initial_structures[not_complete_pos,1],'.')
#plt.plot(-initial_structures[not_incomplete_pos,1],-final_structures[not_incomplete_pos,1],'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.plot(map(lambda x: dispersion_all_phases(x,6),initial_sequences[complete_pos]),-initial_structures[complete_pos,1],'xk')
#plt.plot([0,80],[0,80],'k')
plt.xlabel('Initial Clustering 6')
plt.ylabel('Initial Aggregation Potential')


plt.figure()
plt.plot(map(mean_runlength_normalized,initial_sequences[not_complete_pos]),-initial_structures[not_complete_pos,1],'.')
#plt.plot(-initial_structures[not_incomplete_pos,1],-final_structures[not_incomplete_pos,1],'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.plot(map(mean_runlength_normalized,initial_sequences[complete_pos]),-initial_structures[complete_pos,1],'xk')
#plt.plot([0,80],[0,80],'k')
plt.xlabel('Initial Mean RL')
plt.ylabel('Initial Aggregation Potential')


plt.figure()
plt.plot(map(lambda x: dispersion_all_phases(x,6),initial_sequences[not_complete_pos]),map(lambda x: dispersion_all_phases(x,6),final_sequences[not_complete_pos]),'.')
#plt.plot(map(lambda x: dispersion_all_phases(x,6),initial_sequences[not_incomplete_pos]),map(lambda x: dispersion_all_phases(x,6),final_sequences[not_incomplete_pos]),'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.plot(map(lambda x: dispersion_all_phases(x,6),initial_sequences[complete_pos]),map(lambda x: dispersion_all_phases(x,6),final_sequences[complete_pos]),'xk')
plt.plot([0,1],[0,1],'k')
plt.xlabel('Initial Clustering 6')
plt.ylabel('Final Clustering 6')


plt.figure()
plt.plot(map(mean_runlength_normalized,initial_sequences[not_complete_pos]),map(mean_runlength_normalized,final_sequences[not_complete_pos]),'.')
#plt.plot(map(mean_runlength_normalized,initial_sequences[not_incomplete_pos]),map(mean_runlength_normalized,final_sequences[not_incomplete_pos]),'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.plot(map(mean_runlength_normalized,initial_sequences[complete_pos]),map(mean_runlength_normalized,final_sequences[complete_pos]),'xk')
plt.plot([0,1],[0,1],'k')
plt.xlim([0.5,1.1])
plt.ylim([0.5,1.1])

plt.figure()
#plt.plot(map(lambda x: len(x)-1,structure_all[not_complete_pos]),np.sum((initial_sequences - final_sequences)**2,1)[not_complete_pos],'.')
#plt.plot(map(lambda x: len(x)-1,sequence_all[not_incomplete_pos]),np.sum((initial_sequences - final_sequences)**2,1)[not_incomplete_pos],'o',markersize=10,markerfacecolor='none', markeredgecolor='k')
plt.scatter(map(lambda x: len(x)-1,structure_all[complete_pos]),np.sum((initial_sequences - final_sequences)**2,1)[complete_pos],marker='x',c='k',s=32)#./fold_degen_final**2)
plt.plot([0,30],[0,30],'k')


hamming_tuples=zip(map(lambda x: len(x)-1,structure_all[complete_pos]),np.sum((initial_sequences - final_sequences)**2,1)[complete_pos])
hamming_tuple_weights={_:0 for _ in set(hamming_tuples)}
for _ in hamming_tuples:
    hamming_tuple_weights[_]=hamming_tuple_weights[_]+1
    
hamming_tuple_plot_array=np.array([[_[0],_[1],hamming_tuple_weights[_]] for _ in hamming_tuple_weights])

plt.figure()
plt.plot([0,35],[0,35],'k',zorder=-1)
plt.scatter(hamming_tuple_plot_array[:,0],hamming_tuple_plot_array[:,1],s=6*hamming_tuple_plot_array[:,2],zorder=1)


long_path_pos=[pos for pos,_ in enumerate(sequence_all) if len(_)>60]# and pos in protein_like_pos]

plt.figure()
plt.plot(np.sum(initial_structures[complete_pos,:-1],1),np.sum(final_structures[complete_pos,:-1],1),'xk')
plt.plot(np.sum(initial_structures[long_path_pos,:-1],1),np.sum(final_structures[long_path_pos,:-1],1),'o',markersize=10)
plt.plot([-75,0],[-75,0],'k')
plt.xlabel('Initial F')
plt.ylabel('Final F')

plt.scatter(map(hydrophobicity,initial_sequences[complete_pos]),map(hydrophobicity,final_sequences[complete_pos]),c='r',s=16./fold_degen_final**2)
plt.scatter(map(hydrophobicity,initial_sequences[long_path_pos]),map(hydrophobicity,final_sequences[long_path_pos]),c='b',s=20)
plt.plot([0,1],[0,1],'k')
plt.xlim([0.2,1.])
plt.ylim([0.2,1.])
plt.xlabel('Initial Hydrophobicity')
plt.ylabel('Final Hydrophobicity')

plt.figure()
plt.plot(map(lambda x: dispersion_all_phases(x,6),initial_sequences[complete_pos]),map(lambda x: dispersion_all_phases(x,6),final_sequences[complete_pos]),'xk')
plt.plot(map(lambda x: dispersion_all_phases(x,6),initial_sequences[long_path_pos]),map(lambda x: dispersion_all_phases(x,6),final_sequences[long_path_pos]),'ob')
plt.plot([0,1],[0,1],'k')
plt.xlabel('Initial Clustering 6')
plt.ylabel('Final Clustering 6')






#========================================================================
#Building from shorter sequences
data_dict={_:[] for _ in L}
for _ in range(len(sequence)):
    data_dict[L[_]].append(sequence[_])

constructed_seq=[]
short_pos=[x for x,y in enumerate(L) if y==63]
for i in range(1000):
    random_positions=np.random.choice(short_pos,2)
    constructed_seq.append(np.concatenate([sequence[random_positions[0]],sequence[random_positions[1]]]))

f=mean_runlength_normalized
data=[map(f,constructed_seq),map(f,data_dict[127])]

plt.boxplot(data)






