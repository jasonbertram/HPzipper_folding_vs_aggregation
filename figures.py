import matplotlib.pyplot as plt
import numpy as np
from zip_functions import *
from analysis_functions import *
import cPickle
import os

#L_all=[]
#sequence_all=[]
#structure_all=[]
#mutations=[]
#
#for filename in os.listdir('.'):
#    if filename[:18]=='output1.0_1000_60_':
#        with open(filename,'r') as fin:
#            L_all.append(cPickle.load(fin))
#            sequence_all.append(cPickle.load(fin))
#            structure_all.append(cPickle.load(fin))
#            mutations.append(cPickle.load(fin))


with open ('fold_degeneracy_properties','r') as fin:
    sequence_all=cPickle.load(fin)
    structure_all=cPickle.load(fin)
    chance_complete_initial=cPickle.load(fin)
    chance_complete_final=cPickle.load(fin)
    mutations=cPickle.load(fin)
#    fold_degen_final=cPickle.load(fin)

structure_all=np.array(structure_all)
sequence_all=np.array(sequence_all)
mutations=map(np.array,mutations)

initial_sequences=np.array([_[0] for _ in sequence_all])
initial_structures=np.array([_[0] for _ in structure_all])
final_sequences=np.array([_[-1] for _ in sequence_all])
final_structures=np.array([_[-1] for _ in structure_all])

#chance_complete_initial=np.array(map(lambda x: chance_complete(x,1000),initial_sequences))
#chance_complete_final=np.array(map(lambda x: chance_complete(x,1000),final_sequences))
#fold_degen_final=np.array(map(lambda x: num_folds(x,100),final_sequences[complete_pos]))

#with open ('fold_degeneracy_properties_fixed','w') as fout:
#    cPickle.dump(sequence_all,fout)
#    cPickle.dump(structure_all,fout)
#    cPickle.dump(chance_complete_initial,fout)
#    cPickle.dump(chance_complete_final,fout)
#    cPickle.dump(mutations,fout)
##    cPickle.dump(fold_degen_final,fout)

incomplete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _==0.]
not_incomplete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _>0.]
not_complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _<1.]
not_incomplete_complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _>0. and _<1.]
complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _==1.]
#protein_like_pos=[_ for pos,_ in enumerate(complete_pos) if fold_degen_final[pos]==1]

def scale_points(x,y):
    tuples=zip(x,y)
    tuple_weights={_:0 for _ in set(tuples)}
    for _ in tuples:
        tuple_weights[_]=tuple_weights[_]+1
        
    return np.array([[_[0],_[1],tuple_weights[_]] for _ in tuple_weights])


L=float(len(sequence_all[0][0]))

#===============================================
#misc
#===============================================

ref_ind=complete_pos[0]
for i in complete_pos:
    plt.plot([np.sum((sequence_all[ref_ind][min([j,len(sequence_all[ref_ind])-1])]-sequence_all[i][min([j,len(sequence_all[i])-1])])**2) for j in range(max([len(sequence_all[ref_ind]),len(sequence_all[i])]))])
    
plt.xlim([0,30])


plt.matshow([[np.sum((final_sequences[i]-final_sequences[j])**2) for i in complete_pos] for j in complete_pos])
plt.colorbar()

plt.matshow([[np.sum((final_sequences[i]-final_sequences[j])**2) for i in range(len(sequence_all))] for j in range(len(sequence_all))])
plt.colorbar()


for _ in np.array(mutations)[complete_pos]:
    plt.plot([len([mut for mut in spectrum if mut >= 0]) for spectrum in _])
    
plt.xlim([0,100])

plt.hist(np.argmax([(_[1]-_[0])**2 for _ in sequence_all],1),bins=range(0,59))
plt.hist(np.argmax([(_[1]-_[0])**2 for _ in sequence_all[complete_pos]],1),bins=range(0,59))

plt.plot(range(1,59),np.mean([_[0] for _ in mutations],0))

plt.plot(np.argmax([(_[1]-_[0])**2 for _ in sequence_all],1),np.sum(final_structures[:,:2],1),'.')
plt.ylim([-5,40])

def mutation_positions(sequence_history):
    return np.array([np.argmax((sequence_history[i]-sequence_history[i-1])**2)-1 for i in range(1,len(sequence_history))])

def fitness_effects(index):
    positions=mutation_positions(sequence_all[index])
    return np.array([mutations[index][_][positions[_]] for _ in range(len(positions))])

def mutation_percentile(index):
    positions=mutation_positions(sequence_all[index])
    sorted_mutations=map(np.argsort,mutations[index])
    num_beneficials=np.array([float(len([_ for _ in spectrum if _>=0])) for spectrum in mutations[index]])
    
    return [(np.argwhere(sorted_mutations[_]==positions[_])[0][0]-len(sorted_mutations[_])+num_beneficials[_]+1)/num_beneficials[_] for _ in range(len(positions))]
    
for _ in not_complete_pos:
    plt.plot(fitness_effects(_),c='C0')
    
for _ in complete_pos:
    plt.plot(fitness_effects(_),c='C1')

plt.xlim([0,10])


for _ in not_complete_pos:
    plt.plot(np.sum(np.array(structure_all[_])[:,:2],1),c='C0',linewidth=0.5)

for _ in complete_pos:
    plt.plot(np.sum(np.array(structure_all[_])[:,:2],1),c='C1',linewidth=0.5)

plt.xlim([0,40])


for _ in not_complete_pos:
    plt.plot([np.sum((sequence_all[0][0]-sequence_all[_][j])**2) for j in range(len(sequence_all[_]))],np.sum(np.array(structure_all[_])[:,:2],1),c='C0',linewidth=0.5)

for _ in complete_pos:
    plt.plot([np.sum((sequence_all[0][0]-sequence_all[_][j])**2) for j in range(len(sequence_all[_]))],np.sum(np.array(structure_all[_])[:,:2],1),c='C1',linewidth=0.5)

#plt.xlim([0,60])



for _ in not_complete_pos:
    plt.plot([np.sum((sequence_all[0][0]-sequence_all[_][j])**2) for j in range(len(sequence_all[_]))],c='C0')
    

for _ in complete_pos:
    plt.plot([np.sum((sequence_all[0][0]-sequence_all[_][j])**2) for j in range(len(sequence_all[_]))],c='C1')

plt.xlim([0,100])
    


start=10
window=1
#plt.boxplot([np.mean(np.array([fitness_effects(_)[:window] for _ in not_complete_pos]),1),np.mean(np.array([fitness_effects(_)[:window] for _ in complete_pos]),1)])
plt.boxplot([np.array([fitness_effects(_)[start:start+window] for _ in not_complete_pos]).flatten(),np.array([fitness_effects(_)[start:start+window] for _ in complete_pos]).flatten()])

#===============================================
#Initial vs final
#===============================================

fig1, ((ax2,ax3),(ax4,ax1)) = plt.subplots(nrows=2,ncols=2,figsize=[7.5,7.5])

#===============================================
#Hydrophobicity
   
hydro_all=scale_points(map(hydrophobicity,initial_sequences[incomplete_pos]),map(hydrophobicity,final_sequences[incomplete_pos]))

ax1.plot([0,1],[0,1],'k',zorder=-2)
ax1.scatter(hydro_all[:,0],hydro_all[:,1],s=2*hydro_all[:,2],zorder=0)

hydro_pl=scale_points(map(hydrophobicity,initial_sequences[complete_pos]),map(hydrophobicity,final_sequences[complete_pos]))
ax1.scatter(hydro_pl[:,0],hydro_pl[:,1],s=2*hydro_pl[:,2],zorder=3,c='C3')

hydro_pl=scale_points(map(hydrophobicity,initial_sequences[not_incomplete_pos]),map(hydrophobicity,final_sequences[not_incomplete_pos]))
ax1.scatter(hydro_pl[:,0],hydro_pl[:,1],s=2*hydro_pl[:,2],zorder=1)

ax1.set_xlim([0.3,1.])
ax1.set_ylim([0.3,1.])
ax1.set_xlabel(r'Initial Hydrophobicity',fontsize=12)
ax1.set_ylabel(r'Final Hydrophobicity', fontsize=12)

#===============================================
#Fitness

A_all=scale_points(np.sum(initial_structures[incomplete_pos,:2],1),np.sum(final_structures[incomplete_pos,:2],1))

ax2.plot([-2,0.],[-2,0.],'k',zorder=-1)
ax2.scatter(A_all[:,0]/L,A_all[:,1]/L,s=2*A_all[:,2],zorder=0)

A_pl=scale_points(np.sum(initial_structures[complete_pos,:2],1),np.sum(final_structures[complete_pos,:2],1))
ax2.scatter(A_pl[:,0]/L,A_pl[:,1]/L,s=2*A_pl[:,2],zorder=3,c='C3')

A_pl=scale_points(np.sum(initial_structures[not_incomplete_pos,:2],1),np.sum(final_structures[not_incomplete_pos,:2],1))
ax2.scatter(A_pl[:,0]/L,A_pl[:,1]/L,s=2*A_pl[:,2],zorder=1)


ax2.set_xlim([-2,0.])
ax2.set_ylim([-1,0.6])
ax2.set_xlabel(r'Initial $\overline{F}$ / $L$',fontsize=12)
ax2.set_ylabel(r'Final $\overline{F}$ / $L$',fontsize=12)


#===============================================
#Fitness vs hydro initial

hydro_all=scale_points(map(hydrophobicity,initial_sequences[incomplete_pos]),np.sum(final_structures[incomplete_pos,:2],1))

#ax4.plot([0,1],[0,1],'k',zorder=-2)
ax3.scatter(hydro_all[:,0],hydro_all[:,1]/L,s=2*hydro_all[:,2],zorder=0)

hydro_pl=scale_points(map(hydrophobicity,initial_sequences[complete_pos]),np.sum(final_structures[complete_pos,:2],1))
ax3.scatter(hydro_pl[:,0],hydro_pl[:,1]/L,s=2*hydro_pl[:,2],zorder=3,c='C3')

hydro_pl=scale_points(map(hydrophobicity,initial_sequences[not_incomplete_pos]),np.sum(final_structures[not_incomplete_pos,:2],1))
ax3.scatter(hydro_pl[:,0],hydro_pl[:,1]/L,s=2*hydro_pl[:,2],zorder=1)

ax3.set_xlabel(r'Final Hydrophobicity',fontsize=12)
ax3.set_ylabel(r'Final $\overline{F}$ / $L$',fontsize=12)


#ax3.set_xlim([0.4,1.8])
#ax3.set_ylim([0,1.4])
ax3.set_xlabel(r'Initial Hydrophobicity',fontsize=12)
ax3.set_ylabel(r'Final $\overline{F}$ / $L$',fontsize=12)

#===============================================
#Fitness vs hydro final

hydro_all=scale_points(map(hydrophobicity,final_sequences[incomplete_pos]),np.sum(final_structures[incomplete_pos,:2],1))

#ax4.plot([0,1],[0,1],'k',zorder=-2)
ax4.scatter(hydro_all[:,0],hydro_all[:,1]/L,s=2*hydro_all[:,2],zorder=0)

hydro_pl=scale_points(map(hydrophobicity,final_sequences[complete_pos]),np.sum(final_structures[complete_pos,:2],1))
ax4.scatter(hydro_pl[:,0],hydro_pl[:,1]/L,s=2*hydro_pl[:,2],zorder=3,c='C3')

hydro_pl=scale_points(map(hydrophobicity,final_sequences[not_incomplete_pos]),np.sum(final_structures[not_incomplete_pos,:2],1))
ax4.scatter(hydro_pl[:,0],hydro_pl[:,1]/L,s=2*hydro_pl[:,2],zorder=1)

ax4.set_xlabel(r'Final Hydrophobicity',fontsize=12)
ax4.set_ylabel(r'Final $\overline{F}$ / $L$',fontsize=12)

plt.tight_layout()


#===============================================
#Clustering
#
#ax4.scatter(map(lambda x: dispersion_all_phases(x,3),initial_sequences[incomplete_pos]),map(lambda x: dispersion_all_phases(x,3),final_sequences[incomplete_pos]),s=2)
#ax4.scatter(map(lambda x: dispersion_all_phases(x,3),initial_sequences[not_incomplete_pos]),map(lambda x: dispersion_all_phases(x,3),final_sequences[not_incomplete_pos]),s=2)
#ax4.scatter(map(lambda x: dispersion_all_phases(x,3),initial_sequences[complete_pos]),map(lambda x: dispersion_all_phases(x,3),final_sequences[complete_pos]),c='r',s=2)
#ax4.plot([0,1.6],[0,1.6],'k')
#ax4.set_xlabel('Initial Clustering',fontsize=12)
#ax4.set_ylabel('Final Clustering',fontsize=12)
#
##===============================================
##Aggregation
#
#A_all=scale_points(-initial_structures[incomplete_pos,1],-final_structures[incomplete_pos,1])
#
#ax3.plot([0,2.],[0,2.],'k',zorder=-1)
#ax3.scatter(A_all[:,0]/L,A_all[:,1]/L,s=2*A_all[:,2],zorder=0)
#
#A_pl=scale_points(-initial_structures[complete_pos,1],-final_structures[complete_pos,1])
#ax3.scatter(A_pl[:,0]/L,A_pl[:,1]/L,s=2*A_pl[:,2],zorder=3,c='C3')
#
#A_pl=scale_points(-initial_structures[not_incomplete_pos,1],-final_structures[not_incomplete_pos,1])
#ax3.scatter(A_pl[:,0]/L,A_pl[:,1]/L,s=2*A_pl[:,2],zorder=1)
#
#
#ax3.set_xlim([0.4,1.8])
#ax3.set_ylim([0,1.4])
#ax3.set_xlabel(r'Initial Aggregation Potential / $L$',fontsize=12)
#ax3.set_ylabel(r'Final Aggregation Potential / $L$',fontsize=12)

#===========================================================
#Change vs initial A
#===========================================================




fig2, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=[7.5,7.5])

#===============================================
#Hydrophobicity

delta=np.array(map(hydrophobicity,final_sequences))-np.array(map(hydrophobicity,initial_sequences))
hydro_all=scale_points(-initial_structures[:,1]/L,delta)
ax1.scatter(hydro_all[:,0],hydro_all[:,1],s=2*hydro_all[:,2],zorder=0)

delta=np.array(map(hydrophobicity,final_sequences[protein_like_pos]))-np.array(map(hydrophobicity,initial_sequences[protein_like_pos]))
hydro_pl=scale_points(-initial_structures[protein_like_pos,1]/L,delta)
ax1.scatter(hydro_pl[:,0],hydro_pl[:,1],c='r',s=2*hydro_pl[:,2],zorder=1)
#ax1.set_xlim([0.35,1.])
#ax1.set_ylim([0.35,1.])
ax1.set_xlabel(r'Initial Aggregation Potential / $L$',fontsize=12)
ax1.set_ylabel(r'$\Delta$ Hydrophobicity', fontsize=12)

#===============================================
#Aggregation

delta=(-final_structures[:,1])-(-initial_structures[:,1])
A_all=scale_points(-initial_structures[:,1]/L,delta/L)
ax2.scatter(A_all[:,0],A_all[:,1],s=2*A_all[:,2],zorder=0)

delta=(-final_structures[protein_like_pos,1])-(-initial_structures[protein_like_pos,1])
A_pl=scale_points(-initial_structures[protein_like_pos,1]/L,delta/L)
ax2.scatter(A_pl[:,0],A_pl[:,1],c='r',s=2*A_pl[:,2],zorder=1)
#ax2.set_xlim([0.4,1.8])
#ax2.set_ylim([0,1.4])
ax2.set_xlabel(r'Initial Aggregation Potential / $L$',fontsize=12)
ax2.set_ylabel(r'$\Delta$ Aggregation Potential / $L$',fontsize=12)

#===============================================
#Clustering

delta=np.array(map(lambda x: dispersion_all_phases(x,3),final_sequences))-np.array(map(lambda x: dispersion_all_phases(x,3),initial_sequences))

ax3.scatter(-initial_structures[:,1]/L,delta,s=2)

delta=np.array(map(lambda x: dispersion_all_phases(x,3),final_sequences[protein_like_pos]))-np.array(map(lambda x: dispersion_all_phases(x,3),initial_sequences[protein_like_pos]))

ax3.scatter(-initial_structures[protein_like_pos,1]/L,delta,c='r',s=2)

ax3.set_xlabel(r'Initial Aggregation Potential / $L$',fontsize=12)
ax3.set_ylabel('$\Delta$ Clustering',fontsize=12)

#===============================================
#MeanRL

delta=np.array(map(mean_runlength_normalized,final_sequences))-np.array(map(mean_runlength_normalized,initial_sequences))
meanRL_all=scale_points(-initial_structures[:,1]/L,delta)
ax4.scatter(meanRL_all[:,0],meanRL_all[:,1],s=2*meanRL_all[:,2])

delta=np.array(map(mean_runlength_normalized,final_sequences[protein_like_pos]))-np.array(map(mean_runlength_normalized,initial_sequences[protein_like_pos]))
meanRL_pl=scale_points(-initial_structures[protein_like_pos,1]/L,delta)
ax4.scatter(meanRL_pl[:,0],meanRL_pl[:,1],c='r',s=2*meanRL_pl[:,2])

ax4.set_xlabel(r'Initial Aggregation Potential / $L$',fontsize=12)
ax4.set_ylabel(r'$\Delta$ Mean run length',fontsize=12)

plt.tight_layout()


##===========================================================
##Chance to be proteinlike
##===========================================================
#
#bins=np.arange(0.6,1.9,0.1)
#pl_counts=np.histogram(-initial_structures[protein_like_pos,1]/L,)[0].astype(float)
#total_counts=np.histogram(-initial_structures[:,1]/L,bins=np.arange(0.6,1.9,0.1))[0].astype(float)
#
#plt.plot(pl_counts/total_counts)



#===========================================================
#Hamming plot
#===========================================================

fig1, ax1 = plt.subplots(figsize=[3.5,3.5])
ax1.plot([0,35/L],[0,35/L],'k',zorder=-1)

#hamming_all=scale_points(map(lambda x: len(x)-1,structure_all),np.sum((initial_sequences - final_sequences)**2,1))
#ax1.scatter(hamming_all[:,0]/L,hamming_all[:,1]/L,s=1*hamming_all[:,2],zorder=1)

hamming_pl=scale_points(map(lambda x: len(x)-1,structure_all[complete_pos]),np.sum((initial_sequences - final_sequences)**2,1)[complete_pos])
ax1.scatter(hamming_pl[:,0]/L,hamming_pl[:,1]/L,s=2*hamming_pl[:,2],c='r',zorder=1)

ax1.set_xlabel(r'Number of substitutions$/L$',fontsize=12)
ax1.set_ylabel('Hamming distance$/L$',fontsize=12)


#fig1, ax1 = plt.subplots(figsize=[3.5,3.5])
#
#maziness=np.array(map(lambda x: len(x)-1,structure_all[protein_like_pos]))/np.sum((initial_sequences - final_sequences)**2,1)[protein_like_pos].astype(float)
##maziness_pl=scale_points(-initial_structures[protein_like_pos,1]/L,maziness)
#maziness_pl=scale_points(map(hydrophobicity,initial_sequences[protein_like_pos]),maziness)
#ax1.scatter(maziness_pl[:,0],maziness_pl[:,1],s=2*maziness_pl[:,2],zorder=0)
#
#ax1.set_xlabel(r'Initial Aggregation Potential / $L$',fontsize=12)
#ax1.set_ylabel(r'Maziness',fontsize=12)


#===========================================================
#Backtracking
#===========================================================

long_path_pos=[_ for pos,_ in enumerate(protein_like_pos) if len(sequence_all[_])>50]
#chance_complete_example=map(lambda x: chance_complete(x,1000),sequence_all[pos])

pos=long_path_pos[0]

structure_data=np.array(structure_all[pos])[:,:-1]
structure_data[:,1]=-structure_data[:,1]

fig1, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,ncols=1,figsize=[3.5,7.5])

ax1.plot(structure_data)
ax1.set_ylabel(r'$A$, $S$')
ax1.set_xticklabels([])

ax2.plot(chance_complete_example)
ax2.set_ylabel('Chance complete')
ax2.set_xticklabels([])

ax3.plot(map(hydrophobicity,sequence_all[pos]))
ax3.set_ylabel('Hydrophobicity')
ax3.set_xticklabels([])

ax4.plot(map(lambda x: dispersion_all_phases(x,3),sequence_all[pos]))
ax4.set_ylabel('Clustering')
ax4.set_xticklabels([])

ax5.plot(map(mean_runlength_normalized,sequence_all[pos]))
ax5.set_ylabel('Mean run length')
ax5.set_xlabel('Substitution number')

fig1.set_tight_layout()

#ax2.imshow(sequence_all[long_path_pos[0]])

fig1, ax1 = plt.subplots(nrows=1,ncols=1,figsize=[3.5,3.5])
ax1.imshow(sequence_all[long_path_pos[0]])
ax1.set_xlabel('Sequence position')
ax1.set_ylabel('Substitution number')

#===================================================================

fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

data=scale_points(map(hydrophobicity,initial_sequences),map(lambda x: dispersion_all_phases(x,3),initial_sequences))
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)


fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

data=scale_points(map(hydrophobicity,initial_sequences),chance_complete_initial)
ax1.scatter(data[:,0],data[:,1],s=0.5*data[:,2],zorder=0)

fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

data=scale_points(map(hydrophobicity,final_sequences),chance_complete_final)
ax1.scatter(data[:,0],data[:,1],s=0.5*data[:,2],zorder=0)


fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

data=scale_points(map(hydrophobicity,final_sequences),map(lambda x: dispersion_all_phases(x,3),final_sequences))
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)

data=scale_points(map(hydrophobicity,final_sequences[protein_like_pos]),map(lambda x: dispersion_all_phases(x,3),final_sequences[protein_like_pos]))
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)


fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

data=scale_points(map(lambda x: dispersion_all_phases(x,3),initial_sequences),map(mean_runlength_normalized,initial_sequences))
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)


fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

data=scale_points(map(hydrophobicity,initial_sequences),-initial_structures[:,1])
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)


fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

#data=scale_points(map(hydrophobicity,final_sequences),-initial_structures[:,1])
#ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)

data=scale_points(map(hydrophobicity,final_sequences),-initial_structures[protein_like_pos,1])
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],c='r',zorder=0)


fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

data=scale_points(map(hydrophobicity,initial_sequences),map(mean_runlength_normalized,initial_sequences))
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)


fig1, ax1 = plt.subplots(figsize=[3.5,3.5])

data=scale_points(map(hydrophobicity,final_sequences),map(mean_runlength_normalized,final_sequences))
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)

data=scale_points(map(hydrophobicity,final_sequences[protein_like_pos]),map(mean_runlength_normalized,final_sequences[protein_like_pos]))
ax1.scatter(data[:,0],data[:,1],s=2*data[:,2],zorder=0)




#=================================================
#Initial conditions
#=================================================
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






