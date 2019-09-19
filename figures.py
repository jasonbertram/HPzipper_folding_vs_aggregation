import matplotlib.pyplot as plt
import numpy as np
from zip_functions import *
from analysis_functions import *
import cPickle
import os

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

def path_length(structure_history):
    F_history=np.sum(np.array(structure_history)[:,:2],1)
    #return F_history[1:]-F_history[:-1]
    if len(structure_history)>1:
        return [pos for pos,_ in enumerate(F_history[1:]-F_history[:-1]) if _>0][-1]+1
    else:
        return 0
    
def scale_points(x,y):
    tuples=zip(x,y)
    tuple_weights={_:0 for _ in set(tuples)}
    for _ in tuples:
        tuple_weights[_]=tuple_weights[_]+1
        
    return np.array([[_[0],_[1],tuple_weights[_]] for _ in tuple_weights])



#===============================================
#Random initial vs final
#===============================================

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

incomplete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _==0.]
not_incomplete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _>0.]
not_complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _<1.]
not_incomplete_complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _>0. and _<1.]
complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _==1.]

L=float(len(sequence_all[0][0]))

fig1, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,figsize=[7.5,5.0])
#===============================================
#Fitness

A_all=scale_points(np.sum(initial_structures[incomplete_pos,:2],1),np.sum(final_structures[incomplete_pos,:2],1))

ax1.plot([-2,0.],[-2,0.],'k',zorder=-1)
ax1.scatter(A_all[:,0]/L,A_all[:,1]/L,s=2*A_all[:,2],zorder=0)

A_pl=scale_points(np.sum(initial_structures[complete_pos,:2],1),np.sum(final_structures[complete_pos,:2],1))
ax1.scatter(A_pl[:,0]/L,A_pl[:,1]/L,s=2*A_pl[:,2],zorder=-1,c='C3')

A_pl=scale_points(np.sum(initial_structures[not_incomplete_complete_pos,:2],1),np.sum(final_structures[not_incomplete_complete_pos,:2],1))
ax1.scatter(A_pl[:,0]/L,A_pl[:,1]/L,s=2*A_pl[:,2],zorder=1)

ax1.set_xlim([-2,0.])
ax1.set_ylim([-1,0.6])
ax1.set_xlabel(r'Initial Fitness / $L$',fontsize=12)
ax1.set_ylabel(r'Final Fitness / $L$',fontsize=12)
ax1.yaxis.set_label_coords(-0.22,0.5)
ax1.annotate('a',[0.05,0.9],xycoords='axes fraction',fontsize=12)

#=========================================================
#Delta F vs path length

delta=np.sum(final_structures[incomplete_pos,:2],1)-np.sum(initial_structures[incomplete_pos,:2],1)
A_all=scale_points(np.array(map(path_length,structure_all[incomplete_pos]))/L,delta/L)
ax2.scatter(A_all[:,0],A_all[:,1],s=2*A_all[:,2],zorder=0)

delta=np.sum(final_structures[not_incomplete_complete_pos,:2],1)-np.sum(initial_structures[not_incomplete_complete_pos,:2],1)
A_all=scale_points(np.array(map(path_length,structure_all[not_incomplete_complete_pos]))/L,delta/L)
ax2.scatter(A_all[:,0],A_all[:,1],s=2*A_all[:,2],zorder=1)

delta=np.sum(final_structures[complete_pos,:2],1)-np.sum(initial_structures[complete_pos,:2],1)
A_all=scale_points(np.array(map(path_length,structure_all[complete_pos]))/L,delta/L)
ax2.scatter(A_all[:,0],A_all[:,1],s=2*A_all[:,2],zorder=0,c='C3')

ax2.set_xlim([0,1])
ax2.set_xlabel(r'Path length / $L$',fontsize=12)
ax2.set_ylabel(r'$\Delta$ Fitness / $L$',fontsize=12)
ax2.yaxis.set_label_coords(-0.22,0.5)
ax2.annotate('b',[0.05,0.9],xycoords='axes fraction',fontsize=12)

#===============================================
#Fitness vs hydro initial

hydro_all=scale_points(map(hydrophobicity,initial_sequences[incomplete_pos]),np.sum(final_structures[incomplete_pos,:2],1))
ax3.scatter(hydro_all[:,0],hydro_all[:,1]/L,s=2*hydro_all[:,2],zorder=0)

hydro_pl=scale_points(map(hydrophobicity,initial_sequences[complete_pos]),np.sum(final_structures[complete_pos,:2],1))
ax3.scatter(hydro_pl[:,0],hydro_pl[:,1]/L,s=2*hydro_pl[:,2],zorder=1,c='C3')

hydro_pl=scale_points(map(hydrophobicity,initial_sequences[not_incomplete_complete_pos]),np.sum(final_structures[not_incomplete_complete_pos,:2],1))
ax3.scatter(hydro_pl[:,0],hydro_pl[:,1]/L,s=2*hydro_pl[:,2],zorder=3)

#ax3.set_xlim([0.4,1.8])
ax3.set_ylim([-1,0.6])
ax3.set_xlabel(r'Initial Hydrophobicity',fontsize=12)
ax3.set_ylabel(r'Final Fitness / $L$',fontsize=12)
ax3.yaxis.set_label_coords(-0.22,0.5)
ax3.annotate('c',[0.05,0.9],xycoords='axes fraction',fontsize=12)

#===============================================
#Fitness vs hydro final

hydro_all=scale_points(map(hydrophobicity,final_sequences[incomplete_pos]),np.sum(final_structures[incomplete_pos,:2],1))

#ax4.plot([0,1],[0,1],'k',zorder=-2)
ax4.scatter(hydro_all[:,0],hydro_all[:,1]/L,s=2*hydro_all[:,2],zorder=0)

hydro_pl=scale_points(map(hydrophobicity,final_sequences[complete_pos]),np.sum(final_structures[complete_pos,:2],1))
ax4.scatter(hydro_pl[:,0],hydro_pl[:,1]/L,s=2*hydro_pl[:,2],zorder=0,c='C3')

hydro_pl=scale_points(map(hydrophobicity,final_sequences[not_incomplete_complete_pos]),np.sum(final_structures[not_incomplete_complete_pos,:2],1))
ax4.scatter(hydro_pl[:,0],hydro_pl[:,1]/L,s=2*hydro_pl[:,2],zorder=1)

ax4.set_ylim([-1,0.6])
ax4.set_xlabel(r'Final Hydrophobicity',fontsize=12)
ax4.set_ylabel(r'Final Fitness / $L$',fontsize=12)
ax4.yaxis.set_label_coords(-0.22,0.5)
ax4.annotate('d',[0.05,0.9],xycoords='axes fraction',fontsize=12)

#===============================================
#Hydrophobicity
   
hydro_all=scale_points(map(hydrophobicity,initial_sequences[incomplete_pos]),map(hydrophobicity,final_sequences[incomplete_pos]))

ax5.plot([0,1],[0,1],'k',zorder=-2)
ax5.scatter(hydro_all[:,0],hydro_all[:,1],s=2*hydro_all[:,2],zorder=0)

hydro_pl=scale_points(map(hydrophobicity,initial_sequences[complete_pos]),map(hydrophobicity,final_sequences[complete_pos]))
ax5.scatter(hydro_pl[:,0],hydro_pl[:,1],s=2*hydro_pl[:,2],zorder=0,c='C3')

hydro_pl=scale_points(map(hydrophobicity,initial_sequences[not_incomplete_complete_pos]),map(hydrophobicity,final_sequences[not_incomplete_complete_pos]))
ax5.scatter(hydro_pl[:,0],hydro_pl[:,1],s=2*hydro_pl[:,2],zorder=1,alpha=.7)

ax5.set_xlim([0.3,1.])
ax5.set_ylim([0.3,1.])
ax5.set_xlabel(r'Initial Hydrophobicity',fontsize=12)
ax5.set_ylabel(r'Final Hydrophobicity', fontsize=12)
ax5.yaxis.set_label_coords(-0.22,0.5)
ax5.annotate('e',[0.05,0.9],xycoords='axes fraction',fontsize=12)

#===============================================
#Clustering

ax6.scatter(map(lambda x: dispersion_all_phases(x,3),initial_sequences[incomplete_pos]),map(lambda x: dispersion_all_phases(x,3),final_sequences[incomplete_pos]),s=2.)
ax6.scatter(map(lambda x: dispersion_all_phases(x,3),initial_sequences[complete_pos]),map(lambda x: dispersion_all_phases(x,3),final_sequences[complete_pos]),c='r',s=2.)
ax6.scatter(map(lambda x: dispersion_all_phases(x,3),initial_sequences[not_incomplete_complete_pos]),map(lambda x: dispersion_all_phases(x,3),final_sequences[not_incomplete_complete_pos]),s=2.,alpha=0.7)

ax6.plot([0,1.6],[0,1.6],'k')
ax6.set_xlabel(r'Initial Clustering',fontsize=12)
ax6.set_ylabel(r'Final Clustering',fontsize=12)
ax6.yaxis.set_label_coords(-0.22,0.5)
ax6.annotate('f',[0.05,0.9],xycoords='axes fraction',fontsize=12)
#ax6.set_yticklabels(['0.0','','0.5','','1.0','','1.5'])

plt.tight_layout(w_pad=-.5)


#===========================================================
#Mazes
#===========================================================

with open ('fold_degeneracy_properties_fixed','r') as fin:
    sequence_all=cPickle.load(fin)
    structure_all=cPickle.load(fin)
    chance_complete_initial=cPickle.load(fin)
    chance_complete_final=cPickle.load(fin)
    mutations=cPickle.load(fin)

structure_all=np.array(structure_all)
sequence_all=np.array(sequence_all)
mutations=map(np.array,mutations)

initial_sequences=np.array([_[0] for _ in sequence_all])
initial_structures=np.array([_[0] for _ in structure_all])
final_sequences=np.array([_[-1] for _ in sequence_all])
final_structures=np.array([_[-1] for _ in structure_all])

incomplete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _==0.]
not_incomplete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _>0.]
not_complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _<1.]
not_incomplete_complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _>0. and _<1.]
complete_pos=[pos for pos,_ in enumerate(chance_complete_final) if _==1.]

L=float(len(sequence_all[0][0]))

fig2, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,figsize=[3.,6.],dpi=300)
#===========================================================
#Delt F vs Hamming

delta=np.sum(final_structures[incomplete_pos,:2],1)-np.sum(initial_structures[incomplete_pos,:2],1)
#A_all=scale_points(np.array(map(path_length,structure_all[incomplete_pos]))/L,delta/L)
A_all=scale_points(np.sum((initial_sequences - final_sequences)**2,1)[incomplete_pos]/L,delta/L)
ax1.scatter(A_all[:,0],A_all[:,1],s=2*A_all[:,2],zorder=0)

delta=np.sum(final_structures[not_incomplete_complete_pos,:2],1)-np.sum(initial_structures[not_incomplete_complete_pos,:2],1)
#A_all=scale_points(np.array(map(path_length,structure_all[not_incomplete_pos]))/L,delta/L)
A_all=scale_points(np.sum((initial_sequences - final_sequences)**2,1)[not_incomplete_complete_pos]/L,delta/L)
ax1.scatter(A_all[:,0],A_all[:,1],s=2*A_all[:,2],zorder=0)

delta=np.sum(final_structures[complete_pos,:2],1)-np.sum(initial_structures[complete_pos,:2],1)
#A_all=scale_points(np.array(map(path_length,structure_all[complete_pos]))/L,delta/L)
A_all=scale_points(np.sum((initial_sequences - final_sequences)**2,1)[complete_pos]/L,delta/L)
ax1.scatter(A_all[:,0],A_all[:,1],s=2*A_all[:,2],zorder=0,c='C3')

#ax2.set_xlim([0,1])
ax1.set_xlabel(r'Hamming distance / $L$',fontsize=12)
ax1.set_ylabel(r'$\Delta$ Fitness / $L$',fontsize=12)
#ax1.yaxis.set_label_coords(-0.22,0.5)
ax1.annotate('a',[0.05,0.9],xycoords='axes fraction',fontsize=12)

#===========================================================
#Hamming plot

ax2.plot([0.25,32/L],[0.25,32/L],'k',zorder=-1)

hamming_pl=scale_points(map(path_length,structure_all[complete_pos]),np.sum((initial_sequences - final_sequences)**2,1)[complete_pos])
ax2.scatter(hamming_pl[:,1]/L,hamming_pl[:,0]/L,s=8*hamming_pl[:,2],c='r',zorder=1)

ax2.set_ylabel(r'Path Length / $L$',fontsize=12)
ax2.set_xlabel(r'Hamming distance / $L$',fontsize=12)
ax2.annotate('b',[0.05,0.9],xycoords='axes fraction',fontsize=12)

#===========================================================
#Hydrophobocity vs time

#data=np.concatenate([zip(range(path_length(structure_all[_])+1),map(hydrophobicity,sequence_all[_][:path_length(structure_all[_])+1])) for _ in complete_pos])
#points=scale_points(data[:,0],data[:,1])
#ax3.plot(data[:,0],data[:,1],c='C3')

for _ in complete_pos:
    ax3.plot(np.array(range(path_length(structure_all[_])+1))-1.,map(hydrophobicity,sequence_all[_][:path_length(structure_all[_])+1]),c='C3',linewidth=1,alpha=0.1)


#data=np.concatenate([zip(range(path_length(structure_all[_])+1),map(hydrophobicity,sequence_all[_][:path_length(structure_all[_])+1])) for _ in incomplete_pos])
#points=scale_points(data[:,0],data[:,1])
#ax3.scatter(points[:,0],points[:,1],s=0.1*points[:,2],c='C0')

for _ in incomplete_pos:
    ax3.plot(range(path_length(structure_all[_])+1),map(hydrophobicity,sequence_all[_][:path_length(structure_all[_])+1]),c='C0',linewidth=1.,alpha=0.1)

ax3.set_xlabel(r'Substitution number',fontsize=12)
ax3.set_ylabel(r'Hydrophobicity',fontsize=12)
ax3.annotate('c',[0.05,0.9],xycoords='axes fraction',fontsize=12)

fig2.tight_layout()


#===========================================================
#Backtracking
#===========================================================

long_path_pos=[_ for pos,_ in enumerate(complete_pos) if path_length(structure_all[_])>60]
pos=long_path_pos[0]
chance_complete_example=map(lambda x: chance_complete(x,1000),sequence_all[pos][:path_length(structure_all[pos])+1])
position_data=mutation_positions(sequence_all[pos])
structure_data=np.array(structure_all[pos])[:,:-1]
structure_data[:,1]=-structure_data[:,1]

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch

fig3 = plt.figure(figsize=[6,5],constrained_layout=True,dpi=300)
gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig3)

ax4=fig3.add_subplot((gs[:-1,0]))
#ax4.plot(map(hydrophobicity,sequence_all[pos]))
#im=ax4.imshow(np.arctan(mutations[pos])/np.arctan(1e6),extent=[2,58,69,1])
strong_beneficials=np.array([map(lambda x: 1 if x>=1 else (-1 if x<=-1 else 0),spectrum) for spectrum in mutations[pos]])
#im=ax4.imshow(strong_beneficials,extent=[2,58,69,1])
ax4.plot(position_data+1.5,np.arange(1.5,len(position_data)+1.5),'.k',markersize=2)
im=ax4.imshow(strong_beneficials,extent=[1,59,68,1])
ax4.plot([36.,36],[1.,68],c='white',linewidth=1)
ax4.plot([37.1,37.1],[1.,68],c='white',linewidth=1)
ax4.set_xlabel(r'Sequence position',fontsize=12)
ax4.set_ylabel(r'Substitution number',fontsize=12)
ax4.set_title(r'Beneficial mutations')
ax4.xaxis.set_label_coords(0.5,-0.1)
#axins = inset_axes(ax4,
#                   width="100%",  # width = 5% of parent_bbox width
#                   height="5%",  # height : 50%
#                   loc='lower center',
#                   bbox_to_anchor=(0, -0.25, 1, 1),
#                   bbox_transform=ax4.transAxes,
#                   borderpad=0,
#                   )
#fig3.colorbar(im, cax=axins, orientation="horizontal",ticks=[0])
#ax4.annotate(r'$-$',[0.2,-0.32],xycoords='axes fraction',fontsize=12)
#ax4.annotate(r'$+$',[0.8,-0.32],xycoords='axes fraction',fontsize=12)

ax4=fig3.add_subplot((gs[:-1,1]))
im=ax4.imshow(np.array(sequence_all[pos])[1:-1],extent=[1,59,68,1],cmap='binary')
ax4.set_xlabel(r'Sequence position',fontsize=12)
ax4.xaxis.set_label_coords(0.5,-0.1)

legend_elements = [Patch(facecolor='k', edgecolor='k',label='Hydrophobic'),
                   Patch(facecolor='white', edgecolor='k',label='Polar')]

ax4.legend(handles=legend_elements, ncol=2, loc='upper center',fontsize=10,bbox_to_anchor=(0.5, 1.1),frameon=False)

ax1=fig3.add_subplot(gs[2,0])
ax1.plot(structure_data[:,0]/L,label=r'$\overline{S}/L$')
ax1.plot(structure_data[:,1]/L,label=r'$\overline{A}/L$')
ax1.plot(chance_complete_example,'k',label=r'$\%$ Complete')
ax1.set_ylabel(r'$\overline{S}/L$, $\overline{A}/L$',fontsize=12)
ax1.set_xticklabels([])
ax1.legend(ncol=2,fontsize=6,loc="upper center",framealpha=0.5)
ax1.set_xlabel(r'Substitution number',fontsize=12)

ax3=fig3.add_subplot(gs[2,1])
ax3.plot(map(hydrophobicity,sequence_all[pos]))
ax3.set_ylabel(r'Hydrophobicity',fontsize=12)
ax3.set_ylim([0.5,1])
ax3.set_yticklabels(['.5','','1'])
ax3.set_xlabel(r'Substitution number',fontsize=12)

plt.tight_layout(h_pad=-1.)
plt.tight_layout()
