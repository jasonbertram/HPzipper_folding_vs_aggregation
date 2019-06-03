# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:46:21 2019

@author: jason
"""
import numpy as np
import matplotlib.pyplot as plt
from zip_functions import *

def initial_fold(sequence):
    return np.concatenate([sequence[:1],[sequence[position] for position in xrange(1,len(sequence)-2) if any(sequence[position-1:position+2]-np.array([1,0,1]))],sequence[-1:]])

def dispersion_all_phases(sequence,window):
    length=float(len(sequence))
    sequence=2*sequence-np.ones(len(sequence))
    dispersion=0
    for phase in range(window):
        numblocks=np.floor((length-phase)/window)
        L=window*numblocks
        block_sums=np.array([np.sum(sequence[phase+window*_:phase+window*(_+1)]) for _ in range(int(numblocks))])
        M=np.sum(block_sums)
        if L**2==M**2 or numblocks<=1:
            dispersion=0
        else:
            dispersion=dispersion+(L-1)/((L**2-M**2)*(1-1/numblocks))*sum((block_sums-M/numblocks)**2)

    return dispersion/window

def run_lengths_phobic(sequence_hydro):
    counts=np.zeros(len(sequence_hydro))
    first_hydrophobic=next(x for x,y in enumerate(sequence_hydro) if y==1)
    prev=1
    running_count=0
    phobicity_flag=1
    for amino in sequence_hydro[first_hydrophobic+1:]:
        if amino != prev:
            if phobicity_flag>0:
                counts[running_count]=counts[running_count]+1
            phobicity_flag=-1*phobicity_flag
            running_count=0
        else:
            running_count=running_count+1
        prev=amino
    if phobicity_flag>0:
        counts[running_count]=counts[running_count]+1
    return counts

def hydrophobicity(sequence):
    return np.sum(sequence)/float(len(sequence))

def mean_runlength(sequence):
    runlengths=run_lengths_phobic(sequence)
    return np.sum([(i+1)*runlengths[i] for i in range(len(runlengths))])/np.sum(runlengths)

#normalized mean hydrophobic run length
def mean_runlength_normalized(sequence):
    hp=hydrophobicity(sequence)
    total_hydro=len(sequence)*hp
    if total_hydro==0:
        return 0.
    counts=run_lengths_phobic(sequence)
    meanRL=total_hydro/sum(counts)
    if hp==1:
        norm=len(sequence)
    else:
        norm=1/(1-hp)
    return meanRL/norm

def plot_folded_structure(sequence):
    possible_nucleations=[[position for position,_ in enumerate(sequence[:-3]) if all(sequence[[position,position+3]])][0]]
    for _ in range(len(possible_nucleations)):
        nucleation_contact=np.array([possible_nucleations[_],possible_nucleations[_]+3])
        a,b,c,locations=HPzip(sequence,nucleation_contact,len(sequence),0)
        print len(c)
        fold_graph=np.array([locations[_] for _ in range(len(sequence)) if _ in locations])
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.plot(fold_graph[:,0],fold_graph[:,1],'k')
        col_map={0:'w',1:'k'}
        plt.plot([len(sequence)],[len(sequence)],'r',markersize=10,markeredgecolor='r')
        for _ in locations:
            #plt.text(locations[_][0]+0.05,locations[_][1]+0.05,str(_),fontsize=14)
            plt.plot(locations[_][0],locations[_][1],'o',color=col_map[sequence[_]],markersize=8,markeredgecolor='k')
            plt.axis('off')

    plt.show()
    
    return

def num_folds(sequence,sample_size):
    structure_summaries=set()
    possible_nucleations=[[position for position,_ in enumerate(sequence[:-3]) if all(sequence[[position,position+3]])][0]]
    #cycle through initating nucleation contacts to reduce sample variance
    count=0
    for _ in itertools.cycle(range(len(possible_nucleations))):
        if count==sample_size:
            break
        contact_counts,exposure_counts,percent_ordered=zipped_structure(sequence,possible_nucleations,_)
        structure_summaries.add((contact_counts,exposure_counts,percent_ordered))
        count=count+1
    
    #print structure_summaries
    if all([_[2]<1. for _ in structure_summaries]):
        return 0
    else:
        return len(structure_summaries)
    
def chance_complete(sequence,sample_size):
    possible_nucleations=[[position for position,_ in enumerate(sequence[:-3]) if all(sequence[[position,position+3]])][0]]
    #cycle through initating nucleation contacts to reduce sample variance
    count=0
    num_complete=0
    for _ in itertools.cycle(range(len(possible_nucleations))):
        if count==sample_size:
            break
        contact_counts,exposure_counts,percent_ordered=zipped_structure(sequence,possible_nucleations,_)
        
        if percent_ordered==1.:
            num_complete=num_complete+1
            
        count=count+1
        
    return num_complete/float(sample_size)

