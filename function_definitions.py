#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:12:13 2019

@author: jbertram
"""

from scipy.sparse import csgraph
import itertools
import numpy as np
import matplotlib.pyplot as plt

def initial_fold(sequence):
    return np.concatenate([sequence[:1],[sequence[position] for position in xrange(1,len(sequence)-2) if any(sequence[position-1:position+2]-np.array([1,0,1]))],sequence[-1:]])

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

def dispersion_all_phases(sequence,window):
    length=float(len(sequence))

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

def mean_runlength(sequence):
    runlengths=run_lengths_phobic(sequence)
    return np.sum([(i+1)*runlengths[i] for i in range(len(runlengths))])/np.sum(runlengths)

def mutate(sequence,position):
    temp=np.array(sequence)
    if temp[position]==1:
        temp[position]=0
    else:
        temp[position]=1
    return temp

def generate_initial_sequence(L,H):
    temp=np.array(L*[0])
    #1 for hydrophobic, 0 for polar
    temp[np.random.choice(L,size=int(H*L),replace=False)]=1
    return temp

def neighbors(location):
    return location+np.array([[0,1],[0,-1],[-1,0],[1,0]])

#all paths between start and end consisting of length steps
def routes(start,end,length):
    diff=end-start
    distance=np.sum(np.abs(diff))
    out=[]
    if length==distance:
        stepx=np.array([np.sign(diff[0]),0])
        stepy=np.array([0,np.sign(diff[1])])

        for _ in itertools.combinations(xrange(length),int(abs(diff[1]))):
            steps=np.array(length*[stepx])
            steps[list(_)]=stepy
            out.append(steps)

    elif length>distance:
        longstep=np.array([1,1])-np.abs(diff)

        for _ in [-1,1]:
            steps=(length-1)/2*[_*longstep]+[diff]+(length-1)/2*[-1*_*longstep]
            out.append(steps)

    return np.array(out)

#for a given candidate contact, 
#returns a valid conformation of the unzipped strand(s), if one exists
def unzipped_conformations(contact,occupied_locations,locations,zipped):
    leftbase=min(zipped)
    rightbase=max(zipped)
    valid_paths=[]
    #branches sticking together
    if contact[0] not in zipped and contact[1] not in zipped:
        orientation=np.array([1,1])-np.abs(locations[leftbase]-locations[rightbase])

        possible_receiving_locations=np.array([[locations[leftbase]+_*orientation,locations[rightbase]+_*orientation] for _ in [-1,1]])
        receiving_locations=np.array([_ for _ in possible_receiving_locations if occupied_locations[_[0,0],_[0,1]]==-1 and occupied_locations[_[1,0],_[1,1]]==-1])

        if len(receiving_locations)>0:
            return {'left':[receiving_locations[0,0]],'right':[receiving_locations[0,1]]}
        else:
            return {'left':[],'right':[]}
    #one branch sticking to the zipped structure
    else:
        if contact[1] in zipped:
            attaching_H=contact[0]
            receiving_H=contact[1]
            base=leftbase
            otherbase=rightbase
            code="left"
        else:
            attaching_H=contact[1]
            receiving_H=contact[0]
            base=rightbase
            otherbase=leftbase
            code="right"
            
        #check where other branch is to make sure it isn't blocked
        otherbranch_locations=np.array([_ for _ in neighbors(locations[otherbase]) if occupied_locations[_[0],_[1]]==-1])
        if len(otherbranch_locations)==1:
            otherbranch_constrained=True
            otherbranch_location=otherbranch_locations[0]          
        else:
            otherbranch_constrained=False

        receiving_locations=np.array([_ for _ in neighbors(locations[receiving_H]) if occupied_locations[_[0],_[1]]==-1])
        valid_paths=[]
        for end in receiving_locations:
            for route in routes(locations[base],end,np.abs(base-attaching_H)):
                path=[locations[base]]
                valid=True

                for step in route:
                    next_location=path[-1]+step
                    if occupied_locations[next_location[0],next_location[1]]>=0 or (otherbranch_constrained and not any(otherbranch_location-next_location)):
                            valid=False
                            break

                    path=path+[next_location]

                if valid==True:
                    valid_paths.append(path[1:])

        if len(valid_paths)==2:
            path_choice=np.argmin([np.sum((_[-1]-locations[otherbase])**2) for _ in valid_paths])
            valid_paths=valid_paths[path_choice]
        elif len(valid_paths)==1:
            valid_paths=valid_paths[0]

        if code=='left':
            return {'left':valid_paths,'right':[]}
        else:
            return {'left':[],'right':valid_paths}

#zip for a given nucleation until zipper gets stuck
def HPzip(sequence,nucleation_contact):
    L=len(sequence)
    #track number of contacts for each HP residue
    contacts_HH={position:0 for position,residue in enumerate(sequence) if residue==1}
    contacts_all={position:0 for position,residue in enumerate(sequence)}

    #construct directed adjacency matrix for contact graph
    contact_graph=np.array([[int(y==x+1) for y in range(L)] for x in range(L)])
    contact_graph[nucleation_contact[0],nucleation_contact[1]]=1.
    
    contacts_HH[nucleation_contact[0]]=contacts_HH[nucleation_contact[0]]+1
    contacts_HH[nucleation_contact[1]]=contacts_HH[nucleation_contact[1]]+1
    contacts_all[nucleation_contact[0]]=contacts_all[nucleation_contact[0]]+1
    contacts_all[nucleation_contact[1]]=contacts_all[nucleation_contact[1]]+1
    
    #track zipped H residues
    zipped=set(range(nucleation_contact[0],nucleation_contact[1]+1))

    #array of occupied locations
    occupied_locations=np.array(2*L*[2*L*[-1]])
    occupied_locations[L,L]=nucleation_contact[0]
    occupied_locations[L,L+1]=nucleation_contact[0]+1
    occupied_locations[L+1,L+1]=nucleation_contact[0]+2
    occupied_locations[L+1,L]=nucleation_contact[1]

    #spatial location of each zipped residue
    locations={nucleation_contact[0]:np.array([L,L]),
               nucleation_contact[0]+1:np.array([L,L+1]),
               nucleation_contact[0]+2:np.array([L+1,L+1]),
               nucleation_contact[1]:np.array([L+1,L])}
    
    #count of HH contacts
    contact_count=1
    while 1:
        #matrix of shortest paths from the Dijkstra algorithm
        shortest_paths=csgraph.dijkstra(contact_graph)
         
        candidate_contacts=np.array([[x,y] for (x,y),z in np.ndenumerate(shortest_paths) 
            if z==3. and x>=min(zipped)-3 and y<=max(zipped)+3 #effective contact order of 3 and no new nucleations
            and all(sequence[[x,y]]) and (x not in zipped or y not in zipped) #both H and at least one unzipped  
            and (contacts_all[x]<=1+int(x==L-1)+int(x==0) and contacts_all[y]<=1+int(y==L-1)+int(y==0))]) #not full

        #omit contacts that are incompatible with existing contacts
        possible_contacts={}
        for position,contact in enumerate(candidate_contacts):
            valid_conformation=unzipped_conformations(contact,occupied_locations,locations,zipped)
            if len(valid_conformation['left']) or len(valid_conformation['right'])>0:
                possible_contacts[position]=valid_conformation

        if len(possible_contacts)>0:
            new_contact=possible_contacts.keys()[np.random.randint(len(possible_contacts))]

            leftbase=min(zipped)
            rightbase=max(zipped)
            increment_rule={'left':[leftbase,-1],'right':[rightbase,1]}
            for branch in ['left','right']:
                position=increment_rule[branch][0]
                for _ in possible_contacts[new_contact][branch]:
                    position=position+increment_rule[branch][1]
                    occupied_locations[_[0],_[1]]=position
                    locations[position]=_
                    zipped.add(position)
                    
                    for location in neighbors(_):
                        neighbor_position=occupied_locations[location[0],location[1]]
                        if neighbor_position>=0 and np.abs(neighbor_position-position)>1:
                            contacts_all[position]=contacts_all[position]+1
                            contacts_all[neighbor_position]=contacts_all[neighbor_position]+1
                            
                            if all(sequence[[position,neighbor_position]]):
                                    contact_graph[min([position,neighbor_position]),max([position,neighbor_position])]=1.
                
                                    contacts_HH[position]=contacts_HH[position]+1
                                    contacts_HH[neighbor_position]=contacts_HH[neighbor_position]+1
                                    
                                    contact_count=contact_count+1
                        
        else:
            break

    return contact_count,contacts_all,contacts_HH,locations


def zipped_structure(sequence):
    working_sequence=np.array(sequence)
    contact_count_global=0
    H_exposure_global={position:2+int(position==len(sequence)-1)+int(position==0) for position,residue in enumerate(sequence) if residue==1}
    zipped_global=set({})
    while 1:
        possible_nucleations=np.array([[position,position+3] for position,_ in enumerate(working_sequence[:-3])
            if all(working_sequence[[position,position+3]]) and len({position,position+3}.intersection(zipped_global))==0])
    
        if len(possible_nucleations)==0:
            break
        
        nucleation_contact=possible_nucleations[np.random.randint(len(possible_nucleations))]
        
        contact_count,contacts_all,contacts_HH,locations=HPzip(working_sequence,nucleation_contact)
        contact_count_global=contact_count_global+contact_count
        zipped_global=zipped_global.union(set(locations.keys()))
        working_sequence[locations.keys()]=0
        
        contact_count_global=contact_count_global+contact_count
        for _ in contacts_HH:
            H_exposure_global[_]=H_exposure_global[_]-contacts_all[_]
                
    return contact_count_global, np.sum(H_exposure_global.values())
            
def F(sequence,sample_size):
    contact_counts=np.zeros(sample_size)
    exposure_counts=np.zeros(sample_size)
    for _ in xrange(sample_size):
        contact_counts[_],exposure_counts[_]=zipped_structure(sequence)
    
    return np.array([np.mean(contact_counts),np.mean(exposure_counts)])

def plot_folded_structure(sequence,locations):
    fold_graph=np.array([locations[_] for _ in range(len(sequence)) if _ in locations])
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.plot(fold_graph[:,0],fold_graph[:,1],'k')
    col_map={0:'w',1:'k'}
    for _ in locations:
        plt.text(locations[_][0]+0.05,locations[_][1]+0.05,str(_),fontsize=14)
        plt.plot(locations[_][0],locations[_][1],'o',color=col_map[sequence[_]],markersize=10,markeredgecolor='k')

    plt.show()
    
    return



#==========================================
#old fudges
#==========================================
#def F(sequence):
#    H=np.sum(sequence)
#    return (1/H)*np.sum([np.sum([1./np.abs(posx-posy)**3 for posx,x in enumerate(sequence) if x==1 and np.abs(posx-posy)>1]) for posy,y in enumerate(sequence) if y==1])

#def A(sequence):
#    runlengths=run_lengths_phobic(initial_fold(sequence))
#    #runlengths=run_lengths_phobic(sequence)
#    return np.sum([(i+1)**2*runlengths[i] for i in range(len(runlengths))])/np.sum(runlengths)

