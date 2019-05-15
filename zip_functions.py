#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:12:13 2019

@author: jbertram
"""

from scipy.sparse import csgraph
from scipy.sparse import diags
import itertools
import numpy as np
from collections import deque

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

def generate_initial_sequence_connected(L):
    temp=np.array([1]+(L-2)*[0]+[1])
    temp[1]=np.random.randint(2)
    for _ in range(2,L-1):
        if not any(temp[_-2:_]):
            temp[_]=1
        else:
            temp[_]=np.random.randint(2)

    return temp

def neighbors(location):
    return location+np.array([[0,1],[0,-1],[-1,0],[1,0]])

#all paths between start and end consisting of length steps
def routes(start,end,length):
    diff=end-start
    distance=np.sum(np.abs(diff))
    out=deque()
    if length==distance:
        stepx=np.array([np.sign(diff[0]),0])
        stepy=np.array([0,np.sign(diff[1])])

        for _ in itertools.combinations(xrange(length),int(np.abs(diff[1]))):
            steps=np.array(length*[stepx])
            steps[list(_)]=stepy
            out.append(steps)

    elif length>distance:
        longstep=np.array([1,1])-np.abs(diff)

        for _ in [-1,1]:
            steps=(length-1)/2*[_*longstep]+[diff]+(length-1)/2*[-1*_*longstep]
            out.append(steps)

    return out

#for a given candidate contact, 
#returns a valid conformation of the unzipped strand(s), if one exists
def unzipped_conformations(contact,occupied_locations,locations,leftbase,rightbase,contacts_all,left_end,right_end):
    valid_paths=deque()
    #branches sticking together
    if contact[0]<leftbase and contact[1]>rightbase:
        orientation=np.array([1,1])-np.abs(locations[leftbase]-locations[rightbase])

        possible_receiving_locations=np.array([[locations[leftbase]+_*orientation,locations[rightbase]+_*orientation] for _ in [-1,1]])
        receiving_locations=np.array([_ for _ in possible_receiving_locations if occupied_locations[_[0,0],_[0,1]]==-1 and occupied_locations[_[1,0],_[1,1]]==-1])

        if len(receiving_locations)>0:
            return {'left':[receiving_locations[0,0]],'right':[receiving_locations[0,1]]}
        else:
            return {'left':[],'right':[]}
    #one branch sticking to the zipped structure
    else:
        if contact[0]<leftbase:
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
        if contacts_all[otherbase]==2 and otherbase!=left_end and otherbase!=right_end:
            otherbranch_locations=np.array([_ for _ in neighbors(locations[otherbase]) if occupied_locations[_[0],_[1]]==-1])
            otherbranch_constrained=True
            otherbranch_location=otherbranch_locations[0]    
        else:
            otherbranch_constrained=False

        receiving_locations=np.array([_ for _ in neighbors(locations[receiving_H]) if occupied_locations[_[0],_[1]]==-1])
        valid_paths=[]
        for end in receiving_locations:
            if len([_ for _ in neighbors(end) if occupied_locations[_[0],_[1]]==-1]) \
                >1-int(attaching_H==right_end)-int(attaching_H==left_end)-int(np.abs(attaching_H-base)==1):
                for route in routes(locations[base],end,np.abs(base-attaching_H)):
                    path=deque([locations[base]])
                    valid=True
    
                    for step in route:
                        next_location=path[-1]+step
                        if occupied_locations[next_location[0],next_location[1]]>=0 or (otherbranch_constrained and not any(otherbranch_location-next_location)):
                                valid=False
                                break
    
                        path.append(next_location)
    
                    if valid==True:
                        path.popleft()
                        valid_paths.append(path)
                 
        valid_paths=np.array(valid_paths)
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
def HPzip(sequence,nucleation_contact,total_length,left_bound):
    L=len(sequence)
    #track number of contacts for each HP residue
    contacts_all={position:0 for position,residue in enumerate(sequence)}
    contacts_all[nucleation_contact[0]]=1
    contacts_all[nucleation_contact[1]]=1

    #construct directed adjacency matrix for contact graph
    contact_graph=diags((L-1)*[1],1,format='lil',dtype=np.int8)
    contact_graph[nucleation_contact[0],nucleation_contact[1]]=1
    
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
    
    if left_bound==0:
        left_end=0
    else:
        left_end=-1
        
    right_end=total_length-left_bound-1
    
    Hpositions=[position for position,residue in enumerate(sequence) if residue==1]
    
    #count of HH contacts
    contact_count=1
    while 1:
        leftbase=min(zipped)
        rightbase=max(zipped)
        candidate_nodes=[_ for _ in Hpositions if _>=leftbase-3 and _<=rightbase #no new nucleations
                         and contacts_all[_]<=1+int(_==right_end)+int(_==left_end)] #not full
        
        #matrix of shortest paths
        shortest_paths=csgraph.dijkstra(contact_graph,indices=candidate_nodes)
        
        candidate_contacts=np.array([[candidate_nodes[x],y] for (x,y),z in np.ndenumerate(shortest_paths) 
            if z==3. and (candidate_nodes[x] not in zipped or y not in zipped) and sequence[y]==1]) #effective contact order of 3 and at least one unzipped
        
        #omit contacts that are incompatible with existing contacts
        possible_contacts={}
        for position,contact in enumerate(candidate_contacts):
            valid_conformation=unzipped_conformations(contact,occupied_locations,locations,leftbase,rightbase,contacts_all,left_end,right_end)
            if len(valid_conformation['left'])>0 or len(valid_conformation['right'])>0:
                possible_contacts[position]=valid_conformation

        if len(possible_contacts)>0:
            new_contact=possible_contacts.keys()[np.random.randint(len(possible_contacts))]

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
                                    contact_graph[min([position,neighbor_position]),max([position,neighbor_position])]=1                    
                                    contact_count=contact_count+1
        else:
            break

    return contact_count,contacts_all,zipped,locations


def zipped_structure(sequence,possible_nucleations,nucleation_position):
    contact_count_global=0
    zipped_global=set()
    unzipped_nucleation_positions=range(len(possible_nucleations))
    H_exposure_global={position:2+int(position==len(sequence)-1)+int(position==0) for position,residue in enumerate(sequence) if residue==1}
    while 1:
        nucleation_contact=np.array([possible_nucleations[nucleation_position],possible_nucleations[nucleation_position]+3])
        #only pass unzipped fragments to HPzip for computational efficiency
        left_zipped=[_ for _ in zipped_global if _<nucleation_contact[0]]
        if len(left_zipped)>0:
		left_bound=max(left_zipped)+1
        else:
		left_bound=0

        right_zipped=[_ for _ in zipped_global if _>nucleation_contact[1]]
        if len(right_zipped)>0:
		right_bound=min(right_zipped)
        else:
		right_bound=len(sequence)+1
    
        working_sequence=sequence[left_bound:right_bound]
        
        contact_count,contacts_all,zipped,_=HPzip(working_sequence,nucleation_contact-left_bound,len(sequence),left_bound)
              
        for _ in zipped:
            zipped_global.add(_+left_bound)
        
        min_zipped=min(zipped)
        max_zipped=max(zipped)
        left_zipped_nucleation=len([_ for _ in unzipped_nucleation_positions if possible_nucleations[_]+3<min_zipped+left_bound])
        right_zipped_nucleation=len([_ for _ in unzipped_nucleation_positions if possible_nucleations[_]<=max_zipped+left_bound])
      
        unzipped_nucleation_positions=unzipped_nucleation_positions[:left_zipped_nucleation]+unzipped_nucleation_positions[right_zipped_nucleation:]
        
        for _,residue in enumerate(working_sequence):
            if residue==1:
                H_exposure_global[_+left_bound]=H_exposure_global[_+left_bound]-contacts_all[_]
        
        if len(zipped_global)==len(sequence):
            contact_count_global=contact_count_global+contact_count
                
        num_nucleations=len(unzipped_nucleation_positions)
        
        if num_nucleations==0:
            break
    
        nucleation_position=unzipped_nucleation_positions[np.random.randint(num_nucleations)]
                
    return contact_count_global, np.sum(H_exposure_global.values()),len(zipped_global)/float(len(sequence))
            
def F(sequence,alpha,sample_size):
    triplet_totals=[np.sum(sequence[position:position+3]) for position in range(len(sequence)-3)]
    if min(triplet_totals)==0:
        return np.array([-len(sequence),-len(sequence),-len(sequence)])
    else:
        contact_counts=np.zeros(sample_size)
        exposure_counts=np.zeros(sample_size)
        percent_ordered=np.zeros(sample_size)
        possible_nucleations=[[position for position,_ in enumerate(sequence[:-3]) if all(sequence[[position,position+3]])][0]]
        #cycle through initating nucleation contacts to reduce sample variance
        count=0
        for _ in itertools.cycle(range(len(possible_nucleations))):
            if count==sample_size:
                break
            contact_counts[count],exposure_counts[count],percent_ordered[count]=zipped_structure(sequence,possible_nucleations,_)
            count=count+1
        
        if len(possible_nucleations)==0:
            exposure_counts=exposure_counts+sum([2+int(position==len(sequence)-1)+int(position==0) for position,residue in enumerate(sequence) if residue==1])
    
        exposure_counts=-alpha*exposure_counts
        return np.array([np.mean(contact_counts),np.mean(exposure_counts),-len(sequence)*(1-np.mean(percent_ordered))])


