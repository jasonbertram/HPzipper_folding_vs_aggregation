# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:34:20 2019

@author: jason
"""
import numpy as np
from analysis_functions import *
import cPickle
import os

#computes chance to form a complete fold and aggregates output files into a single pickle


L_all=[]
sequence_all=[]
structure_all=[]
mutations=[]
for filename in os.listdir('.'):
    if filename[:9]=='output0.0':
        with open(filename,'r') as fin:
            L_all.append(cPickle.load(fin))
            sequence_all.append(cPickle.load(fin))
            structure_all.append(cPickle.load(fin))
            mutations.append(cPickle.load(fin))
            
initial_sequences=np.array([_[0] for _ in sequence_all])
initial_structures=np.array([_[0] for _ in structure_all])
final_sequences=np.array([_[-1] for _ in sequence_all])
final_structures=np.array([_[-1] for _ in structure_all])


chance_complete_initial=np.array(map(lambda x: chance_complete(x,1000),initial_sequences))
chance_complete_final=np.array(map(lambda x: chance_complete(x,1000),final_sequences))

with open ('fold_degeneracy_properties_random_0.0','w') as fout:
    cPickle.dump(sequence_all,fout)
    cPickle.dump(structure_all,fout)
    cPickle.dump(chance_complete_initial,fout)
    cPickle.dump(chance_complete_final,fout)
    cPickle.dump(mutations,fout)
