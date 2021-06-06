# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:40:11 2021

@author: cedri
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

df = pd.read_csv('bioactivity_preprocessed_data.csv')

'''
Calculate Lipinski descriptors
    . Molecular weight < 500 Dalton
    . Octanol-water partition coefficient (LogP) < 5
    . Hydrogen bond donors < 5
    . Hydrogen bond acceptors < 10
'''

def lipinski(smiles):
    
    moldata = list()
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)
        
    baseData = np.arange(1,1)
    i = 0
    for mol in moldata:
        
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
        
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])
        
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        
        i = i + 1
        
    columnNames = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
    descriptors = pd.DataFrame(data = baseData, columns = columnNames)
    
    return descriptors

df_lipinski = lipinski(df.canonical_smiles)

df_combined = pd.concat([df,df_lipinski], axis=1)

def IC50_to_pIC50(serie):
    pIC50 = list()
    
    for val in serie['standard_value']:
        molar = val*(10**-9) #nM to M
        pIC50.append(-np.log10(molar))
        
    serie['pIC50'] = pIC50
    x = serie.drop('standard_value', axis = 1)
    
    return x

df_norm = df_combined.copy()
df_norm['standard_value'] = df_norm['standard_value'].clip(None, 100000000)

df_final = IC50_to_pIC50(df_norm)

#Simplifying the problem for a first step by binarizing the target
df_2class = df_final[df_final.bioactivity_label != 'intermediate']

'''
EDA ...
'''