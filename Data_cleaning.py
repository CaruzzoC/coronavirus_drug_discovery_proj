# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:13:02 2021

@author: cedri
"""

import pandas as pd

df = pd.read_csv('bioactivity_data.csv')

#dropping missing data
df2 = df[df.standard_value.notna()]

'''
The bioactivity data is in the IC50 unit. Compounds having values of less than
1000 nM will be labeled as 'active', those greater than 10 000 nM labeled
as 'inactive' and every other compounds will be labeled 'intermediate'.
'''

def make_labels(df):
    bioactivity_label = list()
    for concentration in df2.standard_value:
        if concentration >= 10000:
            bioactivity_label.append("inactive")
        elif concentration <= 1000:
            bioactivity_label.append("active")
        else:
            bioactivity_label.append("intermediate")
    return bioactivity_label
            
bioactivity_label = make_labels(df2)

selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
df3 = df2[selection]

df3.to_csv('bioactivity_preprocessed_data.csv', index=False)