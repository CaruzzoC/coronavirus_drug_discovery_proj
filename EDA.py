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
import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt


'''
We start by ploting the class distribution. It is the basic step to understand
if we will have to deal with class unbalance (in this case we can already
                                              assume there will be a class
                                              unbalanced problem).
'''
#Count plot Inactive vs Active (bioactivity class)
plt.figure(figsize=(5.5, 5.5))
sns.countplot(x='bioactivity_label', data=df_2class, edgecolor='black')

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')

plt.savefig('EDA_fig/plot_bioactivity_class.pdf') #It's always a good thing to
#save as pdf (for supplementary figures for repport for exemple)

'''
scatter plot are great to determine visually if there is a potential tendence/
correlation or visible distinction between the classes.
'''
#Scatter plot MW vs LogP
plt.figure(figsize=(5.5,5.5))
sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='bioactivity_label',
                size='pIC50', edgecolor='black', alpha=0.7)


plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('EDA_fig/plot_MW_vs_LogP.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

'''
Next come the boxplot, which is amazing to visualise inner class distribution.
It is also a very good way of visualizing wether we will be able to classify 
easily.
'''
#Box plot pIC50 vs bioactivity class
plt.figure(figsize=(5.5,5.5))
sns.boxplot(x='bioactivity_label', y='pIC50', data=df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')

plt.savefig('EDA_fig/plot_ic50.pdf')

'''
Then, we apply the mannwhitney test to test if H0 : have the same descriptor
distribution.
'''
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
def mannwhitney(descriptor):
    seed(1)#Set the randome seed for reproductibility
    
    selection = [descriptor, 'bioactivity_label']
    df = df_2class[selection]
    active = df[df.bioactivity_label == 'active']
    active = active[descriptor]
    
    selection = [descriptor, 'bioactivity_label']
    df = df_2class[selection]
    inactive = df[df.bioactivity_label == 'inactive']
    inactive = inactive[descriptor]
    
    stat, p = mannwhitneyu(active, inactive)
    
    alpha = 0.05
    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'
        
    results = pd.DataFrame({'Descriptor':descriptor,
                            'Statistics':stat,
                            'p':p,
                            'alpha':alpha,
                            'Interpretation':interpretation},index=[0])
    filename = 'manwhiteneyu_'+descriptor+'.csv'
    results.to_csv(filename)
    
    return results


results_mannwhitney_pIC50 = mannwhitney('pIC50') #They do not share the same pIC50
#distribution.

#Box plot MW vs bioactivity class
plt.figure(figsize=(5.5,5.5))
sns.boxplot(x='bioactivity_label', y='MW', data=df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')

plt.savefig('EDA_fig/plot_MW.pdf')

results_mannwhitney_MW = mannwhitney('MW')

#Box plot LogP vs bioactivity class
plt.figure(figsize=(5.5,5.5))
sns.boxplot(x='bioactivity_label', y='LogP', data=df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')

plt.savefig('EDA_fig/plot_LogP.pdf')

results_mannwhitney_LogP = mannwhitney('LogP')

#Box plot NumHDonors vs bioactivity class
plt.figure(figsize=(5.5,5.5))
sns.boxplot(x='bioactivity_label', y='NumHDonors', data=df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')

plt.savefig('EDA_fig/plot_NumHDonors.pdf')

results_mannwhitney_NumHDonors = mannwhitney('NumHDonors')

#Box plot NumHAcceptors vs bioactivity class
plt.figure(figsize=(5.5,5.5))
sns.boxplot(x='bioactivity_label', y='NumHAcceptors', data=df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')

plt.savefig('EDA_fig/plot_NumHAcceptors.pdf')

results_mannwhitney_NumHAcceptors = mannwhitney('NumHAcceptors')


'''
Interpretation of Statistical Results

Box Plots
pIC50 values
actives and inactives showed statistically significat difference, which is to
be expected since we thresholded the class from their values:
    IC50 < 1 000 nM = actives / IC50 > 10 000 nM = Inactives
    
Lipinski's descriptors
from the 4 calculated descriptors only the LogP was of a similar distibution
either, the three othe descriptors (MW, NumHDonors, NumHAcceptors) showed
significant difference.
'''

