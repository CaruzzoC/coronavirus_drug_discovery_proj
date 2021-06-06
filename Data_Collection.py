# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 12:12:18 2021

@author: cedri
"""

import pandas as pd
from chembl_webresource_client.new_client import new_client

#search data about coronavirus in ChEMBL
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)

#Select and retrieve bioactivity data for SARS coronavirus 3C-like proteinase
selected_target = targets.target_chembl_id[4]

#Retrieving bioactivity only for coronavirus 3C-like proteinase([slected_target])
#reported as IC50
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(res)

df.to_csv('bioactivity_data.csv', index=False)