#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:25:57 2020

@author: sims
"""

import pandas as pd

a = label_name_sequence.copy()
c = []
for i in range(len(a)-1):
    c.append(a[i])
    c.append(a[i])
    c.append(a[i])
    c.append(a[i])
c.append(a[len(a)-1])
tmp = pd.DataFrame(data=c)
b = pd.read_csv('/home/sims/Downloads/02_s_10_label_new.csv')

count = 0 
for i in tmp[0] == b['label'][:3253]:
    if i == True:
        count+=1