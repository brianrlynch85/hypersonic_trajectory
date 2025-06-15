#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: utils.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains utilities
"""

import numpy as np

# Function to get the shape of a nested tuple
def get_shape(t):
    if isinstance(t, tuple):
        return (len(t),) + tuple(get_shape(sub) for sub in t if isinstance(sub, tuple))
    return ()

#
def check_for_duplicates(array):
   unique_elements, counts = np.unique(array, return_counts=True)
   duplicates = unique_elements[counts > 1]
   
   if duplicates.size > 0:
      raise ValueError(f"Duplicate entries found: {duplicates}")
   
   return

#
def get_extrema_indices(y_):

   # Find derivative of x
   yprime = np.diff(y_)

   # Calc. sign difference
   sign_diff = np.sign(yprime[1:]) - np.sign(yprime[:-1])

   # Find local min and max
   min_indices = [i for i,k in enumerate(sign_diff) if k == 2]
   max_indices = [i for i,k in enumerate(sign_diff) if k == -2]

   return  min_indices, max_indices

#
def get_inflection_indices(y_):

   # Find derivative of x
   yprime = np.diff(y_)
   yprimeprime = np.diff(yprime)

   # Calc. sign difference
   sign_diff = np.sign(yprimeprime[1:]) - np.sign(yprimeprime[:-1])

   # Find local min and max
   min_indices = [i for i,k in enumerate(sign_diff) if k == 2]
   max_indices = [i for i,k in enumerate(sign_diff) if k == -2]

   return  min_indices, max_indices