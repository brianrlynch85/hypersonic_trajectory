#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: coordinate_transforms.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains cheesily wrapped coordinate transform utilities
"""

from src.coordinates import vb2wb as v2w
from src.coordinates import wb2vb as w2v

from src.coordinates import vb2ecr as v2e
from src.coordinates import ecr2vb as e2v

from src.coordinates import ecr2lla as e2l
from src.coordinates import lla2ecr as l2e

# 
def vb2wb(t_, tsigmaTable_, invec_vb_):
   return v2w.vb2wb(t_, tsigmaTable_, invec_vb_)

# 
def wb2vb(t_, tsigmaTable_, invec_wb_):
   return w2v.wb2vb(t_, tsigmaTable_, invec_wb_)

#
def vb2ecr(r_ecr_, v_ecr_, invec_vb):
   return v2e.vb2ecr(r_ecr_, v_ecr_, invec_vb)

#
def ecr2vb(r_ecr_, v_ecr_, invec_ecr_):
   return e2v.ecr2vb(r_ecr_, v_ecr_, invec_ecr_)

# 
def lla2ecr(lla_, bool_is_geodetic_=False):
   return l2e.lla2ecr(lla_, bool_is_geodetic_)

# 
def ecr2lla(r_ecr_, bool_is_geodetic_=False):
   return e2l.ecr2lla(r_ecr_, bool_is_geodetic_)