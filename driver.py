#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: driver.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains coordinate transform utilities
"""

# Import statements
import numpy as np
#from numpy import linalg as LA

from matplotlib import pyplot as plt

#from src import *
import src.coordinate_transforms as coord
import src.physics_constants as const
import src.integrate_rk4 as slv
import src.forces as frc

'''
N = 1

r = np.linspace(1,14,3*N).reshape((3,N))
v = np.linspace(57,9,3*N).reshape((3,N))
out = np.linspace(100,114,3*N).reshape((3,N))

print(r.shape)
print(v.shape)
print(out.shape)

out_vb = coord.ecr2vb(r,v,out)

tt = np.asarray([0.0, 0.0, 1.0 + const.R_EARTH])
tt = tt[:,np.newaxis]
lla = coord.ecr2lla(tt)
print(np.rad2deg(lla[0]), np.rad2deg(lla[1]), lla[-1])

tt = np.asarray([0.0, 1.0 + const.R_EARTH, 0.0])
tt = tt[:,np.newaxis]
lla = coord.ecr2lla(tt)
print(np.rad2deg(lla[0]), np.rad2deg(lla[1]), lla[-1])

tt = np.asarray([1.0 + const.R_EARTH, 0.0, 0.0])
tt = tt[:,np.newaxis]
lla = coord.ecr2lla(tt)
print(np.rad2deg(lla[0]), np.rad2deg(lla[1]), lla[-1])

tt = np.asarray([0.0, 0.0, -1.0 - const.R_EARTH])
tt = tt[:,np.newaxis]
lla = coord.ecr2lla(tt)
print(np.rad2deg(lla[0]), np.rad2deg(lla[1]), lla[-1])

tt = np.asarray([0.0, -1.0 - const.R_EARTH, 0.0])
tt = tt[:,np.newaxis]
lla = coord.ecr2lla(tt)
print(np.rad2deg(lla[0]), np.rad2deg(lla[1]), lla[-1])

tt = np.asarray([-1.0 - const.R_EARTH, 0.0, 0.0])
tt = tt[:,np.newaxis]
lla = coord.ecr2lla(tt)
print(np.rad2deg(lla[0]), np.rad2deg(lla[1]), lla[-1])
'''
altitudeTable = np.loadtxt('data/altitude_vs_time_data.txt').transpose()
time = altitudeTable[0,:]
time = time - time[0]
altitude = altitudeTable[1,:] * const.KFT_TO_KM
error = 5.0 * const.KFT_TO_KM # Half the smallest divison from old archived data
#altitudeTable = np.subtract(altitudeTable[:,:], altitudeTable[0,:]) 


sigmaTable = np.loadtxt('data/bankAngle_vs_time_data.txt').transpose()
tnew = np.linspace(sigmaTable[0,0],500,5000)#sigmaTable[0,-1],1000)

thetaE = np.deg2rad(174.24384)
phiE = np.deg2rad(23.653003)

r_ecr = (1.17e5 + const.R_EARTH) * np.array([np.cos(phiE)*np.cos(thetaE), np.cos(phiE)*np.sin(thetaE), -np.sin(phiE)])
#v_ecr = np.array([-1.23773e3, -10.3795e3,  3.63549e3 ])#11.0e3 * np.array([np.cos(phiE)*np.cos(thetaE), np.cos(phiE)*np.sin(thetaE), np.sin(phiE)])
#r_ecr = np.array([-5928.60e3,  597.622e3, -2592.694e3])#(1.2e5 + const.R_EARTH) * np.array([np.cos(phiE)*np.cos(thetaE), np.cos(phiE)*np.sin(thetaE), -np.sin(phiE)])
v_ecr = np.array([-1.23773e3, -10.3795e3,  3.63549e3 ])#11.0e3 * np.array([np.cos(phiE)*np.cos(thetaE), np.cos(phiE)*np.sin(thetaE), np.sin(phiE)])

print(36.2*const.KFT_TO_KM)
print(np.linalg.vector_norm(v_ecr, axis=0))

ts = tnew[0]
dt = tnew[1] - tnew[0]
i = 0

pos_ecr = []
vel_ecr = []
aero_ecr = []
acc_ecr = []

C_ecr = []
C_vb = []
bank = []
for i in range(0,tnew.shape[0]):#tnew.shape[0]):

    pos_ecr.append(r_ecr)
    vel_ecr.append(v_ecr)

    Ctemp,bankt = coord.wb2vb(np.array(tnew[i]), sigmaTable, const.CW_VEHICLE)
    C_vb.append(Ctemp)
    bank.append(bankt)

    Ctemp = coord.vb2ecr(r_ecr, v_ecr, Ctemp)
    C_ecr.append(Ctemp)

    vn = np.linalg.vector_norm(vel_ecr[i], axis=0)
    aero_ecr.append(-0.5 * frc.mass_density(pos_ecr[i]) * const.S_VEHICLE * np.square(vn) * C_ecr[i])

    acc_ecr.append(frc.net_acceleration(pos_ecr[i], vel_ecr[i], sigmaTable, ts))

    r_ecr, v_ecr, ts = slv.integrate_rk4(r_ecr, v_ecr, sigmaTable, np.array(tnew[i]), dt)

pos_ecr = np.array(pos_ecr).transpose()
vel_ecr = np.array(vel_ecr).transpose()
C_ecr = np.array(C_ecr).transpose()
aero_ecr = np.array(aero_ecr).transpose()
acc_ecr = np.array(acc_ecr).transpose()

#
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize = (18,6))
f.suptitle('Apollo 10 (Notional)')
#plt.tight_layout()

ax1.plot(tnew, coord.ecr2lla(pos_ecr)[2] / 1000.0)
ax1.errorbar(time, altitude, yerr = error, fmt='.', mfc='none', linewidth=1, capsize=4)
ax1.grid()
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Geocentric Altitude (km)")

ax2.plot(tnew, np.linalg.vector_norm(vel_ecr/1000.0, axis=0))
ax2.set_ylim([0, 12])
ax2.grid()
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (km/s)")

ax3.plot(tnew, np.linalg.vector_norm(acc_ecr / const.GRAV_SURF, axis=0))
ax2.set_ylim([0, 12])
ax3.grid()
ax3.set_xlabel("Time (s)")
ax3.set_ylabel(r"Magnitude of Acceleration (g)")

#
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize = (18,6))
f.suptitle('Apollo 10 (Notional)')

ax1.plot(tnew, bank)
ax1.errorbar(sigmaTable[0,:],sigmaTable[1,:], fmt='.', mfc='none', linewidth=1, capsize=4)
ax1.set_ylim([-200, 200])
ax1.grid()
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Bank Angle (deg)")

ax2.plot(tnew, C_ecr[0,:], label=r'$C_{x}$')
ax2.plot(tnew, C_ecr[1,:], label=r'$C_{y}$')
ax2.plot(tnew, C_ecr[2,:], label=r'$C_{z}$')
ax2.set_ylim([-2, 2])
ax2.grid()
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Drag Coefficients in ECR Frame ()")
ax2.legend(loc='upper right')

ax3.plot(tnew, aero_ecr[0,:] / const.GRAV_SURF / const.M_VEHICLE, label=r'$F_{x}$')
ax3.plot(tnew, aero_ecr[1,:] / const.GRAV_SURF / const.M_VEHICLE, label=r'$F_{y}$')
ax3.plot(tnew, aero_ecr[2,:] / const.GRAV_SURF / const.M_VEHICLE, label=r'$F_{z}$')
#ax3.set_ylim([-2, 2])
ax3.grid()
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Aerodynamic Acceleration in ECR Frame (g)")
ax3.legend(loc='upper right')

#plt.tight_layout()
plt.show()
#print(tt.shape, tt)




#to*np.heaviside(-np.mod(i,24),1.0)*np.exp(-(i - 24)/s)
