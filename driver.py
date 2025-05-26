#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: driver.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains a driver to produce a hypersonic trajectory
"""

# Import statements
import numpy as np
from matplotlib import pyplot as plt

import src.coordinate_transforms as coord
import src.physics_constants as const
import src.integrate_rk4 as slv
import src.forces as frc

#
def main():
   altitude_filename = 'data/altitude_vs_time_data.txt'
   bank_angle_filename = 'data/bankAngle_vs_time_data.txt'

   altitudeTable = np.loadtxt(altitude_filename).transpose()
   timeMeas = altitudeTable[0,:]
   timeMeas = timeMeas - timeMeas[0]
   altitude = altitudeTable[1,:] * const.KFT_TO_KM
   error = 5.0 * const.KFT_TO_KM # Half the smallest divison from old archived blurry graph

   sigmaTable = np.loadtxt(bank_angle_filename).transpose()


   thetaE = np.deg2rad(174.0)
   phiE = np.deg2rad(23.7)

   #r_ecr = np.array([11976.0, -15451.0, -8506.0]) * const.KFT_TO_M #
   r_ecr = (altitude[0] * const.KM_TO_M + const.R_EARTH) * np.array([np.cos(phiE)*np.cos(thetaE), np.cos(phiE)*np.sin(thetaE), -np.sin(phiE)])
   v_ecr = np.array([-1.23e3, -10.4e3,  3.64e3 ])
   #v_ecr = np.array([27.5   , 20.5    , 11.9]) * const.KFT_TO_M #
   print(r_ecr)
   print(v_ecr)

   Nsteps = 1000
   r_ecr, v_ecr, time = slv.solvesystem(r_ecr, v_ecr, sigmaTable, timeMeas[0], timeMeas[-1], Nsteps)

   aeroCoeffWB = const.CW_VEHICLE[:,np.newaxis]
   aeroCoeffVB, bank = coord.wind2velocity(time, sigmaTable, aeroCoeffWB)

   aeroCoeffECR = coord.velocity2ecr(r_ecr, v_ecr, aeroCoeffVB)
   
   aeroCoeffVB2 = coord.ecr2velocity(r_ecr, v_ecr, aeroCoeffECR)
   aeroCoeffWB2, bank2 = coord.velocity2wind(time, sigmaTable, aeroCoeffVB2)

   aeroForceECR = frc.aero_force(r_ecr, v_ecr, sigmaTable, time)

   a_ecr = frc.net_acceleration(r_ecr, v_ecr, sigmaTable, time)
   a_vb = coord.ecr2velocity(r_ecr, v_ecr, a_ecr)
   a_wb,_ = coord.velocity2wind(time, sigmaTable, a_vb)
   #print(a_wb.shape)

   #
   f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize = (18,6))
   f.suptitle('Apollo 10 (Notional)')

   ax1.plot(time, coord.ecr2lla(r_ecr)[2] / 1000.0)
   ax1.errorbar(timeMeas, altitude, yerr = error, fmt='.', mfc='none', linewidth=1, capsize=4)
   ax1.grid()
   ax1.set_xlabel("Time (s)")
   ax1.set_ylabel("Geocentric Altitude (km)")

   ax2.plot(time, np.linalg.vector_norm(v_ecr/1000.0, axis=0))
   ax2.set_ylim([0, 12])
   ax2.grid()
   ax2.set_xlabel("Time (s)")
   ax2.set_ylabel("Speed (km/s)")

   ax3.plot(time, np.linalg.vector_norm(a_ecr / const.GRAV_SURF, axis=0))
   ax3.set_ylim([0, 12])
   ax3.grid()
   ax3.set_xlabel("Time (s)")
   ax3.set_ylabel(r"Magnitude of Acceleration (g)")

   #
   '''
   f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize = (18,6))
   f.suptitle('Apollo 10 (Notional)')

   ax1.plot(time, bank)
   ax1.errorbar(sigmaTable[0,:],sigmaTable[1,:], fmt='.', mfc='none', linewidth=1, capsize=4)
   ax1.set_ylim([-200, 200])
   ax1.grid()
   ax1.set_xlabel("Time (s)")
   ax1.set_ylabel("Bank Angle (deg)")

   ax2.plot(time, aeroCoeffECR[0,:], label=r'$C_{x}$')
   ax2.plot(time, aeroCoeffECR[1,:], label=r'$C_{y}$')
   ax2.plot(time, aeroCoeffECR[2,:], label=r'$C_{z}$')
   ax2.set_ylim([-2, 2])
   ax2.grid()
   ax2.set_xlabel("Time (s)")
   ax2.set_ylabel("Drag Coefficients in ECR Frame ()")
   ax2.legend(loc='upper right')

   ax3.plot(time, aeroForceECR[0,:] / const.GRAV_SURF / const.M_VEHICLE, label=r'$F_{x}$')
   ax3.plot(time, aeroForceECR[1,:] / const.GRAV_SURF / const.M_VEHICLE, label=r'$F_{y}$')
   ax3.plot(time, aeroForceECR[2,:] / const.GRAV_SURF / const.M_VEHICLE, label=r'$F_{z}$')
   #ax3.set_ylim([-2, 2])
   ax3.grid()
   ax3.set_xlabel("Time (s)")
   ax3.set_ylabel("Aerodynamic Acceleration in ECR Frame (g)")
   ax3.legend(loc='upper right')

   #
   f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize = (18,6))
   f.suptitle('Apollo 10 (Notional)')

   ax1.plot(time, aeroCoeffECR[0,:], label=r'$C_{x}$')
   ax1.plot(time, aeroCoeffECR[1,:], label=r'$C_{y}$')
   ax1.plot(time, aeroCoeffECR[2,:], label=r'$C_{z}$')
   ax1.set_ylim([-2, 2])
   ax1.grid()
   ax1.set_xlabel("Time (s)")
   ax1.set_ylabel("Drag Coefficients in ECR Frame ()")
   ax1.legend(loc='upper right')

   ax2.plot(time, aeroCoeffVB[0,:], label=r'$C_{x}$')
   ax2.plot(time, aeroCoeffVB[1,:], label=r'$C_{y}$')
   ax2.plot(time, aeroCoeffVB[2,:], label=r'$C_{z}$')
   ax2.set_ylim([-2, 2])
   ax2.grid()
   ax2.set_xlabel("Time (s)")
   ax2.set_ylabel("Drag Coefficients in Velocity Frame ()")
   ax2.legend(loc='upper right')

   ax3.plot(time, aeroCoeffWB2[0,:], label=r'$C_{S}$')
   ax3.plot(time, aeroCoeffWB2[1,:], label=r'$C_{D}$')
   ax3.plot(time, aeroCoeffWB2[2,:], label=r'$C_{L}$')
   ax3.set_ylim([-2, 2])
   ax3.grid()
   ax3.set_xlabel("Time (s)")
   ax3.set_ylabel("Drag Coefficients in Wind Frame ()")
   ax3.legend(loc='upper right')
   '''
   #
   plt.show()

#
if __name__ == '__main__':
   main()