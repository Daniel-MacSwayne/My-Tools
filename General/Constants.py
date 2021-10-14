# Created on Sun Sep  5 14:03:00 2021
# @author: Daniel MacSwayne
# A collection of common constants used in physics.

###############################################################################
# Imports
import math

###############################################################################
# Constants

# Maths
π = math.pi     # Ratio of Circle Diameter to Circumference

# Unit Conversions
C_K = 273.15    # Offset Celcius to Kelvin
AU_m = 1.495978707e11     # Scale Astronomical Units to meters
psi_pa = 6894.76        # Scale Pounds per Square Inch to Pascals
bar_pa = 101325         # Scale bar to Pascals
lbs_kg = 1/2.2            # Scale Pounds to Kilograms
inch_m = 2.54e-2        # Scale Inches to Meters
eV_J = 1.6e-19           # Scale Electron Volts to Joules

# Universal Physics
G = 6.6743e-11      # Gravitational Constant. m3.kg-1.s-2
σ = 5.670374419e-8      # Stefan Boltzmann Constant. kg.s-3.K-4

# Gas Dynamics
k_b = 1.380649e-23      # Boltzmann Constant. J.K-1
N_a = 6.02214076e23     # Avagadro Constant.
R_u = 8.31446261815324      # Universal Gas Constant. J.K-1.mol-1
γ = 1.4     # Ratio of Specific Heats. Diatomic Gas
γ_m = 1.2857      # Ratio of Specific Heats. Monatomic Gas
R_a = 287     # Gas Constant for Air. m3.kg-1.K-1
P_a_sl = 101325
ρ_a_sl = 1.225      # Sea Level Air Density. kg.m-3

# Astro Dynamics
M_E = 5.9722e24     # Mass Earth. kg
μ_E = G*M_E            # Earth Gravitational Parameter 
g_0 = 9.80665        # Earth Average Surface Gravity. m.s-2
R_E = 6.378e6     # Radius Earth. m
# rE = 6.371e6     # Alternative Radius Earth. m
r_GEO = 4.2164e7         # Radius of Geostationary orbit
M_S = 1.99e30     # Mass Sun. kg
μ_S = G*M_S            # Sun Gravitational Parameter

###############################################################################