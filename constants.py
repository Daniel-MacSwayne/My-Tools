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
AU_m = 1.495978707*10**11     # Scale Astronomical Units to meters

# Universal Physics
G = 6.6743*10**-11      # Gravitational Constant. m3.kg-1.s-2.
R_u = 8.314      # Universal Gas Constant
σ = 5.670374419*10**-8      # Stefan Boltzmann Constant. kg.s-3.K-4

# Gas Dynamics
γ = 1.4     # Ratio of Specific Heats. Diatomic Gas
γ_m = 1.2857      # Ratio of Specific Heats. Monatomic Gas
R_a = 287     # Gas Constant for Air. m3.kg-1.K-1
ρ_0 = 1.225      # Sea Level Air Density. kg.m-3

# Astro Dynamics
M_E = 5.9722*10**24     # Mass Earth. kg
μ_E = G*M_E            # Earth Gravitational Parameter 
g_0 = 9.80665        # Earth Average Surface Gravity. m.s-2
R_E = 6.378*10**6     # Radius Earth. m
# rE = 6.371*10**6     # Alternative Radius Earth. m
r_GEO = 42164000         # Radius of Geostationary orbit

M_S = 1.99*10**30     # Mass Sun. kg
μ_S = G*M_S            # Sun Gravitational Parameter

###############################################################################