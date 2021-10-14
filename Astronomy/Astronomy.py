import os, sys
GitHub = os.environ['USERPROFILE'] + '\OneDrive\GitHub'
sys.path = sorted(set(sys.path + [GitHub + '\My Tools\General']))
sys.path = sorted(set(sys.path + [GitHub + '\My Tools\Astronomy']))

from Maths import *
from Constants import *
from Astronomy import *
import time

###############################################################################

Minutes = 60
Hours = 60*Minutes    # Length of an Hour
Days = 24*Hours    # Length of Earth Solar Day
Years = 365.25*Days    # Length of Earth Year
SiderealDay = 86164.0905   # Length of Earth rotation period

Time_JDEpoch_UnixEpoch = 2440587.5*Days
Time_UnixEpoch_J2000 = 10957.5*Days
Time_UnixEpoch_Now = time.time()

Time_JDEpoch_J2000 = Time_JDEpoch_UnixEpoch + Time_UnixEpoch_J2000
Time_JDEpoch_Now = Time_JDEpoch_UnixEpoch + Time_UnixEpoch_Now
Time_J2000_Now = Time_UnixEpoch_Now - Time_UnixEpoch_J2000

###############################################################################

def XYZ_LLA(XYZ, R=R_E):
    """Takes a position vector (XYZ) around a Planet and returns the Latitude, Longitude and Altitude (LON, LAT, ALT) in DEGREES"""
    [X, Y, Z] = XYZ
    r = Mag(XYZ)
    LON = VectorAngle(np.array([X, Y]), [1, 0])
    if Y < 0:
        LON = -LON
    LAT = 90 - Arccos(Z/r)
    ALT = r - R
    return np.array([LON, LAT, ALT])


def LLA_XYZ(LLA, R=R_E):
    """Takes a Latitude, Longitude and Altitude (LON, LAT, ALT) in DEGREES around a Planet and returns the position vector (XYZ)"""
    [LON, LAT, ALT] = LLA
    X = (R + ALT)*Cos(LON)*Cos(LAT)
    Y = (R + ALT)*Sin(LON)*Cos(LAT)
    Z = (R + ALT)*Sin(LAT)
    return np.array([X, Y, Z])


def AP_ENU(Azimuth, Pitch):
    """Takes an Azimuth and Pitch in DEGREES and returns the East, North, Up Vector"""
    A, P = Azimuth, Pitch
    E = Sin(A)*Cos(P)
    N = Cos(A)*Cos(P)
    U = Sin(P)
    return np.array([E, N, U])


def ENU_AP(ENU):
    """Takes an East, North, Up Vector and returns the Azimuth and Pitch in DEGREES"""
    [E, N, U] = Unit(np.array(ENU))
    Pitch = Arcsin(U)
    Azimuth = Atan2(E, N) % 360
    return [Azimuth, Pitch]


def UpVector(XYZ):
    """Takes a Geocentric XYZ and returns the direction of the Up Vector"""
    Up = Unit(XYZ)
    return Up


def EastVector(XYZ):
    """Takes a Geocentric XYZ and returns the direction of the East Vector"""
    Up = UpVector(XYZ)
    East = CrossProduct(np.array([0, 0, 1]), Up)
    return East


def NorthVector(XYZ):
    """Takes a Geocentric XYZ and returns the direction of the North Vector"""
    Up = UpVector(XYZ)
    East = EastVector(XYZ)
    North = CrossProduct(Up, East)
    return North


def XYZ_ENU_Matrix(XYZ):
    """Finds what the local ENU vectors are in respect to XYZ coordinates"""
    E = EastVector(XYZ)
    N = NorthVector(XYZ)
    U = UpVector(XYZ)
    return np.array([E, N, U])

###############################################################################

def Period_Omega(T):
    if T == 0:
        return 0
    else:
        return 2*π/T


def OrbitPeriod(a, μ=μ_E):
    τ = 2*π*((a**3)/μ)**0.5
    return τ


def CircularOrbit(r, μ=μ_E):
    """Takes an orbital radius and returns the orbital velocity necessary for a circular orbit"""
    if type(r) == float:
        v = (μ/r)**0.5
        return v
    elif type(r) == np.ndarray:
        v = (μ/Mag(r))**0.5
        v = CrossProduct([0, 0, 1], Unit(r))*v
        return v


def EscapeAngle(Perigee, Velocity, Direction, Planet):
    """Takes the perigee radius and velocity and finds the escape angle"""
    μ = G*Planet[0]    
    vesc = (2*μ/Perigee)**0.5
    if Velocity >= vesc:
        vinf = (Velocity**2 - vesc**2)**0.5
        e = 1 + (Perigee*vinf**2)/μ
        beta = Arcos(1/e)
        if Direction == 1:
            return beta - 90
        elif Direction == -1:
            return beta + 90
        else: print('Choose Direction')
    else:
        print('Not Fast Enough To Escape')


def Hohmann(r1, r2, Direction='Raise', μ=μ_E):
    """Takes a periapse and an apoapse and returns the properties of the Hohmann Transfer"""
    v1 = CircularOrbit(r1, μ)
    v2 = CircularOrbit(r2, μ)
    a = 0.5*(r1 + r2)
    if Direction == 'Raise':
        rp = r1
        ra = r2
        vp = (μ*(2/rp - 1/a))**0.5
        va = (μ*(2/ra - 1/a))**0.5
        Δv1 = vp - v1
        Δv2 = v2 - va
    elif Direction == 'Lower':
        ra = r1
        rp = r2
        vp = (μ*(2/rp - 1/a))**0.5
        va = (μ*(2/ra - 1/a))**0.5
        Δv1 = v1 - va
        Δv2 = vp - v2
    else: print('Check Direction')
    e = ra/a - 1
    Δv = Δv1 + Δv2
    τ1 = OrbitPeriod(rp, μ)
    τ2 = OrbitPeriod(ra, μ)
    τT = 0.5*OrbitPeriod(a, μ)
    θ = 180 - 360*τT/τ2
    return [v1, v2, a, e, vp, va, Δv1, Δv2, Δv, τ1, τ2, τT, θ]


def Transfer(r1, r2, Planet, Output):
    μ = G*Planet[0]    
    v1 = (μ/r1)**0.5
    v2 = (μ/r2)**0.5
    vp = (μ*(2*r2/(r1*(r1+r2))))**0.5
    va = (μ*(2*r1/(r2*(r1+r2))))**0.5
    ΔV1 = vp - v1
    ΔV2 = v2 - va
    τ1 = 2*math.pi*(r1**3/μ)**0.5
    τ2 = 2*math.pi*(r2**3/μ)**0.5
    τT = math.pi*((0.5*(r1+r2))**3/μ)**0.5
    Angle = 180 - 360*τT/τ2
    #DeltaV1=4 τT=8  Angle=9
    return [v1, v2, vp, va, ΔV1, ΔV2, τ1, τ2, τT, Angle][Output] 


def InverseHohmann(r1, Δv, Direction='Raise', μ=μ_E):
    """Takes an circular radius r1 and finds the maximum or minimum circular radius r2 it can transfer to with a given Δv"""
    def ThisFunction(r2):
        return Hohmann(r1, r2, Direction, μ)[8] - Δv
    if Direction == 'Raise':
        Start = 1.5*r1
    elif Direction == 'Lower':
        Start = 0.5*r1
    else: print('Check Direction')
    r2 = Optimize(ThisFunction, Start)
    return r2

###############################################################################

def True_Eccentric(θ, e):
    """Takes a True Anomaly in DEGREES and an Eccentricity and returns the Eccentric Anomaly in DEGREES"""
    θ = np.mod(θ, 360)      # 0 < θ < 360
    if e == 0:      # Circular
        E = θ
    if 0 < e < 1:       # Elliptical
        E = 2*Arctan(Tan(θ/2)*((1 - e)/(1 + e))**0.5)       # Eccentric Anomaly in terms of True Anomaly. -180 < E < 180
    if e == 1:      # Parabola
        D = Tan(θ/2)        # Parabolic Eccentric Anomaly in terms of True Anomaly
        E = Degrees(D)
    if e > 1:       # Hyperbolic
        F = 2*Arctanh(Tan(θ/2)*((e - 1)/(e + 1))**0.5)        # Hyperbolic Eccentric Anomaly in terms of True Anomaly
        E = Degrees(F)
    return E


def Eccentric_True(E, e):
    """Takes a Eccentric Anomaly in DEGREES and an Eccentricity and returns the True Anomaly in DEGREES"""
    E = Radians(E)
    if e == 0:      # Circular
        θ = E
    if 0 < e < 1:       # Elliptical
        θ = 2*np.arctan(np.tan(E/2)*((1 + e)/(1 - e))**0.5)       # True Anomaly in terms of Eccentric Anomaly
    if e == 1:      # Parabola
        D = E
        θ = 2*np.arctan(D)        # True Anomaly in terms of Parabolic Eccentric Anomaly
    if e > 1:       # Hyperbolic
        F = E
        θ = 2*np.arctan(np.tanh(F/2)*((e + 1)/(e - 1))**0.5)        # True Anomaly in terms of Hyperbolic Eccentric Anomaly
    θ = Degrees(θ)
    θ = np.mod(θ, 360)      # 0 < θ < 360
    return θ


def Eccentric_Mean(E, e):
    """Takes a Eccentric Anomaly in DEGREES and an Eccentricity and returns the Mean Anomaly in DEGREES"""
    E = Radians(E)
    if e == 0:      # Circular
        M = E
    if 0 < e < 1:       # Elliptical
        M = E - e*np.sin(E)       # Mean Anomaly in terms of Eccentric Anomaly
    if e == 1:      # Parabola
        D = E
        M = D + (D**3)/3       # Parabolic Mean Anomaly in terms of Eccentric Anomaly
    if e > 1:       # Hyperbolic
        F = E
        M = e*np.sinh(F) - F       # Hyperbolic Mean Anomaly in terms of Eccentric Anomaly
    M = Degrees(M)
    return M


def Mean_Eccentric(M, e):
    """Takes a Mean Anomaly in DEGREES and an Eccentricity and returns the Eccentric Anomaly in DEGREES"""
    def ThisFunction(E):
        return Eccentric_Mean(E, e) - M
    E = Optimize(ThisFunction, M)      # Inverses Keplers Equation. Finds Eccentric Anomaly from Mean Anomaly.
    return E


def Mean_True(M, e):
    """Takes a Mean Anomaly in DEGREES and an Eccentricity and returns the True Anomaly in DEGREES"""
    E = Mean_Eccentric(M, e)
    θ = Eccentric_True(E, e)
    return θ


def Trues_Time(θs, e, p, μ=μ_E):
    """Takes a two True Anomalies in DEGREES, Semi Major Axis Radius and an Eccentricity and returns the Time between these points."""
    if 0 <= e < 1:
        a = p/(1 - e**2)
        τ = OrbitPeriod(a, μ) 
        n = p**-1.5*μ**0.5*(1 - e**2)**1.5 
    elif e == 1:
        n = p**-1.5*μ**0.5
    elif e > 1:
        n = p**-1.5*μ**0.5*(e**2 - 1)**1.5
    
    n = Degrees(n)       # Mean Motion. Degrees/s
    Es = True_Eccentric(θs, e)      # Eccentric Anomalies
    Ms = Eccentric_Mean(Es, e)      # Mean Anomalies
    tps = Ms/n      # Periapse Times
    if 0 <= e < 1:
        tps = np.mod(tps, τ)
    Δt = tps[1] - tps[0]    # Time between points
    return Δt


def Time_Trues(Δt, θ1, e, p, μ=μ_E):
    """Takes a Time between these points, Semi Major Axis Radius and an Eccentricity and returns the two True Anomalies in DEGREES."""
    if 0 <= e < 1:
        a = p/(1 - e**2)
        τ = OrbitPeriod(a, μ) 
        n = p**-1.5*μ**0.5*(1 - e**2)**1.5 
    elif e == 1:
        n = p**-1.5*μ**0.5
    elif e > 1:
        n = p**-1.5*μ**0.5*(e**2 - 1)**1.5

    n = Degrees(n)       # Mean Motion. Degrees/s
    E1 = True_Eccentric(θ1, e)      # Eccentric Anomaly 1
    M1 = Eccentric_Mean(E1, e)      # Mean Anomaly 1
    tp1 = M1/n      # Periapse Time 1
    tp2 = tp1 + Δt
    M2 = n*tp2
    E2 = Mean_Eccentric(M2, e)
    θ2 = Eccentric_True(E2, e)
    θs = np.array([θ1, θ2])
    # return Δt, θ1, e, p, μ, n, E1, M1, tp1, tp2, M2, E2, θ2
    return θs
    
###############################################################################

def XYZ_Kepler(XYZ, VXYZ, μ=μ_E, R=R_E):
    """Takes a velocity and position vector (VXYZ, XYZ) and returns the equivalent orbital elements"""
    hXYZ = [hx, hy, hz] = CrossProduct(XYZ, VXYZ)      # Specific Angular Momentum Vector
    nXYZ = [nx, ny, nz] = CrossProduct(np.array([0, 0, 1]), hXYZ)
    eXYZ = [ex, ey, ez] = CrossProduct(VXYZ, hXYZ)/μ - Unit(XYZ)       # Eccentricity Vector
    r = Mag(XYZ)       # Radius
    v = Mag(VXYZ)      # Velocity
    h = Mag(hXYZ)      # Specific Angular Momentum
    n = Mag(nXYZ)
    e = Mag(eXYZ)       # Eccentricity
    E = 0.5*v**2 - μ/r      # Specific Energy
    [LON, LAT, ALT] = XYZ_LLA(XYZ, R)
    
    if h == 0:      # Vertical Fall Orbit
        print('Vertical Fall Orbit')
        Orbit = [None, XYZ, VXYZ, None, None, None]
        return Orbit
    
    i = Arccos(hz/h)       # Inclination. 0 < i < 180
    if -1e-15 < e < 1e-15 and E < 0:      # Circular Orbit
        e = 0
        print('Circular Orbit')
        LON = LON % 360
        r = -0.5*μ/E        # Radius. 0 < r < oo
        if i == 0 or i == 180:      # Zero Inclination Orbit
            print('Zero Inclination')
            Orbit = [e, r, i, None, None, LON]
        else:
            Ω = Atan2(hx, -hy) % 360         # Longitude of Ascending Node. 0 < Ω < 360
            u = VectorAngle(XYZ, nXYZ)       # Argument of Latitude. 0 < u < 360.
            if LAT < 0:
                u = 360 - u
            Orbit = [e, r, i, Ω, None, u]
        return Orbit
    
    e_r = eXYZ.dot(XYZ)/(e*r)
    # print(e_r)
    θ = Arccos(e_r) % 360        # True Anomaly. 0 < θ < 360
    if XYZ.dot(VXYZ) < 0:
        θ = 360 - θ
        
    if 0 < e < 1 and E < 0:       # Elliptical Orbit
        print('Elliptical Orbit')
        a = -0.5*μ/E        # Semi Major Axis. -oo < a < oo
        if i == 0 or i == 180:      # Zero Inclination Orbit
            print('Zero Inclination')
            ϖ = (LON - θ) % 360           # Longitude of Periapse.
            Orbit = [e, a, i, None, ϖ, θ]
        else:
            Ω = Atan2(hx, -hy) % 360        # Longitude of Ascending Node. 0 < Ω < 360
            ω = Atan2(ez/Sin(i), ey*Sin(Ω) + ex*Cos(Ω)) % 360       # Argument of Periapse. 0 < ω < 360
            Orbit = [e, a, i, Ω, ω, θ]
        return Orbit
    
    if e == 1 or E == 0:      # Parabolic Orbit
        print('Parabolic Orbit')
        rp = (0.5*h**2)/μ       # Radius of Periapse
        if i == 0 or i == 180:      # Zero Inclination Orbit
            print('Zero Inclination')
            ϖ = (LON - θ) % 360           # Longitude of Periapse.
            Orbit = [e, rp, i, None, ϖ, θ]
        else:
            Ω = Atan2(hx, -hy) % 360        # Longitude of Ascending Node. 0 < Ω < 360
            ω = Atan2(ez/Sin(i), ey*Sin(Ω) + ex*Cos(Ω)) % 360       # Argument of Periapse. 0 < ω < 360
            Orbit = [e, rp, i, Ω, ω, θ]
        return Orbit
    
    if e > 1 and E > 0:       # Hyperbolic Orbit
        print('Hyperbolic Orbit')
        a = -0.5*μ/E        # Semi Major Axis. -oo < a < oo
        if i == 0 or i == 180:      # Zero Inclination Orbit
            print('Zero Inclination')
            ϖ = (LON - θ) % 360           # Longitude of Periapse.
            Orbit = [e, a, i, None, ϖ, θ]
        else:
            Ω = Atan2(hx, -hy) % 360        # Longitude of Ascending Node. 0 < Ω < 360
            ω = Atan2(ez/Sin(i), ey*Sin(Ω) + ex*Cos(Ω)) % 360       # Argument of Periapse. 0 < ω < 360
            Orbit = [e, a, i, Ω, ω, θ]
        return Orbit

    print('Check Orbit')
    return


def KeplerΔt(Orbit, Δt, μ=μ_E, R=R_E):
    """Takes a Kepler orbit (Orbit) and changes the orbital elements according to the time increment"""
    [O1, O2, O3, O4, O5, O6] = Orbit
    
    e = O1
    if e == None:      # Vertical Fall Orbit
        XYZ, VXYZ =  O2, O3
        AXYZ = -μ*Unit(XYZ)*Mag(XYZ)**-2
        XYZΔt = XYZ + VXYZ*Δt + 0.5*AXYZ*Δt**2
        VXYZΔt = VXYZ + AXYZ*Δt
        return [None, XYZΔt, VXYZΔt, None, None, None]

    i = O3
    if e == 0:      # Circular Orbit
        r = O2
        τ = OrbitPeriod(r, μ)
        n = 360/τ
        if i == 0 or i == 180:
            LON = O6
            LONΔt = (LON + Δt*n) % 360
            return [e, r, i, None, None, LONΔt]
        else:
            Ω, u = O4, O6
            uΔt = u + Δt*n
            return [e, r, i, Ω, None, uΔt]
    
    θ = O6
    if e == 1:      # Parabolic Orbit
        rp = O2
        p = 2*rp
    
    else:       # Elliptical or Hyperbolic Orbit
        a = O2
        p = a*(1 - e**2)        

    θΔt = Time_Trues(Δt, θ, e, p, μ)[1]
    OrbitΔt = Orbit
    OrbitΔt[5] = θΔt
    return OrbitΔt


def Kepler_XYZ(Orbit, μ=μ_E, R=R_E):
    """Takes orbital elements (Orbit) and returns the equivalent velocity and position vector (VXYZ, XYZ)"""
    [O1, O2, O3, O4, O5, O6] = Orbit
    
    e = O1
    if e == None:      # Vertical Fall Orbit
        print('Vertical Fall Orbit')
        XYZ, VXYZ = O2, O3
        return [XYZ, VXYZ]
    
    i = O3
    if e == 0:      # Circular Orbit
        print('Circular Orbit')
        r = O2
        v = (μ/r)**0.5
        XYZ = np.array([r, 0, 0])
        VXYZ = np.array([0, v, 0])
        if i == 0 or i == 180:
            print('Zero Inclination')
            LON = O6
            Q = RotateZ(LON)
        else:
            Ω, u = O4, O6
            Q = RotateZ(Ω).dot(RotateX(i).dot(RotateZ(u)))

    θ = O6
    if 0 < e < 1:       # Elliptical Orbit
        print('Elliptical Orbit')
        a = O2
        p = a*(1 - e**2)
        h = (p*μ)**0.5
        r = p/(1 + e*Cos(θ))
        XYZ = np.array([Cos(θ), Sin(θ), 0])*r
        VXYZ = np.array([-Sin(θ), e + Cos(θ), 0])*μ/h
        if i == 0 or i == 180:
            print('Zero Inclination')
            ϖ = O5
            Q = RotateZ(ϖ)
        else:
            Ω = O4
            ω = O5
            Q = RotateZ(Ω).dot(RotateX(i).dot(RotateZ(ω)))
    
    if e == 1:      # Parabolic Orbit
        print('Parabolic Orbit')
        rp = O2
        p = 2*rp
        h = (p*μ)**0.5
        r = p/(1 + Cos(θ))
        XYZ = np.array([Cos(θ), Sin(θ), 0])*r
        VXYZ = np.array([-Sin(θ), e + Cos(θ), 0])*μ/h
        if i == 0 or i == 180:
            print('Zero Inclination')
            ϖ = O5
            Q = RotateZ(ϖ)
        else:
            Ω = O4
            ω = O5
            Q = RotateZ(Ω).dot(RotateX(i).dot(RotateZ(ω)))
        
    elif e > 1:       # Hyperbolic Orbit
        print('Hyperbolic Orbit')
        a = O2
        p = a*(1 - e**2)
        h = (p*μ)**0.5
        r = p/(1 + e*Cos(θ))
        XYZ = np.array([Cos(θ), Sin(θ), 0])*r
        VXYZ = np.array([-Sin(θ), e + Cos(θ), 0])*μ/h
        if i == 0 or i == 180:
            print('Zero Inclination')
            ϖ = O5
            Q = RotateZ(ϖ)
        else:
            Ω = O4
            ω = O5
            Q = RotateZ(Ω).dot(RotateX(i).dot(RotateZ(ω)))
    
    XYZ = Q.dot(XYZ)
    VXYZ = Q.dot(VXYZ)
    return [XYZ, VXYZ]

###############################################################################

def Occlusion(PositionXYZ, ObjectXYZ, Radius, TargetXYZ):
    """Takes a Position, Object and Target Vector and returns a 1 if Target is Occluded by Object from Position and 0 if not"""
    ObjectNXYZ = ObjectXYZ - PositionXYZ    # Normalise Object to Position
    TargetNXYZ = TargetXYZ - PositionXYZ    # Normalise Target to Position
    O = Mag(ObjectNXYZ)    # Distance to Object
    R = Radius      # Radius of Object
    T = Mag(TargetNXYZ)    # Distance to Target
    θ = VectorAngle(ObjectNXYZ, TargetNXYZ)    # Angle between Target and Object
    h = O - R   # Height above the Objects surface
    E = 0   # Assume no Occlusion
    if h < 0:   # If Height is below surface: 
        E = 1   # Occluded
    else:
        H = (O**2 - R**2)**0.5      # Distance to Horizon
        g = (H**2 - h**2)/(2*(R + h))           
        ψ = Arcos((h + g)/H)    # Angle between Object and Horizon
        if θ < ψ:   # If Target is below Horizon:    
            if H < T:   # And Target is behind Horizon:
                E = 1   # Occluded
    return E

###############################################################################

# def WhereIs(Planet):
#     SO17LLA = [-1.385914, 50.931167, 0]     # Southampton Address Lon Lat Alt
#     PO16LLA = [-1.145366, 50.848140, 0]     # Portsmouth Address Lon Lat Alt
#     LLA = PO16LLA 
#     NXYZ = LLA_XYZ(LLA, R_E)    # Geocentric XYZ Position of Address
#     NXYZ_ENU = XYZ_ENU_Matrix(NXYZ)    # Transformation Matrix from Geocentric XYZ to Localised East North Up 
    
#     XYZEquinox = Earth('Equinox')[3]    # Heliocentric XYZ of Earth at 18:28 September 22 2000 Equinox
#     # XYZEquinox = np.array([AU, 0, 0])
#     Obliquity = 23.43661    # Angle in DEGREES between Earth's spin axis and the Ecliptic 
#     QObliquity = AxisAngle_Q(XYZEquinox, Obliquity)    #Quarternion representing a rotation of 23 DEGREES around the Equinox axis
#     RM = RotateZ(-Earth('Now')[5])      # Rotation Matrix adjusting for time of day
    
#     TargetNXYZ = Planet[3] - Earth('Now')[3]    # Geocentric XYZ of Target Planet
#     TargetNXYZ = RM.dot(RotateVectorQ(TargetNXYZ, QObliquity)) - NXYZ      # Orientation adjusted Planet XYZ and Renormalised to Position
#     TargetENU = NXYZ_ENU.dot(TargetNXYZ)        # Geocentric, Rotated Planet XYZ in terms of Localised East North Up Vectors
#     [A, P] = ENU_AP(TargetENU)      # Azimuth and Pitch in DEGREES of Target Planet
#     Azimuth, Pitch = round(A, 1), round(P, 1)
#     print(Planet[6], 'is', Azimuth, 'degrees from North and', Pitch, 'degrees from the horizon')
#     return A, P




