# Created on Sun Sep  5 18:24:13 2021
# @author: Daniel MacSwayne
# A collection of mathematical tools

###############################################################################

import math
import numpy as np
import pandas as pd
import numba
import sympy as sy
import scipy.integrate as spint
import scipy.optimize as spopt
from scipy.misc import derivative
import matplotlib.pyplot as plt
import time as pytime
old_err_state = np.seterr(divide='raise')

π = math.pi

@numba.jit
###############################################################################
# Plotting

def CreatePlot(Size=(6, 6), Dim=2):
    Figure = plt.figure(figsize=Size)
    if Dim == 2:
        Axes = Figure.gca()
    elif Dim == 3:
        Axes = Figure.add_subplot(111, projection='3d')
    return Figure, Axes


def Plot_Function(f, I, x_axis=True):
    x_s = np.linspace(I[0], I[1], 101)
    y_s = f(x_s)
    Figure, Axes = CreatePlot()
    Axes.plot(x_s, y_s, color='blue', label='y = f(x)')
    if x_axis == True:
        Axes.plot([I[0], I[1]], [0, 0], '--', color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    return Figure, Axes

###############################################################################
# Trigonometry

def Deg(θ):
    """Takes Angles in RADIANS and returns Angles in DEGREES"""
    return np.rad2deg(θ)


def Rad(θ):
    """Takes Angles in DEGREES and returns Angles in RADIANS"""
    return np.deg2rad(θ)


def Sin(θ):
    """Takes Angles in DEGREES and returns the Sines of the Angles"""
    return np.sin(Rad(θ))


def Cos(θ):
    """Takes Angles in DEGREES and returns the Cosines of the Angles"""
    return np.cos(Rad(θ))


def Tan(θ):
    """Takes Angles in DEGREES and returns the Tangents of the Angles"""
    return np.tan(Rad(θ))


def Arcsin(Component):
    """Takes the Sines of Angles and returns the Angles in DEGREES"""
    return Deg(np.arcsin(Component))


def Arccos(Component):
    """Takes the Cosines of Angles and returns the Angles in DEGREES"""
    return Deg(np.arccos(Component))


def Arctan(Component):
    """Takes the Tangents of Angles and returns the Angles in DEGREES"""
    return Deg(np.arctan(Component))


def Atan2(y, x):
    """Returns the Angle in DEGREES from the x [0, 360]"""
    if x == 0:
        if y == 0:
            θ = 0
        elif y > 0:
            θ = 90
        elif y < 0:
            θ = -90
    elif x > 0 and y >= 0:
        θ = Arctan(y/x)
    elif x > 0 and y <= 0:
        θ = Arctan(y/x)
    elif x < 0 and y >= 0:
        θ = 180 + Arctan(y/x)
    elif x < 0 and y <= 0:
        θ = Arctan(y/x) - 180
    return θ

###############################################################################
# Hyperbolic Trigonometry

def Sinh(x):
    """Returns the Hyperbolic Sine of x"""
    return np.sinh(x)


def Cosh(x):
    """Returns the Hyperbolic Cosine of x"""
    return np.cosh(x)


def Tanh(x):
    """Returns the Hyperbolic Tangent of x"""
    return np.cosh(x)


def Arcsinh(x):
    """Returns the Inverse Hyperbolic Sine of x"""
    return np.arcsinh(x)


def Arccosh(x):
    """Returns the Inverse Hyperbolic Cosine of x"""
    return np.arccosh(x)


def Arctanh(x):
    """Returns the Inverse Hyperbolic Tangent of x"""
    return np.arctanh(x)

###############################################################################
# Vector Operations

def Mag(v):
    """Returns the Magnitude of a Vector"""
    return np.linalg.norm(np.array(v), axis=-1)


def Unit(v):
    """Returns the Unit Vector"""
    try:
        return np.array(v)/Mag(v)
    except:
        return np.zeros(len(v))
                      

def DotProduct(v_1, v_2):
    """Returns the Dot Product of two Vectors"""
    return np.array(v_1).dot(np.array(v_2))


def CrossProduct(v_1, v_2):
    """Returns the Cross Product of two Vectors"""
    np.array(v_1, 'float64')
    np.array(v_2, 'float64')
    return np.cross(v_1, v_2)


def VectorAngle(v_1, v_2):
    """Returns the smallest angle in DEGREES between two Vectors"""
    Dot = DotProduct(v_1, v_2)
    Mag1 = Mag(v_1)
    Mag2 = Mag(v_2)
    try:
        return Arccos(Dot/(Mag1*Mag2))
    except:
        return 0

###############################################################################
# Matrix Operations

def TransposeMatrix(M):
    """Returns the Transposed Matrix"""
    return np.transpose(np.array(M))


def MultiplyMatrix(M_1, M_2):
    """Returns the Multiplication of two Matrices"""
    return np.array(M_1).Dot(np.array(M_2))


def DetMatrix(M):
    """Returns the Determinant of a Matrix"""
    return np.linalg.det(np.array(M))


def InverseMatrix(M):
    """Return the Inverse Matrix"""
    try:
        return np.linalg.inv(np.array(M))
    except:
        return np.array(M)


def EigenMatrix(M):
    """Returns the Eigen Values of a Matrix"""
    try:
        S, V = np.linalg.eig(np.array(M))
        return S, V
    except:
        return np.zeros(len(M))


def RotateX(θ):
    """Rotates a Vector Rotated about the X Axis by an Angle in DEGREES. +ACW"""
    c = Cos(θ)
    s = Sin(θ)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def RotateY(θ):
    """Rotates a Vector Rotated about the Y Axis by an Angle in DEGREES. +ACW"""
    c = Cos(θ)
    s = Sin(θ)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def RotateZ(θ):
    """Rotates a Vector Rotated about the Z Axis by an Angle in DEGREES. +ACW"""
    c = Cos(θ)
    s = Sin(θ)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ReflectInPlane(v):
    """Takes a Normal Vector of a Plane and returns the matrix that reflects in this plane"""
    [a, b, c] = Unit(v)
    A = np.array([[1 - 2*a**2, -2*a*b, -2*a*c],
                  [-2*a*b, 1 - 2*b**2, -2*b*c],
                  [-2*a*c, -2*b*c, 1 - 2*c**2]])
    return A

###############################################################################
# Quarternions Operations

def AxisAngle_Q(a, θ):
    """Takes an Axis and an Angle and returns a Quarternion"""
    Q = np.zeros(4)
    Q[0] = Cos(θ/2)
    Q[1:] = Unit(a)
    return Q


def Q_AxisAngle(Q):
    """Takes a Quarternion and returns an Axis and an Angle """
    a = Unit(Q[1:])    
    θ = 2*Arccos(Q[0])
    return [a, θ]

    
def ConjugateQ(Q):
    """Returns the Conjugate of a Quarternion"""
    return np.array([Q[0], -Q[1], -Q[2], -Q[3]])


def MultiplyQ(Q1, Q2):
    """Multiplies two Quarternions"""
    w = Q1[0]*Q2[0] - Q1[1]*Q2[1] - Q1[2]*Q2[2] - Q1[3]*Q2[3]
    x = Q1[0]*Q2[1] + Q1[1]*Q2[0] + Q1[2]*Q2[3] - Q1[3]*Q2[2]
    y = Q1[0]*Q2[2] + Q1[2]*Q2[0] + Q1[3]*Q2[1] - Q1[1]*Q2[3]
    z = Q1[0]*Q2[3] + Q1[3]*Q2[0] + Q1[1]*Q2[2] - Q1[2]*Q2[1]
    return np.array([w, x, y, z])


def RotateVectorQ(v, Q):
    """Rotates a Vector with a Quarternion Q.V.Q*"""
    V = np.append([0], v)
    return MultiplyQ(MultiplyQ(Q, V), ConjugateQ(Q))[1:]

###############################################################################
# Calculus

def Differentiate(Function, Point, dx=1e-12):
    """Finds the Derivative of a Function at a given Point"""
    D = derivative(Function, Point, dx)
    return D


def Integrate(Function, Start, End):
    """Takes a Function and Intergrates between an Interval""" 
    I, ERROR = spint.quad(Function, Start, End)
    return I


def RK4_Integrator(f, y_0, t_0, dt, T):
    """Takes a function, f, to be integrated, an initial state vector y_0,
    an initial time t_0, a timestep dt. It then computes iterations of an 
    RK4 integration up to t=T. Returns the State Vector and Time lists"""    
    def RK4_Step(f, y, t, dt):
        """Performs 1 iteration of the RK4 algorithm."""
        k_1 = dt*f(y, t)                     # Coefficients of Integration
        k_2 = dt*f(y + k_1/2, t + dt/2)
        k_3 = dt*f(y + k_2/2, t + dt/2)
        k_4 = dt*f(y + k_3, t + dt)
        y_ = y + (k_1 + 2*k_2 + 2*k_3 + k_4)/6       # Update State Vector
        return y_
    
    I = round((T - t_0)/dt)                  # Number of Iterations
    if (T - t_0)/dt != I:
        print("""Time Step, dt, does not fit into Duration, T, an integer 
              number of times. Iterated up to T =""", dt*I)
    
    y_s = np.zeros((I + 1,) + np.shape(y_0))        # Initialise State vector List
    t_s = np.zeros(I + 1)                             # Initialise Times list 
    y_s[0] = y_0
    t_s[0] = t_0
    
    for I in range(1, I + 1):         # Loop Iterations
        y = y_s[I - 1]
        t = t_s[I - 1]
        y_s[I] = RK4_Step(f, y, t, dt)            # Update State Vector
        t_s[I] = t + dt                           # Update Time
    return y_s, t_s

###############################################################################

def Optimize(Function, Start):
    """Finds a value of the Function that returns 0. Uses Newton Raphson with a Start point"""
    Root = spopt.newton(Function, Start)
    return Root


def Bisection(f, I, n=100, Plot=False):
    [x_l0, x_r0] = I
    x_l, x_r = x_l0, x_r0
    
    if f(x_l) * f(x_r) > 0:
        print('No Roots Detected within given I. Could be an even number of Roots.')
        return
    
    xl_s = np.zeros(n)
    xr_s = np.zeros(n)
    xm_s = np.zeros(n)
    
    for i in range(n):
        x_m = 0.5*(x_l + x_r)
        
        xl_s[i] = x_l
        xr_s[i] = x_r
        xm_s[i] = x_m

        if f(x_m) < 0:
            x_l = x_m
        elif f(x_m) > 0:
            x_r = x_m
        
            
    yl_s = f(xl_s)
    yl_s = f(xl_s)
    
    if Plot == True:
        Figure, Axes = Plot_Function(f, I)
        for i in range(n):
            Axes.plot(xm_s, f(xm_s), 'o', color='red')

    # return xl_s, xr_s, xm_s
    return x_m


def Iterative_g(f, x_0, n=100, Plot=False):
    # def g(x):
    #     return x - f(x)
    def g(x):
        return x + f(x)/13
    
    x_s = np.zeros(n + 1)
    x_s[0] = x_0
    
    for i in range(n):
        x_s[i + 1] = g(x_s[i])
    
    y_s = x_s.copy()
    # y_s[0] = 0
    y_s[0] = min(y_s)
        
    I = np.array([[min(x_s), max(x_s)], [min(y_s), max(y_s)]])
    
    s = x_s[-1]
    
    if Plot == True:
        Figure1, Axes1 = Plot_Function(f, I[0])
        
        X_s = np.linspace(I[0][0], I[0][1], 101)
        Y_s = g(X_s)      
        Figure2, Axes2 = CreatePlot()
        Axes2.plot(X_s, Y_s, color='red', label='y = g(x)')
        Axes2.plot(X_s, X_s, color='blue', label='y = x')
        
        for i in range(n):
            Axes2.plot([x_s[i], x_s[i]], [y_s[i], y_s[i + 1]], '--', color='black')
            Axes2.plot([x_s[i], x_s[i + 1]], [y_s[i + 1], y_s[i + 1]], '--', color='black')
        
        plt.xlabel('x')
        plt.ylabel('y') 
        plt.legend()
    
    return x_s, y_s, I, s

###############################################################################
# Laplace Operations

def Laplace(f):
    """Takes a SYMPY Function and returns the Laplace Transform F"""
    t, s  = sy.symbols('t s')
    F = sy.laplace_transform(f, t, s, noconds=True)
    return F


def InverseLaplace(F):
    """Takes a SYMPY Laplacian and returns the Inverse Laplace Transform f"""
    t, s  = sy.symbols('t s')    
    f = sy.inverse_laplace_transform(F, s, t)
    return f
    
###############################################################################
# Extrapolation

def Relu(x):
    return max(x, 0)


def LinearInterpolation(x_actual, x_lower, x_upper, y_lower, y_upper):
    """Takes known points on a line and finds linearly interpolated point on the line connecting the known points""" 
    f = (x_actual - x_lower)/(x_upper - x_lower)
    y_interpolate = (y_upper - y_lower)*f + y_lower
    return y_interpolate


def BestFitLine(x_s, y_s, Order=1):
    Coefficients = np.polyfit(x_s, y_s, Order)
    return Coefficients

###############################################################################
# Statistics and Combinatorics

def NormalDistribution(μ, σ):
    x_s = np.linspace(μ - 3*σ, μ + 3*σ, 101)
    p_s = np.exp(-0.5*((x_s - μ)/σ)**2)/(2*π*σ**2)
    return x_s, p_s


def PascalsTriangle(n, r):
    return math.comb(n, r) 

###############################################################################
# Algorithms

def Dijkstra(Network, Start, Target):
    """Takes an Array of Weights and returns the shortest path between the Starty and the Target"""
    Nodes = list(range(len(Network)))               # List of Node IDs    
    if Start > len(Nodes) or Target > len(Nodes):
        print('Check Node Inputs')
    Unvisited = Nodes                               # An updating list of Unvisited Nodes 
    Lengths = [np.inf] * len(Nodes)                 # An updating list of the smallest Length to each Node from the Start Node
    Lengths[Start] = 0                              # Start Node Length is 0
    Parents = [None] * len(Nodes)                   # An updating list of each Node's previous Node

    def NearestNode(Lengths, Unvisited):            
        MinLength = np.inf                          # Assume the smallest Length is inf
        MinNode = None                              
        for Node in Nodes:                          
            if Lengths[Node] < MinLength and Node in Unvisited:
                MinLength = Lengths[Node]           # If another Length to Node is less than current Length to Node: Update current Length in Lengths
                MinNode = Node 
        return MinNode
    
    def FindPath(Parents, Node):                    # Returns a list of Nodes that have been used to get to the Target
        Path = []
        while Node is not None:
            Path.append(Node)
            Node = Parents[Node]
        Path.reverse()    
        return Path

    while Unvisited: 
        NextNode = NearestNode(Lengths, Unvisited)  # The Next Node to visit is the nearest Node
#        print(Unvisited)
        if NextNode == None:
            print('Network is Incomplete')
            if Target in Unvisited:
                print('Target is Unreachable')
            Unvisited = []
        else:
            Unvisited.remove(NextNode)                  # This Node is now Visited
            for Node in Nodes: 
                if Node in Unvisited and Lengths[NextNode] + Network[NextNode][Node] < Lengths[Node]: 
                    Lengths[Node] = Lengths[NextNode] + Network[NextNode][Node] 
                    Parents[Node] = NextNode 

    Path = FindPath(Parents, Target)
    TargetArcs = []
    TargetLengths = []
    for Node in Path:
        ArcNodes = []
        try:
            NextNode = Path[Path.index(Node) + 1]
            TargetLengths.append(Network[Node][NextNode])
            ArcNodes.append(Node)
            ArcNodes.append(NextNode)
            TargetArcs.append(ArcNodes)
        except IndexError:
            pass
        
    return [TargetArcs, TargetLengths]

###############################################################################





