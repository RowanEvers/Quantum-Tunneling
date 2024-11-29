import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline
import time
from scipy.optimize import root_scalar
#start time 
st = time.time()

hbar = 1
m = 1
p = 20
dx = 1/(2*p)
sigma = 0.9
# define the discretised space
x_min = -20
x_max = 20
N = int((x_max-x_min)/dx)
xi = -4

x = np.linspace(x_min,x_max,N+1)
w = 2.5
h = 100

# define wavefunction
#time dependent part
def f(t,Ei):                    
    return np.exp(-1j*Ei*t/hbar)    

# inital value of whole wf
def create_Psi0(X,sig,P,x0):    
    Psi0 = np.exp(-(X[1:-1]+x0)**2/sig**2)*np.exp(1j*P*(X[1:-1]+x0))
    #normalise
    A = np.sum(np.abs(Psi0)**2*dx)
    Psi0 = Psi0/np.sqrt(A)
    return Psi0

#functions to create different potential barriers 
def r_pot_barrier(w,h,x0):
    #create a numpy list with 0 values, (0 potential everywhere)
    V = x*0      
    #add rectangular potential
    for i in range(len(V)):
        if x[i] >= x0 and x[i] <= x0 + w:
            V[i] = h
    return V

def parabolic_barrier(w,h):
    V = x*0
    for i in range(len(V)):
        if x[i]> -w/2  and x[i] < w/2:
            V[i] = -(x[i]**2)
    V_min = min(V)
    for i in range(len(V)):
        if x[i]> -w/2 and x[i]<w/2:
            V[i] = -(h/V_min)*(V[i] - V_min)
    return V
def dirac_del_barrier(x0):
    V = x*0
    for i in range(len(V)):
        if (x[i] > x0 - dx*0.5 and x[i]< x0 + dx*0.5):
            V[i] = 1/dx
            V[i+1] = 1/dx
    return V

def triangular_barrier(w,h,x0):
    V = x*0
    left = x0 - 0.5*w
    right = x0 + 0.5*w
    for i, xi in enumerate(x):
        if left <= xi <= x0:  # Left slope
            V[i] = h * (xi - left) / (x0 - left)
        elif x0 < xi <= right:  # Right slope
            V[i] = h * (right - xi) / (right - x0)
    return V

#solve schrodinger eqn at time t
def solve_SE(t,V,p):
    Psi0 = create_Psi0(x, sigma, p, -xi)
    # define Hamiltonian Matrix
    k = hbar**2/(m*dx**2)
    main_diag = k + V[1:-1]  # Main diagonal
    off_diag = -k * np.ones(N-2)  # Off-diagonal values

    # Use np.diag to construct the full dense matrix efficiently
    H = (
    np.diag(main_diag) +
    np.diag(off_diag, k=1) +
    np.diag(off_diag, k=-1)
        )
    # Compute all eigenvalues and eigenvectors
    
    E, psi = np.linalg.eigh(H)
    psi = psi.T

    #normalise
    A = np.sum(np.abs(psi[0]**2*dx))
    psi = psi/np.sqrt(A)

    #calculate coefficents
    c = Psi0*0
    for i in range(len(c)):
        c[i] = np.sum(np.conj(psi[i])*Psi0*dx)
    Psi = psi[0]*0*1j
    #print(Psi)
    for i in range(len(c)):
        Psi += (c[i]*psi[i]*f(t,E[i]))
    A = np.sum(np.abs(Psi)**2 * dx)
    Psi = Psi / np.sqrt(A)
    return Psi

#simulate

#largest range of momentum we found 
#V = dirac_del_barrier(0)*11.37 + dirac_del_barrier(1.07)*11.37

#can create other barriers to try 
V = parabolic_barrier(2, 70)

#find start and end points of potential 
end_val = 0
i = 0
end = False
while V[len(V)-i-1] == 0 and not end:
    i+=1
    end_val = N-i
    if i == len(V)-1:
        end = True
        
start_val = 0
i = 0
end = False
while V[i] == 0 and not end:
    i+=1
    start_val = i
    if i == len(V)-1:
        end = True
        
#calculate WF with a range of momenta
p_range = np.linspace(4,p,50)
transmission_co_list = []
for i in range(len(p_range)):
    hwp_width = sigma*5
    tfm = (m/p_range[i])*(xi-x_min-(hwp_width))
    Psi = solve_SE(tfm, V, p_range[i])
    reflect_co = np.sum(np.abs(Psi[1:start_val])**2*dx)
    transmission_co = np.sum(np.abs(Psi[end_val:-1])**2*dx)
    transmission_co_list.append((1/(transmission_co+reflect_co))*transmission_co)


#interpolate data 
cont_func = CubicSpline(p_range, transmission_co_list)


#find range of momentum where probibility is within a certain threshold 
thresh = 0.005
p_range = np.linspace(4,p,5000)
transmission_co_list = []
p_in_thresh = []
inner_list = []
for i in range(len(p_range)):
    transmission_co_list.append(cont_func(p_range[i]))
    if cont_func(p_range[i]) > 0.5 - thresh and cont_func(p_range[i]) < 0.5 + thresh:
        inner_list.append(p_range[i])
        #p_max = root_scalar(lambda x: cont_func(x) - (0.5+thresh), bracket=[p_range[i-1],p_range[i+1]])
        #p_min = root_scalar(lambda x: cont_func(x) - (0.5-thresh), bracket=[p_range[i-1],p_range[i+1]])
        if i == len(p_range)-1 or not cont_func(p_range[i+1]) > 0.5 - thresh or not cont_func(p_range[i+1]) < 0.5 + thresh:
            p_in_thresh.append(inner_list)
            inner_list = []
                                                                  
total_range = 0                                                                       
for i in range(len(p_in_thresh)):
    total_range += max(p_in_thresh[i]) - min(p_in_thresh[i])
    print('range',i+1,':',np.round(min(p_in_thresh[i]),5),'to',
          np.round(max(p_in_thresh[i]),5),
          '      del_p = ', np.round(max(p_in_thresh[i]) - min(p_in_thresh[i]),5)
          )
print('total range of p: ', total_range)
fig = plt.figure(figsize=(15, 8))

#plot momentum probability graph
pg = fig.add_subplot(2,1,1)
pg_handle, = pg.plot(p_range, transmission_co_list)
plt.ylabel('Tunnelling probibility')
plt.xlabel('Momentum')
hlf = 0*p_range + 0.5
plt.plot(p_range,hlf+thresh,linestyle='dotted')
plt.plot(p_range,hlf-thresh,linestyle='dotted')
plt.show()

#animate 
FPS = 24

#get different points in time 
hwp_width = sigma*5
#set p for animation 
p = 8
t = (m/p)*(xi-x_min-hwp_width)
t_pts = np.linspace(0,t,FPS)
Psi_pts = []
for i in range(len(t_pts)):
    Psi_pts.append(solve_SE(t_pts[i], V, p))

anig = fig.add_subplot(2,1,2)
y_max = max(Psi_pts[0])
def animate(i):
    anig.clear()
    Psi = Psi_pts[i]
    anig_handle, = anig.plot((x[1:-1]),np.abs(Psi))
    anig_handle, = anig.plot((x[1:-1]),np.real(Psi))
    anig.axes.set_ylim(0,3)
    plt.plot(x,2.5*V/max(V))
    
plt.tight_layout()

ani = animation.FuncAnimation(fig,animate,frames = len(t_pts),interval = 2000/FPS,repeat = True)

print('computation time', np.round(time.time()-st,2),'s')