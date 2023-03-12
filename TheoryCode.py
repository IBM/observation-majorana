# Theory code
# Author: Oles Shtanko

import numpy as np
from numpy import logical_and as AND
from numpy import logical_not as NOT
from numpy import logical_or as OR
from scipy.linalg import expm
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])   
H = np.array([[1,1],[1,-1]])/np.sqrt(2) 
Sdg = np.array([[1,0],[0,-1j]]) 
XX = np.kron(X,X)
ZZ = np.kron(Z,Z)
pauli_matrices = [I,X,Y,Z]

def show_mode(n,cycles,theta_frac,phi_frac,eta_frac=0,tp='MZM'):

    # defining string operators
    zx_pos_list = np.empty(n,np.ndarray)
    zx_pauli_list = np.empty(n,np.ndarray)
    for x in range(n):
        zx_pos_list[x] = np.arange(x+1)
        zx_pauli_list[x] = 3*np.ones(x+1,int)
        zx_pauli_list[x][-1] = 1

    zy_pos_list = np.empty(n,np.ndarray)
    zy_pauli_list = np.empty(n,np.ndarray)
    for x in range(n):
        zy_pos_list[x] = np.arange(x+1)
        zy_pauli_list[x] = 3*np.ones(x+1,int)
        zy_pauli_list[x][-1] = 2
    
    # evaluating phase values
    theta = theta_frac*np.pi
    phi = phi_frac*np.pi
    eta = eta_frac*np.pi

    qstate = quantum_state(n)
    # initiate in |+> state
    qstate.psi = np.ones(2**n)/2**n

    # define the left mode string
    zx_string = np.zeros([cycles,n])
    zy_string = np.zeros([cycles,n])

    # define the gates
    uz  = expm(-1j*phi*Z)
    uxx = expm(-1j*theta*np.kron(X,X))
    uzz = expm(-1j*eta*np.kron(Z,Z))

    for ti in range(cycles):

        # evaluate the values of strings
        for x in range(n):
            zx_string[ti][x] = find_observable(state = qstate,x_list = zx_pos_list[x],
                                              pauli_list = zx_pauli_list[x]).real
            zy_string[ti][x] = find_observable(state = qstate,x_list = zx_pos_list[x],
                                              pauli_list = zy_pauli_list[x]).real
        # apply circuit layer
        for x in range(n):
            qstate.apply_1qubit_gate(uz,x)
        for x in range(n-1):
            qstate.apply_2qubit_gate(uxx,x,x+1)
        if eta>0:
            for x in range(n-1):
                qstate.apply_2qubit_gate(uzz,x,x+1)

    if tp=='MZM':
        zx_string_fourier = np.sum(zx_string,axis=0)
        zy_string_fourier = np.sum(zy_string,axis=0)
        
    if tp=='MPM':
        zx_string_fourier = np.sum(zx_string[::2],axis=0)-np.sum(zx_string[1::2],axis=0)
        zy_string_fourier = np.sum(zy_string[::2],axis=0)-np.sum(zy_string[1::2],axis=0)

    # plotting (expected) experimental curves
    
    norm1 = zx_string_fourier[0]
    # ax.plot(np.abs(zx_string_fourier)/norm1,marker = 'o',ls='None',zorder=1,label = 'even mode, measured')
    # ax.plot(np.abs(zy_string_fourier)/norm1,marker = 'o',ls='None',zorder=1,label = 'odd mode, measured')

    # plotting free-fermion theory curves
    
    psiR,psiL = majorana_modes(n,phi=phi_frac*np.pi,theta=theta_frac*np.pi,mode = tp,eps=0.1)
    
    norm2 = np.abs(psiL[::2])[0]
    # ax.plot(np.abs(psiL[::2])/norm2,c='k',label = 'even mode, theory',zorder = 0)
    # ax.plot(np.abs(psiL[1::2])/norm2,c='k',ls="--",label = 'odd mode, theory',zorder=0)
    # ax.set_title(tp+r' mode, $\theta=$'+str(theta_frac)+'$\pi$, $\phi=$'+str(phi_frac)+'$\pi$')
    # ax.legend()
    return np.abs(zx_string_fourier)/norm1, np.abs(zy_string_fourier)/norm1
    
def logical_zeros(n,x):
    # group sizes
    d1,d2 = 2**(n-x-1),2**x
    logic0  = np.tile(np.repeat([True,False],d1),d2)
    return logic0

def shr_circuit(size,depth,theta,phi,x=0,basis ='x'):
    
    circ = circuit_1d(size,depth)
    uZ  = expm(-1j*phi*Z)
    uXX = expm(-1j*theta*XX)
    Uc  = expm(-1j*np.pi/4*np.kron(Y,X))
    Ucy = expm(+1j*np.pi/4*np.kron(X,Y))
    circ.gate_alphabet = [uZ,H,uXX,Uc,Sdg,Ucy]
    circ.gate_size = [2,2,4,4,2,4]
    
    #circuit = np.ones([size,depth],int)
    circuit = np.zeros([depth,size],int)
    #applies first set of 2q gates
    circuit[1::3].T[::2] = 3
    #applies second set of 2q gates
    circuit[2::3].T[1::2] = 3
    circuit.T[-1] = 0
    circuit[::3] = 1
    for i in range(x):
        layer = np.zeros(size,int)
        if basis == 'y' or basis == 'Y':
            layer[x-i-1] = 6
        else:
            layer[x-i-1] = 4
        circuit = np.vstack((circuit,layer))
    if basis == 'y' or basis == 'Y':
        #next three lines add SDG gate to measurement qubit
        layer = np.zeros(size,int)
        layer[0] = 5
        circuit = np.vstack((circuit,layer))
    #next three lines add Hadamard gate to measurement qubit
    layer = np.zeros(size,int)
    layer[0] = 2
    circuit = np.vstack((circuit,layer))
    # next two lines add Hadamard gates
    layer = 2*np.ones(size,int)
    circuit = np.vstack((layer,circuit))
    circ.circuit = circuit
    return circ

def observable(size,depth_arr,theta,phi,x0,basis = 'x'):
    x_av = np.zeros(len(depth_arr))
    for i in range(len(depth_arr)):
        depth = 3*depth_arr[i]
        circ = shr_circuit(size,depth,theta,phi,x=x0,basis=basis)
        qstate = quantum_state(size)
        qstate.apply(circ)
        p0 = qstate.prob_0(x=0)
        x_av[i] = 2*p0-1 
    return x_av

def find_observable(state,x_list,pauli_list):
    state_temp = quantum_state(n = state.size)
    state_temp.psi = state.psi.copy()
    psi_bra = state_temp.psi.conj()
    for i in range(len(pauli_list)):
        x,O = x_list[i],pauli_matrices[pauli_list[i]]
        state_temp.apply_1qubit_gate(O,x)
    psi_ket = state_temp.psi
    return np.dot(psi_bra,psi_ket)

class circuit_1d():
    
    def __init__(self,size,depth):
        self.size = size
        self.depth = depth
        self.circuit = np.zeros([depth,size],int)
        
    def show(self,ax):
        gate_color = ['green','yellow','blue','red']
        for x in range(self.size):
            ax.plot([-1,2*len(self.circuit)-1],[x,x],c='k',zorder=0)
        patches = []
        col = []
        for d in range(len(self.circuit)):
            for x in range(len(self.circuit.T)):
                if self.circuit[d,x]>0:
                    sq=int(self.gate_size[self.circuit[d,x]-1]==2)
                    x0,y0,xs,ys = 2*d,x+0.5-0.5*sq,1,1.75-1*sq
                    rect = mpatches.Rectangle([x0-xs/2,y0-ys/2], 
                                              xs, ys, edgecolor = 'k')
                    patches.append(rect)
                    col.append(gate_color[self.circuit[d,x]-1])
        collection = PatchCollection(patches, cmap=plt.cm.hsv,facecolor=col,
                                     edgecolor = 'k',zorder=1)
        ax.add_collection(collection)
        ax.set_ylim(-1,len(self.circuit.T)+1)
        ax.set_axis_off()
        
class quantum_state():
    
    def __init__(self,n):
        self.size = n
        self.psi = np.zeros(2**n,complex)
        self.psi[0] = 1

    def apply_1qubit_gate(self,u,x):
        # step 1: defining logical basis of two qubits A and B
        logic0 = logical_zeros(self.size,x)
        logic1 = NOT(logic0)
        # step2: performing the unitary transform
        psi_new = 0j*self.psi
        psi_new[logic0] = u[0,0]*self.psi[logic0]+u[0,1]*self.psi[logic1]
        psi_new[logic1] = u[1,0]*self.psi[logic0]+u[1,1]*self.psi[logic1]
        self.psi = psi_new.copy()
       
    def apply_2qubit_gate(self,u,x1,x2):
        # step 1: defining logical basis of two qubits A and B
        logic0A = logical_zeros(self.size,x1)
        logic0B = logical_zeros(self.size,x2)
        logic1A,logic1B = NOT(logic0A),NOT(logic0B)
        logic00 = AND(logic0A,logic0B)
        logic01 = AND(logic0A,logic1B)
        logic10 = AND(logic1A,logic0B)
        logic11 = AND(logic1A,logic1B)
        # step2: performing the unitary transform
        psi_new = 0j*self.psi
        psi_new[logic00] = u[0,0]*self.psi[logic00]+u[0,1]*self.psi[logic01]+\
                           u[0,2]*self.psi[logic10]+u[0,3]*self.psi[logic11]
        psi_new[logic01] = u[1,0]*self.psi[logic00]+u[1,1]*self.psi[logic01]+\
                           u[1,2]*self.psi[logic10]+u[1,3]*self.psi[logic11]
        psi_new[logic10] = u[2,0]*self.psi[logic00]+u[2,1]*self.psi[logic01]+\
                           u[2,2]*self.psi[logic10]+u[2,3]*self.psi[logic11]
        psi_new[logic11] = u[3,0]*self.psi[logic00]+u[3,1]*self.psi[logic01]+\
                           u[3,2]*self.psi[logic10]+u[3,3]*self.psi[logic11]
        self.psi = psi_new.copy() 
    
    def prob_0(self,x):
        logic0 = logical_zeros(self.size,x)
        p0 = np.sum(np.abs(self.psi[logic0])**2)
        return p0
    
    def apply(self,circ):
        for d in range(len(circ.circuit)):
            for x in range(self.size):
                if circ.circuit[d,x] !=0:
                    u = circ.gate_alphabet[circ.circuit[d,x]-1]
                    dim = len(u)
                    if dim == 2:
                        self.apply_1qubit_gate(u,x)
                    if dim == 4:
                        self.apply_2qubit_gate(u,x,(x+1)%self.size)


## JW_CALCULUS

def unitary(M):
    E,Q = np.linalg.eigh(M)
    return np.dot(Q,np.dot(np.diag(np.exp(-1j*E)),Q.T.conj()))

def spectrum_fermion_modes(N,phi_values,theta_values):

    Hxx = 1j*(np.float_(np.diag(np.arange(2*N-1)%2==1,+1))\
              -np.float_(np.diag(np.arange(2*N-1)%2==1,-1)))
    Hz  = 1j*(np.float_(np.diag(np.arange(2*N-1)%2==0,+1))\
              -np.float_(np.diag(np.arange(2*N-1)%2==0,-1)))
                
    dim = len(phi_values)
    spectrum = np.zeros([dim,2*N])
    for pi in range(dim):
        phi = phi_values[pi]
        theta = theta_values[pi]
        Uxx = unitary(-2*theta*Hxx)
        Uz  = unitary(-2*phi*Hz)
        UF = np.dot(Uz,Uxx)
        E = np.linalg.eigvals(UF)
        spectrum[pi] = np.sort(np.angle(E))
        
    return spectrum
            
def majorana_modes(N,phi,theta,mode,eps = 1e-1):
    
    Hxx = 1j*(np.float_(np.diag(np.arange(2*N-1)%2==1,+1))\
              -np.float_(np.diag(np.arange(2*N-1)%2==1,-1)))
    Hz  = 1j*(np.float_(np.diag(np.arange(2*N-1)%2==0,+1))\
              -np.float_(np.diag(np.arange(2*N-1)%2==0,-1)))

    Uxx = unitary(-2*theta*Hxx)
    Uz  = unitary(-2*phi*Hz)
    UF = np.dot(Uz,Uxx)
    
    E,Q = np.linalg.eig(UF)
    
    if mode=='MZM':
        
        indx_zero = np.argwhere(np.abs(np.angle(E))<eps).flatten()
        
        if len(indx_zero)==2:
            
            psi1 = Q.T[indx_zero[0]]
            psi2 = Q.T[indx_zero[1]]
            
            # approximate separation on left (L) and right (R) mode
            
            psiR = psi2[0]*psi1-psi1[0]*psi2
            psiL = psi2[-1]*psi1-psi1[-1]*psi2
            
            return psiR, psiL
            
        else:
            
            print('No MZM found with given splitting, e='+str(eps))
            
    if mode=='MPM':
        
        indx_zero = np.argwhere(OR(np.abs(np.angle(E)-np.pi)<eps,
                                   np.abs(np.angle(E)+np.pi)<eps)).flatten()
        
        if len(indx_zero)==2:
            
            psi1 = Q.T[indx_zero[0]]
            psi2 = Q.T[indx_zero[1]]
            
            # approximate separation on left (L) and right (R) mode
            
            psiR = psi2[0]*psi1-psi1[0]*psi2
            psiL = psi2[-1]*psi1-psi1[-1]*psi2
    
            return psiR, psiL
            
        else:
            
            print('No MPM found with given splitting, e='+str(eps))  


def show_mode_new(n,cycles,theta_frac,phi_frac,eta_frac=0,decoupled = 1):
    
    # defining operator strings
    gg_postn = np.empty(n,np.ndarray)
    gg_pauli = np.empty(n,np.ndarray)
    gg_pauli[0] = [3]
    gg_postn[0] = [0]
    for i in range(1,n):
        gg_postn[i] = np.arange(i+1)
        gg_pauli[i] = 3*np.ones(i+1,int)
        gg_pauli[i][0] = 2
        gg_pauli[i][i] = 2
    
    # evaluating phase values
    theta = theta_frac*np.pi
    phi = phi_frac*np.pi
    eta = eta_frac*np.pi

    # defining initial state
    qstate = quantum_state(n)
    
    # setting wavefunction
    psi = 1
    v1 = np.array([2,1j])/np.sqrt(5)
    for i in range(n-2):
        a = np.random.randint(2)
        psi = np.kron([a,1-a],psi)
    psi = np.kron(v1,psi)
    psi = np.kron(psi,v1)
    qstate.psi = psi.copy()

    # -- define the left mode string
    gg_str = np.zeros([cycles,n])

    # -- define the gates
    uz  = expm(-1j*phi*Z)
    uxx = expm(-1j*theta*np.kron(X,X))
    uzz = expm(-1j*eta*np.kron(Z,Z))

    phi_edge = 0.1*phi
    theta_edge = 0.5*theta
    uz_edge  = expm(-1j*phi_edge*Z)
    uxx_edge  = expm(-1j*theta_edge*np.kron(X,X))
            
    for ti in range(cycles):
    
        # evaluate the values of strings
        for i in range(n):
            gg_str[ti][i] = find_observable(state = qstate,x_list = gg_postn[i],pauli_list = gg_pauli[i]).real
            
        # apply circuit layer
        if decoupled:
            qstate.apply_1qubit_gate(uz_edge,0)
            qstate.apply_1qubit_gate(uz_edge,n-1)
        for x in range(decoupled,n-decoupled):
            qstate.apply_1qubit_gate(uz,x)
        if decoupled:
            qstate.apply_2qubit_gate(uxx_edge,0,1)
            qstate.apply_2qubit_gate(uxx_edge,n-2,n-1)
        for x in range(decoupled,n-1-decoupled):
            qstate.apply_2qubit_gate(uxx,x,x+1)
        if eta>0:
            for x in range(n-1):
                qstate.apply_2qubit_gate(uzz,x,x+1)
                
    gg_str_fourier = np.sum(gg_str,axis=0)/cycles

    return gg_str_fourier,gg_str