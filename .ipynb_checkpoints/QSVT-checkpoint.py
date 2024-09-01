import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from pyqsp.angle_sequence import angle_sequence
from qiskit.quantum_info import Operator
import scipy
from pyqsp.angle_sequence import QuantumSignalProcessingPhases

'''Exact Methods'''
### Get angles that exactly implement P, given k,P,Q
def get_angles(k, P, Q):
    if not( (len(P)-1 - k) % 2 == 0 and (len(Q)-k) % 2 == 0):
        raise ValueError("Check Polynomial Pairity")
    if not( (len(P)-1 <=k) and (len(Q)-1 <= k-1)):
        raise ValueError("Check Polynomial Degree")
    
    extra = np.pi/2 * (-1)**np.arange(0,k+1-len(P))
    phi_list = []
    p_temp = np.array(P,dtype='complex128')
    q_temp = np.array(Q,dtype='complex128')
    
    for i in range(len(P)-1):
        
        phi = np.angle(p_temp[-1]/q_temp[-1])/2
        phi_list = np.concatenate(([phi],phi_list))
        
        p_temp_copy = p_temp
        q_temp_copy = q_temp
        
        p_temp=np.exp(-1j*phi) *(np.concatenate(([0], p_temp_copy)) + np.exp(2*1j*phi) * np.concatenate((q_temp_copy,[0,0]))-np.exp(2*1j*phi) * np.concatenate(([0,0],q_temp_copy)))[:-2]
        q_temp=np.exp(-1j*phi) * (np.exp(2*1j*phi) * np.concatenate(([0],q_temp_copy)) -p_temp_copy)[:-2]
    phi = np.angle(p_temp[-1])
    phi_list = np.concatenate(([phi],phi_list))
    return np.concatenate((phi_list,extra))

###Get Q given admissible complex P (P(ix) * P*(ix) >=1 )
def getQ(P):
    A  = (1-np.polynomial.Polynomial(np.array(P, dtype='complex128'))*np.polynomial.Polynomial(np.array(P, dtype='complex128').conjugate())).coef
    A_tilde = A[np.arange(0,len(P)*2,2)]
    if(len(P) % 2 == 0):
        A_tilde1 = np.polydiv(A_tilde,[1,-1])[0]
        K = np.sqrt(A_tilde1[-1])
        roots = np.polynomial.Polynomial(A_tilde1).roots()
        W = K*np.polynomial.polynomial.polyfromroots(roots[roots.imag.argsort()][:len(roots)//2])
        return np.insert(arr = W, obj=range(1,len(W)), values=0)
    else:
        A_tilde1 = np.polydiv(A_tilde[1:],[1,-1])[0]
        K = np.sqrt(A_tilde1[-1])
        roots = np.polynomial.Polynomial(A_tilde1).roots()
        W = K*np.polynomial.polynomial.polyfromroots(roots[roots.imag.argsort()][:len(roots)//2])
        return np.concatenate(([0],np.insert(arr = W, obj=range(1,len(W)), values=0)))


def getAngles(P):
    return np.array(get_angles(len(P)-1, P, getQ(P)))
    

'''Real Part Methods'''
def getRealAngles(P):
    return np.array(QuantumSignalProcessingPhases(P, eps=1e-6, suc=1-1e-5,signal_operator="Wx"))


'''W->R Angles Change'''
def get_new_angles(angles):
    new_angles = [angles[0]+angles[-1]+(len(angles)-2)*np.pi/2]
    for i in angles[1:-1]:
        new_angles.append(i-np.pi/2)
    return np.array(new_angles)

'''QSP Functions (Matrix Only)'''
def R_op(x):
    return np.array([[x,np.sqrt(1-x**2)],[np.sqrt(1-x**2),-x]])

def W_op(x):
    return np.array([[x,1j * np.sqrt(1-x**2)],[1j * np.sqrt(1-x**2),x]])

def Rfunc(angles,x):
    op = np.array([[1,0],[0,1]],dtype='complex128')
    for i in angles:
        op @= np.array([[np.exp(1j * i),0],[0,np.exp(-1j * i)]])
        op @= R_op(x)
    return op

def Wfunc(angles,x):
    op = np.array([[np.exp(1j * angles[0]),0],[0,np.exp(-1j * angles[0])]])
    for i in angles[1:]:
        op @= W_op(x)
        op @= np.array([[np.exp(1j * i),0],[0,np.exp(-1j * i)]])
    return op

'''QSVT Functions (Matrix Only)'''
def projector(rank,n_qubits):
    return np.diag([1] * rank + [0] * (2**n_qubits-rank))

def U_phi(angles,U,left_rank,right_rank):
    n_qubits=int(np.log2(U.shape[0]))
    
    tilde_Pi =projector(left_rank,n_qubits)
    Pi =projector(right_rank,n_qubits)
    
    Op = np.eye(2**n_qubits,dtype='complex128')
    ###Odd parity
    if(len(angles) % 2 == 1):
        for i,v in enumerate(angles):
            if(i % 2 == 0):
                Op @= scipy.linalg.expm(1j * v * (2 * tilde_Pi - np.eye(2**n_qubits)))
                Op @= U
            else:
                Op @= scipy.linalg.expm(1j * v * (2 * Pi - np.eye(2**n_qubits)))
                Op @= U.conjugate().T

    ###Even Parity
    else:
        for i,v in enumerate(angles):
            if(i % 2 == 1):
                Op @= scipy.linalg.expm(1j * v * (2 * tilde_Pi - np.eye(2**n_qubits)))
                Op @= U
            else:
                Op @= scipy.linalg.expm(1j * v * (2 * Pi - np.eye(2**n_qubits)))
                Op @= U.conjugate().T
    return Op
    
def QSVTfunc(angles,U,left_rank,right_rank):
    
    n_qubits=int(np.log2(U.shape[0]))
    
    ###Odd parity
    if(len(angles) % 2 == 1):
        return U_phi(angles,U,left_rank,right_rank)[:left_rank,:right_rank]

    ###Even Parity
    else:
        return U_phi(angles,U,left_rank,right_rank)[:right_rank,:right_rank]

def QSVTRealfunc(angles,U,left_rank,right_rank):
    
    n_qubits=int(np.log2(U.shape[0]))
    
    Op_plus = U_phi(angles,U,left_rank,right_rank)
    Op_minus =U_phi(-angles,U,left_rank,right_rank)
    
    H = np.sqrt(0.5) * np.kron(np.array([[1,1],[1,-1]]),np.eye(2**n_qubits,dtype='complex128'))
    combined_Op = H @ (np.kron(np.array([[1,0],[0,0]]),Op_plus) + np.kron(np.array([[0,0],[0,1]]),Op_minus)) @ H

    ###Odd parity
    if(len(angles) % 2 == 1):
        return  combined_Op[:left_rank,:right_rank]

    ###Even Parity
    else:
        return combined_Op[:right_rank,:right_rank]
    return 

def PolySV(P,A):
    V,S,W=scipy.linalg.svd(A)
    W=W.conjugate().T

    ###Even degree
    if(len(P) % 2 == 0):
        M=np.zeros(A.shape,dtype='complex128')
        for i in range(len(S)):
            M += np.outer(V[:,i],W[:,i].conjugate())*np.polynomial.Polynomial(P)(S[i]) 
        return M
    else:
        n=A.shape[-1]
        M=np.zeros((n,n),dtype='complex128')
        for i in range(n):
            if(i<len(S)):
                M += np.outer(W[:,i],W[:,i].conjugate())*np.polynomial.Polynomial(P)(S[i]) 
            else:
                M += np.outer(W[:,i],W[:,i].conjugate())*P[0]
        return M

'''Circuit Methods'''
def QSP_Wcircuit(angles,x):
    qc = QuantumCircuit(1)
    for i,v in enumerate(angles[::-1]):
        qc.rz(-2 * v, 0)
        if(i < len(angles)-1):
            qc.rx(-np.arccos(x) * 2, 0 )
    return qc

def QSP_Rcircuit(angles,x):
    qc = QuantumCircuit(1)
    for i,v in enumerate(angles[::-1]):
        qc.global_phase -= np.pi/2
        qc.rz(-np.pi/2, 0 )
        qc.rx(-np.arccos(x) * 2, 0 )
        qc.rz(-np.pi/2, 0 )
        qc.rz(-2 * v, 0)
    return qc
    
def get_projector(rank,n_qubits):
    Pi = np.diag([1] * rank + [0] * (2**n_qubits-rank))
    C_Pi = np.kron(np.array([[0,1],[1,0]]),Pi)+np.kron(np.eye(2),np.eye(2**n_qubits)-Pi)
    qc=QuantumCircuit(n_qubits+1)
    qc.unitary(Operator(C_Pi),range(n_qubits+1)[::-1])
    return qc.to_gate()

def exp_projector(rank,n_qubits,phi):
    qc = QuantumCircuit(n_qubits+1)
    qc.append(get_projector(rank,n_qubits),range(n_qubits+1))
    qc.rz(2*phi,0)
    qc.append(get_projector(rank,n_qubits),range(n_qubits+1))
    return qc

def QSVTcircuit(angles,U_circ,left_rank,right_rank):
    n_qubits=U_circ.num_qubits
    qc=QuantumCircuit(n_qubits+1)
    for i,v in enumerate(angles[::-1]):
        if(i % 2 == 1):
            qc=qc.compose(U_circ.inverse(),range(1,n_qubits+1))
            qc=qc.compose(exp_projector(right_rank,n_qubits,v),range(n_qubits+1))
        else:
            qc=qc.compose(U_circ,range(1,n_qubits+1))
            qc=qc.compose(exp_projector(left_rank,n_qubits,v),range(n_qubits+1))
    return qc

def QSVTRealcircuit(angles,U_circ,left_rank,right_rank):
    n_qubits=U_circ.num_qubits
    qc=QuantumCircuit(n_qubits+1)
    qc.h(0)

    for i,v in enumerate(angles[::-1]):
        if(i % 2 == 0):
            qc=qc.compose(U_circ,range(1,n_qubits+1))
            qc=qc.compose(exp_projector(left_rank,n_qubits,v),range(n_qubits+1))
        else:
            qc=qc.compose(U_circ.inverse(),range(1,n_qubits+1))
            qc=qc.compose(exp_projector(right_rank,n_qubits,v),range(n_qubits+1))

    qc.h(0)
    return qc

'''Full QSVT (only circuit for now)'''
def poly_decomp(P):
    poly =  np.array(P,dtype='complex128')
    flip = np.array([(-1)**v for v in range(len(P))])
    polys = []
    polys.append(poly + flip * poly + poly.conjugate() + flip * poly.conjugate())
    polys.append(-1j * (poly + flip * poly - poly.conjugate() - flip * poly.conjugate()))
    polys.append(poly - flip * poly + poly.conjugate() - flip * poly.conjugate())
    polys.append(-1j * (poly - flip * poly - poly.conjugate() + flip * poly.conjugate()))

    polys = np.array(polys).real

    for i in range(len(P)):
        if(i % 2 == 1):
            polys[0][i]=0
            polys[1][i]=0
        else:
            polys[2][i]=0
            polys[3][i]=0

    if(len(P) % 2==1):
        return [polys[0],polys[1],polys[2][:-1],polys[3][:-1]]
    if(len(P) % 2==0):
        return [polys[0][:-1],polys[1][:-1],polys[2],polys[3]]
        
def norm(P,samples=1000):
    return P/(4*np.max(abs(np.polynomial.Polynomial(P)(np.linspace(-1,1,samples)))))

def ccrz_circuit(phi):
    qc=QuantumCircuit(3)
    qc.ccx(0,1,2)
    qc.rz(-phi,2)
    qc.ccx(0,1,2)
    qc.rz(phi,2)
    return qc

def FourAngleRotCirc(phi):
    qc=QuantumCircuit(3)
    qc=qc.compose(ccrz_circuit(phi[3]),range(3))
    qc.x(0)
    qc=qc.compose(ccrz_circuit(phi[1]),range(3))
    qc.x(1)
    qc=qc.compose(ccrz_circuit(phi[0]),range(3))
    qc.x(0)
    qc=qc.compose(ccrz_circuit(phi[2]),range(3))
    qc.x(1)
    return qc
    

def four_exp_projectorcirc(rank,n_qubits,phi):
    qc = QuantumCircuit(n_qubits+3)
    qc.append(get_projector(rank,n_qubits),range(2,n_qubits+3))
    qc=qc.compose(FourAngleRotCirc(phi),[0,1,2])
    qc.append(get_projector(rank,n_qubits),range(2,n_qubits+3))
    return qc

def FullQSVTFunc(angles,U_circ,rank):
    n_qubits=U_circ.num_qubits
    qc=QuantumCircuit(n_qubits+3)
    qc.h([0,1,2])
    qc.x(2)
    qc.cs(1,2)
    qc.x(2)
    qc.cs(1,2)
    ###Odd Case
    if(len(angles[0]) < len(angles[2])):
        temp_angles = np.array([np.concatenate(([0],angles[0])),np.concatenate(([0],angles[1])),angles[2],angles[3]])
        for i,v in enumerate(temp_angles.T[::-1]):
            if(i < len(temp_angles.T)-1):
                if(i % 2 == 0):
                    qc=qc.compose(U_circ,range(3,n_qubits+3))
                    qc=qc.compose(four_exp_projectorcirc(rank,n_qubits,v),range(n_qubits+3))
                else:
                    qc=qc.compose(U_circ.inverse(),range(3,n_qubits+3))
                    qc=qc.compose(four_exp_projectorcirc(rank,n_qubits,v),range(n_qubits+3))
            else:
                qc=qc.compose(U_circ.control(1),[0]+list(range(3,n_qubits+3)))
                qc=qc.compose(four_exp_projectorcirc(rank,n_qubits,v),range(n_qubits+3))
                
            
    ###Even Case Case
    else:
        temp_angles = np.array([angles[0],angles[1],np.concatenate(([0],angles[2])),np.concatenate(([0],angles[3]))])
        for i,v in enumerate(temp_angles.T[::-1]):
            if(i < len(temp_angles.T)-1):
                if(i % 2 == 0):
                    qc=qc.compose(U_circ,range(3,n_qubits+3))
                    qc=qc.compose(four_exp_projectorcirc(rank,n_qubits,v),range(n_qubits+3))
                else:
                    qc=qc.compose(U_circ.inverse(),range(3,n_qubits+3))
                    qc=qc.compose(four_exp_projectorcirc(rank,n_qubits,v),range(n_qubits+3))
            else:
                qc.x(0)
                qc=qc.compose(U_circ.inverse().control(1),[0]+list(range(3,n_qubits+3)))
                qc.x(0)
                qc=qc.compose(four_exp_projectorcirc(rank,n_qubits,v),range(n_qubits+3))
    qc.h([0,1,2])
    return qc