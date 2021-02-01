from pyscf import gto, scf, ao2mo, fci, cc
import numpy as np
from matplotlib import pyplot as plt
from quantum_systems import construct_pyscf_system_rhf


atom = "h 0.0 0.0 0.0; h 0.0 0.0 1.385414814"

basis = "aug-cc-pvdz"
charge = 0

system = construct_pyscf_system_rhf(
    molecule=atom,
    basis=basis,
    symmetry=False,
    add_spin=False,
    charge=charge,
    cart=False,
    anti_symmetrize=False,
)


####################################################################################
#This solves part diagonlizes the CIS Hamiltonian H_{IJ} = <Phi_I|H|Phi_J>, I,J = 0,1,2,...,
occ = system.o
virt = system.v

f = system.construct_fock_matrix(system.h, system.u)

e_ref = 2 * np.einsum("ii->", system.h[occ, occ])
e_ref += 2 * np.einsum("ijij->", system.u[occ, occ, occ, occ])
e_ref -= np.einsum("ijji->", system.u[occ, occ, occ, occ])

dim_cis = 1 + system.m * system.n
print(f"dim(CIS)={dim_cis}")
Hcis = np.zeros((dim_cis, dim_cis))

Hcis[0, 0] = e_ref
Hcis[0, 1:] = np.sqrt(2) * f[virt, occ].ravel()
Hcis[1:, 0] = np.sqrt(2) * f[occ, virt].ravel()

Iocc = np.eye(system.n)
Ivirt = np.eye(system.m)

term1 = np.einsum("ab,ij->bjai", Ivirt, Iocc) * e_ref
term2 = np.einsum("ij,ab->bjai", Iocc, f[virt, virt])
term3 = np.einsum("ab,ij->bjai", Ivirt, f[occ, occ])
term4 = 2 * np.einsum("bija->bjai", system.u[virt, occ, occ, virt])
term4 -= np.einsum("biaj->bjai", system.u[virt, occ, virt, occ])
term = term1 + term2 - term3 + term4
term = term.reshape((system.m * system.n, system.m * system.n))
Hcis[1:, 1:] = term

eps, C = np.linalg.eigh(Hcis)
####################################################################################

import sys
sys.path.append('../ci_singles/')

from scipy.integrate import complex_ode
from tdcis import TDCIS
from quantum_systems.time_evolution_operators import LaserField

class linear_laser:

    def __init__(self, E0, omega):
        self.E0 = E0
        self.omega = omega
        self.t_c = 2*np.pi/omega

    def __call__(self,t):
        if (0 <= t < self.t_c):
            return self.omega*t/(2*np.pi)*self.E0*np.sin(self.omega*t)
        elif(self.t_c <= t < 2*self.t_c):
            return self.E0*np.sin(self.omega*t)
        elif(2*self.t_c <= t < 3*self.t_c):
            return (3-self.omega*t/(2*np.pi))*self.E0*np.sin(self.omega*t)
        else:
            return 0

F_str = 0.12
omega = 0.06
t_cycle = 2 * np.pi / omega
print(f"Field strength={F_str}")
print(f"Carrier frequency={omega}, 1 optical cycle={t_cycle}")

tprime = 3*t_cycle

time_after_pulse = 0
tfinal = np.floor(tprime + time_after_pulse)

pulse = linear_laser(F_str, omega)

polarization = np.zeros(3)
polarization_direction = 2
polarization[polarization_direction] = 1
system.set_time_evolution_operator(
    LaserField(pulse, polarization_vector=polarization)
)

tdcis = TDCIS(system)

r = complex_ode(tdcis).set_integrator("vode")
y0 = np.complex128(np.zeros(dim_cis))
y0[0] = 1 + 0j
r.set_initial_value(y0)

dt = 1e-2
num_steps = int(tfinal / dt) + 1
print(f"num_steps={num_steps}")
time_points = np.linspace(0, tfinal, num_steps)
dipole_moment = np.zeros(num_steps, dtype=np.complex128)

rho = np.zeros(system.h.shape, dtype=np.complex128)
nocc = occ.stop
nvirt = virt.stop - nocc


for i in range(num_steps - 1):
    
    Ct = r.integrate(r.t + dt)

    Ct_ai = Ct[1:].reshape(nvirt, nocc)
    
    rho[occ, occ] = (
        2 * Ct[0] * Ct[0].conj() * Iocc
        - np.einsum("ak,al->kl", Ct_ai, Ct_ai.conj())
        + 2 * Iocc * np.einsum("ai,ai->", Ct_ai, Ct_ai.conj())
    )
    
    rho[virt, occ] = np.sqrt(2) * Ct[0] * Ct_ai.conj()
    rho[occ, virt] = rho[virt, occ].T.conj()
    rho[virt, virt] = np.einsum("di,ci->cd", Ct_ai, Ct_ai.conj())

    dipole_moment[i + 1] = np.trace(
        np.dot(rho, system.dipole_moment[polarization_direction])
    )

    if(i%1000==0):
        print(i)
    

plt.figure()
plt.plot(time_points, -dipole_moment.real)
plt.grid()

plt.show()
