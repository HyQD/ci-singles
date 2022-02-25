import numpy as np
from matplotlib import pyplot as plt
from quantum_systems import construct_pyscf_system_rhf
import basis_set_exchange as bse

atom = "h 0.0 0.0 0.0; h 0.0 0.0 1.385414814"
basis = "aug-cc-pvdz"
charge = 0

basis_set = bse.get_basis(basis, fmt='nwchem')

system = construct_pyscf_system_rhf(
    molecule=atom,
    basis=basis_set,
    symmetry=False,
    add_spin=False,
    charge=charge,
    cart=False,
    anti_symmetrize=False,
)
####################################################################################
# This solves part diagonlizes the CIS Hamiltonian H_{IJ} = <Phi_I|H|Phi_J>, I,J = 0,1,2,...,
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


def compute_density_matrix(C):

    rho = np.zeros(system.h.shape, dtype=np.complex128)
    nocc = occ.stop
    nvirt = virt.stop - nocc

    C_ai = C[1:].reshape(nvirt, nocc)

    rho[occ, occ] = 2 * Iocc * (
        C[0] * C[0].conj() + np.einsum("ia,ai->", C_ai.T.conj(), C_ai)
    ) - np.einsum("ja, ai->ji", C_ai.conj().T, C_ai)

    rho[virt, occ] = np.sqrt(2) * C[0].conj() * C_ai
    rho[occ, virt] = np.sqrt(2) * C[0] * C_ai.T.conj()
    rho[virt, virt] = np.einsum("ia,bi->ba", C_ai.T.conj(), C_ai)

    return rho


rho_gs = compute_density_matrix(C[:, 0])
rho_x1 = compute_density_matrix(C[:, 1])

print(np.trace(rho_gs))
print(np.trace(rho_x1))
