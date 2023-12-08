class CIS:
    def __init__(self, system, **kwargs):

        self.system = system
        self.np = self.system.np

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m
        self.o = self.system.o
        self.v = self.system.v

        self.num_states = 1 + system.m * system.n

        self.h = system.h
        self.u = system.u
        self.f = system.construct_fock_matrix(system.h, system.u)

    def compute_one_body_density_matrix_IJ(self, I, J):
        np = self.np

        Iocc = np.eye(self.system.n)

        occ = self.system.o
        virt = self.system.v

        nocc = occ.stop
        nvirt = virt.stop - nocc

        CI0 = self.C[0, I]
        CJ0 = self.C[0, J]
        CI_ai = self.C[1:, I].reshape(nvirt, nocc)
        CJ_ai = self.C[1:, J].reshape(nvirt, nocc)

        rho = np.zeros(self.system.h.shape, dtype=np.complex128)

        rho[occ, occ] = 2 * Iocc * (
            CI0 * CJ0.conj() + np.einsum("ia,ai->", CI_ai.T.conj(), CJ_ai)
        ) - np.einsum("ja, ai->ji", CI_ai.conj().T, CJ_ai)

        rho[virt, occ] = np.sqrt(2) * CI0.conj() * CJ_ai
        rho[occ, virt] = np.sqrt(2) * CJ0 * CI_ai.T.conj()
        rho[virt, virt] = np.einsum("ia,bi->ba", CI_ai.T.conj(), CJ_ai)

        return rho

    def compute_transition_dipole_moment(self, I, J):
        r = self.system.position
        rho = self.compute_one_body_density_matrix_IJ(I, J)
        return self.np.einsum("qp,ipq->i", rho, r)

    def compute_transition_moment(self, I, J, mat):
        rho = self.compute_one_body_density_matrix_IJ(I, J)
        if len(mat.shape) == 2:
            return self.np.einsum("qp,pq->", rho, mat)
        else:
            return self.np.einsum("qp,ipq->i", rho, mat)

    def compute_eigenstates(self):
        np = self.np

        occ = self.o
        virt = self.v

        e_ref = 2 * np.einsum("ii->", self.h[occ, occ])
        e_ref += 2 * np.einsum("ijij->", self.u[occ, occ, occ, occ])
        e_ref -= np.einsum("ijji->", self.u[occ, occ, occ, occ])

        self.hamiltonian = np.zeros((self.num_states, self.num_states))

        self.hamiltonian[0, 0] = e_ref
        self.hamiltonian[0, 1:] = np.sqrt(2) * self.f[virt, occ].ravel()
        self.hamiltonian[1:, 0] = np.sqrt(2) * self.f[occ, virt].ravel()

        Iocc = self.np.eye(self.n)
        Ivirt = self.np.eye(self.m)

        term1 = self.np.einsum("ab,ij->bjai", Ivirt, Iocc) * e_ref
        term2 = self.np.einsum("ij,ab->bjai", Iocc, self.f[virt, virt])
        term3 = self.np.einsum("ab,ij->bjai", Ivirt, self.f[occ, occ])
        term4 = 2 * self.np.einsum("bija->bjai", self.u[virt, occ, occ, virt])
        term4 -= self.np.einsum("biaj->bjai", self.u[virt, occ, virt, occ])
        term = term1 + term2 - term3 + term4
        term = term.reshape((self.m * self.n, self.m * self.n))
        self.hamiltonian[1:, 1:] = term

        self.eps, self.C = self.np.linalg.eigh(self.hamiltonian)

        return self
