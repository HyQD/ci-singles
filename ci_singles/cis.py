class CIS:
    def __init__(self, system):

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

    def compute_ground_state(self):

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
