class TDCIS:
    def __init__(self, system):

        self.np = system.np
        self.system = system

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)
        self.o = self.system.o
        self.v = self.system.v

        self.E0 = system.compute_reference_energy()

        self.last_timestep = None

    def compute_one_body_density_matrix(self, current_time, y):
        np = self.np

        Iocc = np.eye(self.system.n)

        occ = self.system.o
        virt = self.system.v

        nocc = occ.stop
        nvirt = virt.stop - nocc

        Ct_ai = y[1:].reshape(nvirt, nocc)

        rho = np.zeros(self.system.h.shape, dtype=np.complex128)

        rho[occ, occ] = 2 * Iocc * (
            y[0] * y[0].conj() + np.einsum("ia,ai->", Ct_ai.T.conj(), Ct_ai)
        ) - np.einsum("ja, ai->ji", Ct_ai.conj().T, Ct_ai)

        rho[virt, occ] = np.sqrt(2) * y[0].conj() * Ct_ai
        rho[occ, virt] = np.sqrt(2) * y[0] * Ct_ai.T.conj()
        rho[virt, virt] = np.einsum("ia,bi->ba", Ct_ai.T.conj(), Ct_ai)

        return rho

    def compute_energy(self, current_time, h, u, y):
        """
        h is the matrix representation of a generic one-body operator
        u is the tensor representation of a generic two-body operator
        """

        o, v = self.system.o, self.system.v
        nocc = o.stop
        nvirt = v.stop - nocc

        f = self.system.construct_fock_matrix(h, u)
        Ct_0 = y[0]
        Ct_ai = y[1:].reshape(nvirt, nocc)

        energy = (
            self.np.sqrt(2)
            * Ct_0.conj()
            * self.np.einsum("ia, ai->", f[o, v], Ct_ai)
        )
        energy += (
            self.np.sqrt(2)
            * Ct_0
            * self.np.einsum("ai, ai->", f[v, o], Ct_ai.conj())
        )
        energy += self.np.einsum("ab, ai, bi->", f[v, v], Ct_ai.conj(), Ct_ai)
        energy -= self.np.einsum(
            "ai,bj,jaib->", Ct_ai.conj(), Ct_ai, u[o, v, o, v]
        )
        energy += 2 * self.np.einsum(
            "ai,bj,jabi->", Ct_ai.conj(), Ct_ai, u[o, v, v, o]
        )
        energy -= self.np.einsum("ij, aj, ai->", f[o, o], Ct_ai.conj(), Ct_ai)
        return energy

    def compute_one_body_expectation_value(self, rho, mat):
        return self.np.einsum("qp,pq->", rho, mat)

    def update_hamiltonian(self, current_time, y):

        if self.last_timestep == current_time:
            return

        self.last_timestep = current_time

        if self.system.has_one_body_time_evolution_operator:
            self.h = self.system.h_t(current_time)

        if self.system.has_two_body_time_evolution_operator:
            self.u = self.system.u_t(current_time)

        self.f = self.system.construct_fock_matrix(self.h, self.u)

        self.E0 = 2 * self.np.einsum("ii->", self.h[self.o, self.o])
        self.E0 += 2 * self.np.einsum(
            "ijij->", self.u[self.o, self.o, self.o, self.o]
        )
        self.E0 -= self.np.einsum(
            "ijji->", self.u[self.o, self.o, self.o, self.o]
        )

    def __call__(self, current_time, C):

        o, v = self.system.o, self.system.v

        self.update_hamiltonian(current_time, C)

        nocc = o.stop
        nvirt = v.stop - nocc

        Cai = C[1:].reshape(nvirt, nocc)

        C_new = self.np.zeros(C.shape, dtype=C[0].dtype)

        C_new[0] = -1j * (
            self.np.sqrt(2) * self.np.einsum("ia,ai->", self.f[o, v], Cai)
        )

        rhs_ai = self.np.einsum("ba,aj->bj", self.f[v, v], Cai)
        rhs_ai -= self.np.einsum("ij,bi->bj", self.f[o, o], Cai)
        rhs_ai += 2 * self.np.einsum("bija,ai->bj", self.u[v, o, o, v], Cai)
        rhs_ai -= self.np.einsum("biaj,ai->bj", self.u[v, o, v, o], Cai)
        rhs_ai += self.np.sqrt(2) * self.f[v, o] * C[0]

        C_new[1:] = -1j * rhs_ai.ravel()

        self.last_timestep = current_time

        return C_new
