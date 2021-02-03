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
            self.E0 * C[0]
            + self.np.sqrt(2) * self.np.einsum("ia,ai->", self.f[o, v], Cai)
        )

        rhs_ai = self.E0 * Cai
        rhs_ai += self.np.einsum("ba,aj->bj", self.f[v, v], Cai)
        rhs_ai -= self.np.einsum("ij,bi->bj", self.f[o, o], Cai)
        rhs_ai += 2 * self.np.einsum("bija,ai->bj", self.u[v, o, o, v], Cai)
        rhs_ai -= self.np.einsum("biaj,ai->bj", self.u[v, o, v, o], Cai)
        rhs_ai += self.np.sqrt(2) * self.f[v, o] * C[0]

        C_new[1:] = -1j * rhs_ai.ravel()

        self.last_timestep = current_time

        return C_new
