import numpy as np
from matplotlib import pyplot as plt
import tqdm
import basis_set_exchange as bse

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from ci_singles import CIS, TDCIS


from lasers import sine_square_laser, AdiabaticLaser

def tdcis_quasi_energy(
    molecule,
    basis,
    E0,
    omega,
    phase,
    n_cycles_ramp=3,
    n_cycles_post_ramp=1,
    pol_dir=2,
    time_step=0.1,
):

    charge = 0
    basis_set = bse.get_basis(basis, fmt="nwchem")

    # Do Hartree-Fock and change integrals to MO-basis.
    system = construct_pyscf_system_rhf(
        molecule=molecule,
        basis=basis_set,
        add_spin=False,
        charge=charge,
        cart=False,
        anti_symmetrize=False,
    )

    cis = CIS(system)
    H0 = cis.setup_hamiltonian(system.h, system.u)

    V = cis.setup_hamiltonian(
        system.dipole_moment[pol_dir],
        np.zeros((system.l, system.l, system.l, system.l)),
    )

    t_cycle = 2 * np.pi / omega
    tfinal = t_cycle * n_cycles_post_ramp + n_cycles_ramp * t_cycle

    electric_field = AdiabaticLaser(
        F_str=E0,
        omega=omega,
        phase=phase,
        n_switch=n_cycles_ramp,
        switch="quadratic",
    )

    I = np.complex128(np.eye(H0.shape[0]))
    Ct = np.complex128(np.zeros(H0.shape[0]))
    Ct[0] = 1.0 + 0j
    C0 = Ct.copy()

    num_steps = int(tfinal / time_step) + 1
    print(f"number of time steps={num_steps}")

    time_points = np.zeros(num_steps)
    dipole_moment = np.zeros(num_steps, dtype=np.complex128)
    dipole_moment[0] = np.dot(Ct.conj(), V @ Ct)

    Q_t = np.zeros(num_steps)
    phi0_psit = np.vdot(C0, Ct)
    H_psit = np.dot(H0, Ct)
    phi0_H_psit = np.vdot(C0, H_psit)
    Q_t[0] = (phi0_psit.conj() * phi0_H_psit).real

    for i in tqdm.tqdm(range(num_steps - 1)):
        time_points[i + 1] = (i + 1) * time_step
        H_t_mid = H0 - V * electric_field(time_points[i] + time_step / 2)
        A_p = I + 1j * time_step / 2 * H_t_mid
        A_m = I - 1j * time_step / 2 * H_t_mid
        Ct = np.linalg.solve(A_p, A_m @ Ct)
        dipole_moment[i + 1] = np.dot(Ct.conj(), V @ Ct)

        phi0_psit = np.vdot(C0, Ct)

        H_t = H0 - V * electric_field(time_points[i + 1])
        H_psit = np.dot(H_t, Ct)
        phi0_H_psit = np.vdot(C0, H_psit)
        Q_t[i + 1] = (phi0_H_psit / phi0_psit).real

    samples = dict()
    samples["time_points"] = time_points
    samples["electric_field"] = electric_field(time_points)
    samples["dipole_moment"] = dipole_moment
    samples["Q_t"] = Q_t

    return samples

if __name__ == "__main__":
    
    molecule = "h 0.0 0.0 -0.7; h 0.0 0.0 0.7"
    basis = "aug-cc-pvdz"

    E0 = 0.001
    omega = 0.2
    phase = 0.0
    n_cycles_ramp = 3
    n_cycles_post_ramp = 3


    samples = tdcis_quasi_energy(
        molecule,
        basis,
        E0,
        omega,
        phase,
        n_cycles_ramp=n_cycles_ramp,
        n_cycles_post_ramp=n_cycles_post_ramp,
        pol_dir=2,
        time_step=0.1,
    )

    time_points = samples["time_points"]
    electric_field = samples["electric_field"]
    dipole_moment = samples["dipole_moment"]
    Q_t = samples["Q_t"]

    t_cycle = 2*np.pi/omega

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].plot(time_points, electric_field, color="green", label=r"$E(t)$")
    ax[0].axvline(t_cycle*n_cycles_ramp, color="red", linestyle="--")
    
    ax[1].plot(time_points, dipole_moment.real, label="dipole moment", color="blue")
    ax[1].axvline(t_cycle*n_cycles_ramp, color="red", linestyle="--")

    ax[2].plot(time_points, Q_t, label="Q(t)", color="black")
    ax[2].axvline(t_cycle*n_cycles_ramp, color="red", linestyle="--")
    plt.show()