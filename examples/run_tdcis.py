import numpy as np
from matplotlib import pyplot as plt
import tqdm
from scipy.integrate import complex_ode
import basis_set_exchange as bse
import time

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from ci_singles import CIS, TDCIS

from gauss_integrator import GaussIntegrator
from lasers import sine_square_laser

# Choose molecular geometry and basis set.
molecule = "h 0.0 0.0 -0.7; h 0.0 0.0 0.7"
# molecule = "li 0.0 0.0 0.0; h 0.0 0.0 3.05"
# molecule = "c 0.0 0.0 0.0; h 0.0 0.0 2.137130"
basis = "aug-cc-pvdz"
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

# Compute the (field-free) CIS eigenstates
cis = CIS(system).compute_eigenstates()
eps, C = cis.eps, cis.C
print(cis.eps[0:6] + system.nuclear_repulsion_energy)
print()
# Compute and print transition dipole moments from the groundstate to an excited state J.
print(f"** Transition dipole moments **")
for J in range(1, 6):
    dipole_transitions_0_to_J = (
        np.abs(cis.compute_transition_dipole_moment(0, J)) ** 2
    )
    print(
        f"|<0|x|{J}>|^2: {dipole_transitions_0_to_J[0]:.2e}, |<0|y|{J}>|^2: {dipole_transitions_0_to_J[1]:.2e}, |<0|z|{J}>|^2: {dipole_transitions_0_to_J[2]:.2e}"
    )
print()

# Set maximum field strength (E0), frequency (omega), duration of laser (td), total simulation time (tfinal)
# and polarization vector.
E0 = 0.03
omega = eps[1] - eps[0]
t_cycle = 2 * np.pi / omega
td = 3 * t_cycle
time_after_pulse = 1 * t_cycle
tfinal = np.floor(td + time_after_pulse)
pulse = sine_square_laser(E0=E0, omega=omega, td=td)

polarization = np.zeros(3)
polarization_direction = 2
polarization[polarization_direction] = 1
system.set_time_evolution_operator(
    DipoleFieldInteraction(pulse, polarization_vector=polarization)
)
print(f"E0={E0:.2f}, omega={omega:.2f}, 1 optical cycle={t_cycle:.2f}a.u.")

# Set integrator and initial state
tdcis = TDCIS(system)
integrator = complex_ode(tdcis).set_integrator(
    "GaussIntegrator", s=3, eps=1e-10
)
Ct = np.complex128(C[:, 0])
integrator.set_initial_value(Ct)


dt = 1e-2
num_steps = int(tfinal / dt) + 1
print(f"number of time steps={num_steps}")

time_points = np.zeros(num_steps)

rho_t = tdcis.compute_one_body_density_matrix(time_points[0], Ct)

dipole_moment = np.zeros(num_steps, dtype=np.complex128)
dipole_moment[0] = tdcis.compute_one_body_expectation_value(
    rho_t, system.dipole_moment[polarization_direction]
)

populations = np.zeros((6, num_steps))
for j in range(6):
    populations[j, 0] = np.abs(np.vdot(C[:, j], integrator.y)) ** 2


for i in tqdm.tqdm(range(num_steps - 1)):
    time_points[i + 1] = (i + 1) * dt
    Ct = integrator.integrate(integrator.t + dt)

    rho_t = tdcis.compute_one_body_density_matrix(time_points[i + 1], Ct)

    dipole_moment[i + 1] = tdcis.compute_one_body_expectation_value(
        rho_t,
        system.dipole_moment[polarization_direction],
    )

    for j in range(6):
        populations[j, i + 1] = np.abs(np.vdot(C[:, j], Ct)) ** 2

plt.figure()
plt.subplot(311)
plt.plot(time_points, pulse(time_points), color="red", label=r"$E(t)$")
plt.legend()
plt.subplot(312)
plt.plot(time_points, dipole_moment.real, label=r"$\langle \mu_z(t) \rangle$")
plt.axvline(3 * t_cycle, linestyle="dashed", color="red")
plt.legend()
plt.grid()
plt.subplot(313)
for j in range(6):
    plt.plot(
        time_points,
        populations[j],
        label=r"$|\langle \Psi_{%d}|\Psi(t) \rangle|^2$" % j,
    )
plt.legend()
plt.grid()
plt.show()
