import numpy as np

import tqdm
import basis_set_exchange as bse

from quantum_systems import construct_pyscf_system_rhf
from ci_singles import CIS

from scipy.sparse.linalg import expm_multiply, expm


def tdcis_absorbtion_spectrum(
    molecule,
    basis,
    E0=0.001,
    dt=2e-2,
    t_final=2000,
):

    charge = 0
    # Get basis set.
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

    # Compute Hamiltonian and dipole moment operators
    cis = CIS(system)
    H0 = cis.setup_hamiltonian(system.h, system.u)
    V = [
        cis.setup_hamiltonian(
            system.dipole_moment[alpha],
            np.zeros((system.l, system.l, system.l, system.l)),
        )
        for alpha in range(3)
    ]

    print(f"CIS Hilbert space is {H0.shape[0]}-dimensional.")

    # Get the ground state, which in CIS is just the HF state.
    C_ground_state = np.zeros(H0.shape[0], dtype=complex)
    C_ground_state[0] = 1.0

    # Set up time grid and laser pulse
    n_steps = int(t_final / dt)
    time_points = np.linspace(0, t_final, n_steps + 1)
    dt = time_points[1] - time_points[0]

    dipole_moment = np.zeros((n_steps + 1, 3), dtype=np.float64)

    # Set up exact propagator for fast post-pulse propagation
    U0 = expm(-1j * dt * H0)

    # Loop over polarization directions
    for alpha in range(3):

        # Start in ground state
        Ct = np.complex128(C_ground_state)

        # Compute dipole moment along polarization direction for t=0,
        # just before pulse
        dipole_moment[0, alpha] = (Ct.conj().T @ (V[alpha] @ Ct)).real

        # Set up kicked wavefunction
        # by propagating through the pulse
        Ct = expm_multiply(-1j * E0 * V[alpha], Ct)

        # Do time integration after pulse
        for i in tqdm.tqdm(range(n_steps)):
            # Exact solution of TDSE
            Ct = U0 @ Ct

            # Compute dipole moment along polarization direction for t=time_points[i+1]
            dipole_moment[i + 1, alpha] = (Ct.conj().T @ (V[alpha] @ Ct)).real

    samples = dict()
    samples["time_points"] = time_points
    samples["dipole_moment"] = dipole_moment
    samples["n_steps"] = n_steps
    samples["dt"] = dt

    return samples


if __name__ == "__main__":

    from scipy.signal import find_peaks
    from matplotlib import pyplot as plt
    from numpy.fft import fft, fftfreq, fftshift

    name = "h2"
    molecule = "h 0.0 0.0 0.0; h 0.0 0.0 1.4"
    basis = "aug-cc-pvdz"

    samples = tdcis_absorbtion_spectrum(
        molecule,
        basis,
        E0=0.001,
        dt=2e-2,
        t_final=2000,
    )

    print(samples.keys())

    time_points = samples["time_points"]
    dipole_moment = samples["dipole_moment"]
    dt = samples["dt"]
    n_steps = samples["n_steps"]

    # Compute FFT of dipole moment
    eV = 27.2114
    # Get FFT frequency grid
    om = eV * fftshift(fftfreq(n_steps, dt) * 2 * np.pi)

    # Select a plotting window
    min_freq = 0  # shuld be >= 0
    max_freq = np.inf  # np.inf = infinity

    # damping factor to broaden peaks
    damping = np.exp(-0.002 * time_points[1:])
    # induced dipole moments along each polarization direction
    dip_ind0 = dipole_moment[1:, 0] - dipole_moment[0, 0]
    dip_ind1 = dipole_moment[1:, 1] - dipole_moment[0, 1]
    dip_ind2 = dipole_moment[1:, 2] - dipole_moment[0, 2]

    # total signal
    signal = damping * (dip_ind0 + dip_ind1 + dip_ind2)

    # select range of frequencies 0 < freq < max_freq
    filter = np.where((om >= min_freq) & (om <= max_freq))

    # Compute S(omega)
    signal_fft = fftshift(fft(signal, norm="ortho"))
    S = np.abs(om * signal_fft.imag)

    # Estimate peaks in spectrum
    peaks, _ = find_peaks(S[filter], height=0.01, prominence=0.01)
    print("Estimated spectral points: ", om[filter][peaks])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(om[filter], S[filter])
    plt.plot(om[filter][peaks], S[filter][peaks], "x")
    plt.xlabel("Excitation energy (eV)")
    plt.ylabel("Intensity")
    plt.title("Absorbtion spectrum")
    plt.grid("on")
    plt.xlim(0, 100)
    plt.savefig(name + "_spectrum.pdf")
    plt.show()
