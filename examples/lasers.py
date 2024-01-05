import numpy as np


class linear_laser:
    def __init__(self, E0, omega, n_ramp):
        self.E0 = E0
        self.omega = omega
        self.T0 = 2 * np.pi / omega
        self.n_ramp = n_ramp

    def __call__(self, t):
        T0 = self.n_ramp * self.T0
        if t <= T0:
            ft = t / T0
        else:
            ft = 1
        return ft * np.sin(self.omega * t) * self.E0


class sine_laser:
    def __init__(self, E0, omega, td, phase=0.0, start=0.0):
        self.E0 = E0
        self.omega = omega
        self.tprime = td
        self.phase = phase
        self.t0 = start

    def __call__(self, t):
        dt = t - self.t0
        return (
            self.E0
            * np.sin(self.omega * t + self.phase)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )


class sine_square_laser:
    def __init__(self, E0, omega, td, phase=0.0, start=0.0):
        self.F_str = E0
        self.omega = omega
        self.tprime = td
        self.phase = phase
        self.t0 = start

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.F_str
        )
        return pulse


class sine_square_laser_velocity:
    def __init__(self, E0, omega, td, phase=0.0, start=0.0):
        self.F_str = E0
        self.omega = omega
        self.tprime = td
        self.phase = phase
        self.t0 = start
        self.A = np.pi / self.tprime

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            -self.F_str
            / (8 * self.A**2 * self.omega - 2 * self.omega**3)
            * (
                np.cos(self.omega * t)
                * (
                    -4 * self.A**2
                    - self.omega**2 * np.cos(2 * self.A * t)
                    + self.omega**2
                )
                + 2
                * self.A
                * (
                    2 * self.A
                    - self.omega
                    * np.sin(2 * self.A * t)
                    * np.sin(self.omega * t)
                )
            )
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )
        return pulse


class square_length_dipole:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.sin(np.pi * dt / self.tprime)
            * (
                self.omega
                * np.sin(np.pi * dt / self.tprime)
                * np.sin(self.omega * dt + self.phase)
                - (2 * np.pi / self.tprime)
                * np.cos(np.pi * dt / self.tprime)
                * np.cos(self.omega * dt + self.phase)
            )
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * self.A0
        )
        return pulse


class square_velocity_dipole:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class FermiSwitch:

    def __init__(self, t0, t1, tau):
        self.t0 = t0
        self.t1 = t1
        self.tau = tau

    def name(self):
        return 'Fermi'

    def __call__(self, t):
        if type(t) == np.ndarray:
            # t is assumed in ascending order and uniformly spaced
            t_mid = 0.5*(self.t1 - self.t0)
            dt = t[1] - t[0]
            indx_0 = int(self.t0/dt)
            indx_1 = int(self.t1/dt)
            res = np.empty(t.shape[0])
            res[:indx_0] = 0.
            res[indx_0:indx_1] = 1 - 1/(1 + np.exp((t[indx_0:indx_1] - t_mid)/self.tau))
            res[indx_1:] = 1.
        else:
            if t < self.t0:
                res = 0.
            elif t <= self.t1:
                res = (t - self.t0)/(self.t1 - self.t0)
            else:
                res = 1.
        return res


class LinearSwitch:

    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1

    def name(self):
        return 'linear'

    def __call__(self, t):
        if type(t) == np.ndarray:
            # t is assumed in ascending order and uniformly spaced
            dt = t[1] - t[0]
            indx_0 = int(self.t0/dt)
            indx_1 = int(self.t1/dt)
            res = np.empty(t.shape[0])
            res[:indx_0] = 0.
            res[indx_0:indx_1] = (t[indx_0:indx_1] - self.t0)/(self.t1 - self.t0)
            res[indx_1:] = 1.
        else:
            if t < self.t0:
                res = 0.
            elif t <= self.t1:
                res = (t - self.t0)/(self.t1 - self.t0)
            else:
                res = 1.
        return res


class QuadraticSwitch:

    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1
        self.t_mid = 0.5*(t1 - t0)
        self.alpha = 2/(t1-3*t0)**2

    def name(self):
        return 'quadratic'

    def __call__(self, t):
        if type(t) == np.ndarray:
            # t is assumed in ascending order and uniformly spaced
            dt = t[1] - t[0]
            indx_0 = int(self.t0/dt)
            indx_m = int(self.t_mid/dt)
            indx_1 = int(self.t1/dt)
            res = np.empty(t.shape[0])
            res[:indx_0] = 0.
            res[indx_0:indx_m] = self.alpha*(t[indx_0:indx_m] - self.t0)**2
            res[indx_m:indx_1] = -self.alpha*(t[indx_m:indx_1] - self.t1)**2 + 1
            res[indx_1:] = 1.
        else:
            if t < self.t0:
                res = 0.
            elif t <= self.t_mid:
                res = self.alpha*(t - self.t0)**2
            elif t <= self.t1:
                res = -self.alpha*(t - self.t1)**2 + 1
            else:
                res = 1.
        return res
                

class AdiabaticLaser:

    def __init__(self, F_str, omega, phase=0.0, n_switch=1, switch='quadratic', delta=None):
        self.F_str = F_str
        self.omega = omega
        self.n_switch = n_switch
        self.t_cycle = 2*np.pi/omega
        self.phase = phase
        self._envelope = self._select_envelope(switch.lower(), delta)

    def _continuous_wave(self, t):
        return self.F_str*np.cos(self.omega*t+self.phase)

    def _select_envelope(self, switch, delta):
        t0 = 0.
        t1 = self.n_switch*self.t_cycle
        if switch == 'fermi':
            if delta is None:
                eps = 1e-4*self.F_str
            else:
                eps = delta
            tau = -0.5*t1/np.log(eps/(1-eps))
            return FermiSwitch(t0, t1, tau)
        elif switch == 'linear':
            return LinearSwitch(t0, t1)
        elif switch == 'quadratic':
            return QuadraticSwitch(t0, t1)
        else:
            raise ValueError(f'Illegal switch: {switch}')

    def __call__(self, t):
        return self._continuous_wave(t)*self._envelope(t)