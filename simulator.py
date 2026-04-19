import numpy as np


class PitchDynamics:
    def __init__(self, Iyy=0.05, damping=0.1, stiffness=0.8, dt=0.01):
        """
        Simple 1-axis pitch dynamics model.

        Parameters
        ----------
        Iyy : float
            Pitch moment of inertia [kg*m^2]
        damping : float
            Damping coefficient
        stiffness : float
            Restoring stiffness coefficient
        dt : float
            Simulation time step [s]
        """
        self.Iyy = Iyy
        self.damping = damping
        self.stiffness = stiffness
        self.dt = dt

        self.theta = 0.0  # pitch angle [rad]
        self.q = 0.0      # pitch rate [rad/s]

    def reset(self, theta0=0.0, q0=0.0):
        """
        Reset the system state.

        Parameters
        ----------
        theta0 : float
            Initial pitch angle [rad]
        q0 : float
            Initial pitch rate [rad/s]
        """
        self.theta = theta0
        self.q = q0
        return np.array([self.theta, self.q], dtype=float)

    def step(self, u, disturbance=0.0):
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        u : float
            Control input torque
        disturbance : float
            External disturbance torque

        Returns
        -------
        state : np.ndarray
            Updated state [theta, q]
        """
        q_dot = (u + disturbance - self.damping * self.q - self.stiffness * self.theta) / self.Iyy
        theta_dot = self.q

        self.q += q_dot * self.dt
        self.theta += theta_dot * self.dt

        return np.array([self.theta, self.q], dtype=float)

    def get_state(self):
        """
        Return the current state.
        """
        return np.array([self.theta, self.q], dtype=float)


if __name__ == "__main__":
    sim = PitchDynamics()

    state = sim.reset(theta0=np.deg2rad(10), q0=0.0)

    for i in range(200):
        state = sim.step(u=0.0, disturbance=0.0)
        theta_deg = np.rad2deg(state[0])
        q_deg = np.rad2deg(state[1])
        print(f"Step {i:03d} | theta = {theta_deg:7.3f} deg | q = {q_deg:7.3f} deg/s")