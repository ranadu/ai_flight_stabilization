import numpy as np


class PDController:
    def __init__(self, kp=2.0, kd=0.6, u_limit=2.0):
        """
        Simple PD controller for pitch stabilization.

        Parameters
        ----------
        kp : float
            Proportional gain on pitch angle
        kd : float
            Derivative gain on pitch rate
        u_limit : float
            Maximum absolute control output
        """
        self.kp = kp
        self.kd = kd
        self.u_limit = u_limit

    def compute_control(self, theta, q, theta_ref=0.0):
        """
        Compute control action.

        Parameters
        ----------
        theta : float
            Current pitch angle [rad]
        q : float
            Current pitch rate [rad/s]
        theta_ref : float
            Desired pitch angle [rad]

        Returns
        -------
        u : float
            Control input
        """
        error = theta_ref - theta
        u = self.kp * error - self.kd * q
        u = np.clip(u, -self.u_limit, self.u_limit)
        return u


if __name__ == "__main__":
    from simulator import PitchDynamics

    sim = PitchDynamics()
    controller = PDController(kp=2.0, kd=0.6, u_limit=2.0)

    state = sim.reset(theta0=np.deg2rad(10), q0=0.0)

    for i in range(200):
        theta, q = state
        u = controller.compute_control(theta, q, theta_ref=0.0)
        state = sim.step(u=u)

        theta_deg = np.rad2deg(state[0])
        q_deg = np.rad2deg(state[1])

        print(
            f"Step {i:03d} | "
            f"theta = {theta_deg:7.3f} deg | "
            f"q = {q_deg:7.3f} deg/s | "
            f"u = {u:7.3f}"
        )