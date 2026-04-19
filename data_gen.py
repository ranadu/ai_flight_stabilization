import numpy as np
import csv

from simulator import PitchDynamics
from controller import PDController


def generate_dataset(
    num_episodes=200,
    steps_per_episode=200,
    output_file="training_data.csv"
):
    sim = PitchDynamics()
    controller = PDController(kp=2.0, kd=0.6, u_limit=2.0)

    rows = []

    for episode in range(num_episodes):
        theta0_deg = np.random.uniform(-20.0, 20.0)
        q0_deg = np.random.uniform(-30.0, 30.0)

        theta0 = np.deg2rad(theta0_deg)
        q0 = np.deg2rad(q0_deg)

        state = sim.reset(theta0=theta0, q0=q0)

        for step in range(steps_per_episode):
            theta, q = state

            u = controller.compute_control(theta, q, theta_ref=0.0)

            rows.append([theta, q, u])

            disturbance = np.random.normal(loc=0.0, scale=0.01)
            state = sim.step(u=u, disturbance=disturbance)

    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta", "q", "u"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} samples to {output_file}")


if __name__ == "__main__":
    generate_dataset()