import numpy as np
import matplotlib.pyplot as plt
import torch

from simulator import PitchDynamics
from controller import PDController
from train_model import ControlNet


def load_trained_model(model_path="control_model.pth"):
    checkpoint = torch.load(
    model_path,
    map_location=torch.device("cpu"),
    weights_only=False
)

    model = ControlNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_mean = checkpoint["X_mean"]
    X_std = checkpoint["X_std"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]

    return model, X_mean, X_std, y_mean, y_std


def ai_control(model, theta, q, X_mean, X_std, y_mean, y_std):
    x = np.array([[theta, q]], dtype=np.float32)
    x_norm = (x - X_mean) / X_std

    x_tensor = torch.tensor(x_norm, dtype=torch.float32)

    with torch.no_grad():
        y_norm = model(x_tensor).numpy()

    u = y_norm * y_std + y_mean
    return float(u[0, 0])


def run_simulation_with_pd(theta0_deg=10.0, steps=300):
    sim = PitchDynamics()
    controller = PDController(kp=2.0, kd=0.6, u_limit=2.0)

    state = sim.reset(theta0=np.deg2rad(theta0_deg), q0=0.0)

    theta_history = []
    q_history = []
    u_history = []

    for _ in range(steps):
        theta, q = state
        u = controller.compute_control(theta, q)

        theta_history.append(np.rad2deg(theta))
        q_history.append(np.rad2deg(q))
        u_history.append(u)

        disturbance = np.random.normal(0.0, 0.01)
        state = sim.step(u=u, disturbance=disturbance)

    return np.array(theta_history), np.array(q_history), np.array(u_history)


def run_simulation_with_ai(model, X_mean, X_std, y_mean, y_std, theta0_deg=10.0, steps=300):
    sim = PitchDynamics()

    state = sim.reset(theta0=np.deg2rad(theta0_deg), q0=0.0)

    theta_history = []
    q_history = []
    u_history = []

    for _ in range(steps):
        theta, q = state
        u = ai_control(model, theta, q, X_mean, X_std, y_mean, y_std)

        theta_history.append(np.rad2deg(theta))
        q_history.append(np.rad2deg(q))
        u_history.append(u)

        disturbance = np.random.normal(0.0, 0.01)
        state = sim.step(u=u, disturbance=disturbance)

    return np.array(theta_history), np.array(q_history), np.array(u_history)


def plot_results(theta_pd, theta_ai, u_pd, u_ai, dt=0.01):
    time = np.arange(len(theta_pd)) * dt

    plt.figure(figsize=(10, 5))
    plt.plot(time, theta_pd, label="PD Controller")
    plt.plot(time, theta_ai, label="AI Controller", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch Angle [deg]")
    plt.title("Pitch Stabilization Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time, u_pd, label="PD Controller")
    plt.plot(time, u_ai, label="AI Controller", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Control Input")
    plt.title("Control Effort Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    model, X_mean, X_std, y_mean, y_std = load_trained_model()

    theta_pd, q_pd, u_pd = run_simulation_with_pd(theta0_deg=10.0, steps=300)
    theta_ai, q_ai, u_ai = run_simulation_with_ai(
        model, X_mean, X_std, y_mean, y_std, theta0_deg=10.0, steps=300
    )

    print(f"Final PD pitch angle: {theta_pd[-1]:.4f} deg")
    print(f"Final AI pitch angle: {theta_ai[-1]:.4f} deg")

    plot_results(theta_pd, theta_ai, u_pd, u_ai)


if __name__ == "__main__":
    main()