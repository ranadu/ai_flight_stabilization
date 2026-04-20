import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

from simulator import PitchDynamics
from controller import PDController
from train_model import ControlNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def run_simulation_with_pd(theta0_deg=10.0, q0_deg=0.0, steps=300, disturbance_std=0.01):
    sim = PitchDynamics()
    controller = PDController(kp=2.0, kd=0.6, u_limit=2.0)

    state = sim.reset(
        theta0=np.deg2rad(theta0_deg),
        q0=np.deg2rad(q0_deg)
    )

    theta_history = []
    q_history = []
    u_history = []

    for _ in range(steps):
        theta, q = state
        u = controller.compute_control(theta, q)

        theta_history.append(np.rad2deg(theta))
        q_history.append(np.rad2deg(q))
        u_history.append(u)

        disturbance = np.random.normal(0.0, disturbance_std)
        state = sim.step(u=u, disturbance=disturbance)

    return np.array(theta_history), np.array(q_history), np.array(u_history)


def run_simulation_with_ai(
    model,
    X_mean,
    X_std,
    y_mean,
    y_std,
    theta0_deg=10.0,
    q0_deg=0.0,
    steps=300,
    disturbance_std=0.01
):
    sim = PitchDynamics()

    state = sim.reset(
        theta0=np.deg2rad(theta0_deg),
        q0=np.deg2rad(q0_deg)
    )

    theta_history = []
    q_history = []
    u_history = []

    for _ in range(steps):
        theta, q = state
        u = ai_control(model, theta, q, X_mean, X_std, y_mean, y_std)

        theta_history.append(np.rad2deg(theta))
        q_history.append(np.rad2deg(q))
        u_history.append(u)

        disturbance = np.random.normal(0.0, disturbance_std)
        state = sim.step(u=u, disturbance=disturbance)

    return np.array(theta_history), np.array(q_history), np.array(u_history)


def compute_metrics(theta_pd, theta_ai, u_pd, u_ai):
    final_pd_pitch_error = theta_pd[-1]
    final_ai_pitch_error = theta_ai[-1]

    peak_abs_pitch_pd = np.max(np.abs(theta_pd))
    peak_abs_pitch_ai = np.max(np.abs(theta_ai))

    rms_pitch_difference = np.sqrt(np.mean((theta_ai - theta_pd) ** 2))
    mean_abs_control_difference = np.mean(np.abs(u_ai - u_pd))

    metrics = {
        "final_pd_pitch_error_deg": final_pd_pitch_error,
        "final_ai_pitch_error_deg": final_ai_pitch_error,
        "peak_abs_pitch_pd_deg": peak_abs_pitch_pd,
        "peak_abs_pitch_ai_deg": peak_abs_pitch_ai,
        "rms_pitch_difference_deg": rms_pitch_difference,
        "mean_abs_control_difference": mean_abs_control_difference,
    }

    return metrics


def print_metrics(metrics):
    print("Performance Metrics:")
    print(f"  Final PD pitch error:        {metrics['final_pd_pitch_error_deg']:.4f} deg")
    print(f"  Final AI pitch error:        {metrics['final_ai_pitch_error_deg']:.4f} deg")
    print(f"  Peak |pitch| PD:             {metrics['peak_abs_pitch_pd_deg']:.4f} deg")
    print(f"  Peak |pitch| AI:             {metrics['peak_abs_pitch_ai_deg']:.4f} deg")
    print(f"  RMS pitch difference:        {metrics['rms_pitch_difference_deg']:.4f} deg")
    print(f"  Mean |control difference|:   {metrics['mean_abs_control_difference']:.6f}")


def plot_case_results(case_name, theta_pd, theta_ai, u_pd, u_ai, dt=0.01, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    time = np.arange(len(theta_pd)) * dt

    plt.figure(figsize=(10, 5))
    plt.plot(time, theta_pd, label="PD Controller")
    plt.plot(time, theta_ai, label="AI Controller", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch Angle [deg]")
    plt.title(f"Pitch Stabilization Comparison - {case_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{case_name}_pitch.png"), dpi=200)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time, u_pd, label="PD Controller")
    plt.plot(time, u_ai, label="AI Controller", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Control Input")
    plt.title(f"Control Effort Comparison - {case_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{case_name}_control.png"), dpi=200)
    plt.show()


def evaluate_case(model, X_mean, X_std, y_mean, y_std, case_name, theta0_deg, q0_deg):
    print(f"\n--- {case_name} ---")
    print(f"Initial condition: theta0 = {theta0_deg} deg, q0 = {q0_deg} deg/s")

    theta_pd, q_pd, u_pd = run_simulation_with_pd(
        theta0_deg=theta0_deg,
        q0_deg=q0_deg,
        steps=300
    )

    theta_ai, q_ai, u_ai = run_simulation_with_ai(
        model,
        X_mean,
        X_std,
        y_mean,
        y_std,
        theta0_deg=theta0_deg,
        q0_deg=q0_deg,
        steps=300
    )

    metrics = compute_metrics(theta_pd, theta_ai, u_pd, u_ai)
    print_metrics(metrics)

    plot_case_results(case_name, theta_pd, theta_ai, u_pd, u_ai)


def main():
    set_seed(42)

    model, X_mean, X_std, y_mean, y_std = load_trained_model()

    test_cases = [
        {"case_name": "case_1_positive_angle", "theta0_deg": 10.0, "q0_deg": 0.0},
        {"case_name": "case_2_negative_angle", "theta0_deg": -15.0, "q0_deg": 0.0},
        {"case_name": "case_3_pitch_rate_disturbance", "theta0_deg": 5.0, "q0_deg": 20.0},
    ]

    for case in test_cases:
        evaluate_case(
            model,
            X_mean,
            X_std,
            y_mean,
            y_std,
            case_name=case["case_name"],
            theta0_deg=case["theta0_deg"],
            q0_deg=case["q0_deg"]
        )


if __name__ == "__main__":
    main()