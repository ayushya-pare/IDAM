import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

# ========================================
# 1. Core Mathematical Functions
# ========================================

def loss_function(theta):
    """
    Calculates the value of the 2D loss function for a given theta.
    """
    x, y = theta
    return 0.5 * x**2 + 2 * y**2 + 0.5 * x * y + 2 * np.sin(2 * x) * np.sin(2 * y)

def gradient(theta):
    """
    Calculates the gradient of the loss function at a given theta.
    """
    x, y = theta
    dx = x + 0.5 * y + 4 * np.cos(2 * x) * np.sin(2 * y)
    dy = 4 * y + 0.5 * x + 4 * np.sin(2 * x) * np.cos(2 * y)
    return np.array([dx, dy])

# ========================================
# 2. Optimization
# ========================================

def run_optimization(methods, lr=0.05, steps=100, beta=0.9, alpha=0.1, eps=1e-8):
    """
    Runs various optimization algorithms on the loss function.

    Returns:
        - A dictionary of performance metrics for each method.
        - A dictionary of trajectories (paths taken) for each method.
        - A dictionary of loss histories for each method.
    """
    metrics = {}
    trajectories = {}
    loss_histories = {}

    for method in methods:
        theta = np.array([2.5, 2.5])  # Starting point
        trajectory = [theta.copy()]
        m = np.zeros(2)
        v = np.zeros(2)
        G = np.zeros(2)
        theta_prev = theta.copy()
        grad_calls = 0

        start_time = time.time()

        for k in range(1, steps + 1):
            grad = gradient(theta)
            grad_calls += 1

            if method == "SGD":
                theta -= lr * grad
            elif method == "Momentum":
                m = beta * m + (1 - beta) * grad
                theta -= lr * m
            elif method == "AdaGrad":
                G += grad**2
                theta -= lr * grad / (np.sqrt(G) + eps)
            elif method == "Adam":
                m = beta * m + (1 - beta) * grad
                v = beta * v + (1 - beta) * grad**2
                m_hat = m / (1 - beta**k)
                v_hat = v / (1 - beta**k)
                theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
            elif method == "IDAM":
                displacement = theta - theta_prev
                eta_adaptive = alpha / (np.sqrt(1 + displacement**2))
                theta_prev = theta.copy()
                theta -= eta_adaptive * grad

            trajectory.append(theta.copy())

        runtime = time.time() - start_time
        trajectories[method] = np.array(trajectory)
        losses = [loss_function(t) for t in trajectory]
        loss_histories[method] = losses
        metrics[method] = {
            "runtime_sec": runtime,
            "gradient_evals": grad_calls,
            "final_loss": losses[-1]
        }

    return metrics, trajectories, loss_histories

# ========================================
# 3. Plotting
# ========================================

def plot_loss_history(loss_histories, output_dir="results"):
    """
    Plots the loss over iterations for each method and saves it to a file.
    """
    plt.figure(figsize=(12, 6))
    for method, losses in loss_histories.items():
        plt.plot(losses, label=method)
    
    plt.title("Loss Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "loss_over_iterations.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss history plot to {output_path}")

def plot_trajectories(trajectories, output_dir="results"):
    """
    Plots the optimizer trajectories on the loss surface and saves it to a file.
    """
    x_vals = np.linspace(-3, 3, 400)
    y_vals = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = loss_function([X, Y])

    plt.figure(figsize=(12, 10))
    contours = plt.contour(X, Y, Z, levels=60, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)

    for method, traj in trajectories.items():
        plt.plot(traj[:, 0], traj[:, 1], label=method)
        plt.scatter(traj[0, 0], traj[0, 1], color='black', marker='o')  # Start
        plt.scatter(traj[-1, 0], traj[-1, 1], color='red', marker='x')  # End

    plt.title("Optimizer Trajectories on Skewed Loss Surface")
    plt.xlabel("Theta 1 (x)")
    plt.ylabel("Theta 2 (y)")
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(output_dir, "optimizer_trajectories.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved trajectories plot to {output_path}")

# ========================================
# 4. Main Block
# ========================================

def main():
    """
    Main function to run the entire analysis.
    """
    # Define the methods to be tested
    methods = ["SGD", "Momentum", "AdaGrad", "Adam", "IDAM"]
    
    # Create the results directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Run the optimization
    metrics, trajectories, loss_histories = run_optimization(methods)

    # Display metrics as a formatted DataFrame
    results_df = pd.DataFrame(metrics).T
    print("\n=== Optimizer Metrics ===")
    print(results_df)

    # Save metrics to a CSV file in the results folder
    metrics_path = os.path.join(output_dir, "optimizer_metrics.csv")
    results_df.to_csv(metrics_path)
    print(f"\nSaved metrics to {metrics_path}")

    # Generate and save plots
    plot_loss_history(loss_histories, output_dir)
    plot_trajectories(trajectories, output_dir)

if __name__ == "__main__":
    main()
