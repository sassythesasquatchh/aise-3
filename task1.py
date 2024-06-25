import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from jax import jit
import jax
import matplotlib.pyplot as plt
import ipdb
import equinox as eqx
from equinox import nn
import optax
from functools import partial
from argparse import ArgumentParser
import numpy as np
from matplotlib.animation import FuncAnimation


# Constants
M = 1.0
m = 0.1
g = 9.81
l = 1.0


# @jit
def rk4(y, t, h, forcing):
    """
    y is the augmented state vector consisting of (x, x_dot, theta, theta_dot)
    """
    # ipdb.set_trace()
    # jax.debug.print(t)
    k1 = dynamics(y, t, forcing)
    k2 = dynamics(y + 0.5 * h * k1, t + 0.5 * h, forcing)
    k3 = dynamics(y + 0.5 * h * k2, t + 0.5 * h, forcing)
    k4 = dynamics(y + h * k3, t + h, forcing)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# @jit
def dynamics(y, t, forcing):
    """
    y is the augmented state vector consisting of (x, x_dot, theta, theta_dot)
    """
    x_dot = y[1]
    x_dot_dot = (
        forcing(t)
        - m * g * jnp.cos(y[2]) * jnp.sin(y[2])
        + (m * l * y[3] ** 2) * jnp.sin(y[2])
    ) / (M + m * (1 - jnp.cos(y[2]) ** 2))
    theta_dot = y[3]
    theta_dot_dot = (g * jnp.sin(y[2]) - jnp.cos(y[2]) * x_dot_dot) / l
    return jnp.array([x_dot, x_dot_dot, theta_dot, theta_dot_dot])


# @jit
def ODESolver(y0, t0, tf, h, forcing):
    """
    Solve the ODE using the Runge-Kutta 4th order method
    """
    t_values = jnp.arange(t0, tf, h)  # Array of time steps

    def step(y, t):
        new_y = rk4(y, t, h, forcing)  # Compute the next state using RK4
        return new_y, new_y  # Return the new state and store it

    y, y_intermediate = lax.scan(step, y0, t_values)  # Perform the scan operation
    return t_values, y_intermediate  # Return all intermediate values


def plot(
    t,
    y,
    title: str,
    task_num: int = 1,
    forcing=None,
):

    if task_num == 1:
        vels = False
        if vels:
            fig, axs = plt.subplots(2, 2)
            fig.suptitle("Pendulum motion")
            axs[0, 0].plot(t, y[:, 0])
            axs[0, 0].set_title("x")
            axs[0, 1].plot(t, y[:, 1])
            axs[0, 1].set_title("x_dot")
            axs[1, 0].plot(t, y[:, 2])
            axs[1, 0].set_title("theta")
            axs[1, 1].plot(t, y[:, 3])
            axs[1, 1].set_title("theta_dot")
            title += "_vels"
        else:
            fig, axs = plt.subplots(1, 2)
            fig.suptitle("Pendulum motion")
            axs[0].plot(t, y[:, 0])
            axs[0].set_title("x")
            axs[1].plot(t, y[:, 2])
            axs[1].set_title("theta")
    else:
        assert forcing is not None, "Please pass a forcing function to visualise"
        three_quarters = t[-1] * 0.75
        fig, axs = plt.subplots(1, 3)
        fig.suptitle("Pendulum motion")
        axs[0].plot(t, y[:, 0])
        axs[0].set_title("x")
        axs[1].plot(t, y[:, 2])
        axs[1].axvline(x=three_quarters, color="r", linestyle="--")
        axs[1].set_title("theta")
        force = jax.vmap(forcing)(jnp.expand_dims(t, 1))
        axs[2].plot(t, force)
        axs[2].axvline(x=three_quarters, color="r", linestyle="--")
        axs[2].set_title("Learnt F(t)")

    plt.tight_layout()
    plt.savefig(title + ".png", dpi=300)


def animate(y, title):
    x_center = y[:, 0]
    theta = y[:, 2]

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")  # Equal aspect ratio
    ax.set_xlim(np.min(x_center) - 1, np.max(x_center) + 1)
    ax.set_ylim(-4, 4)  # Adjust as needed

    cart_width = 1.0
    cart_height = 0.5
    cart_bottom = 0.0
    pendulum_length = 1.0
    rect = plt.Rectangle((0, 0), cart_width, cart_height, color="blue")
    ax.add_patch(rect)
    (pendulum_line,) = ax.plot([], [], color="black")
    (pendulum_circle,) = ax.plot([], [], "o", color="red")

    def update(frame):
        x = x_center[frame]
        angle = theta[frame]

        # Update cart position
        rect.set_xy((x - cart_width / 2, cart_bottom))

        # Update pendulum position
        pendulum_top_x = x
        pendulum_top_y = cart_bottom + cart_height
        pendulum_end_x = pendulum_top_x + pendulum_length * np.sin(angle)
        pendulum_end_y = pendulum_top_y + pendulum_length * np.cos(angle)
        pendulum_line.set_data(
            [pendulum_top_x, pendulum_end_x], [pendulum_top_y, pendulum_end_y]
        )
        pendulum_circle.set_data([pendulum_end_x], [pendulum_end_y])

        return rect, pendulum_line, pendulum_circle

    ani = FuncAnimation(fig, update, frames=range(len(y)), blit=True)
    ani.save(title + ".mp4", fps=len(y) / 5)


class FCN(eqx.Module):
    """
    Fully-connected neural network
    """

    layers: list

    def __init__(self, input_dim, hidden_dim, output_dim, key):
        key1, key2, key3 = jr.split(key, 3)

        self.layers = [
            nn.Linear(input_dim, hidden_dim, key=key1),
            jax.nn.relu,
            nn.Linear(hidden_dim, hidden_dim, key=key2),
            jax.nn.relu,
            nn.Linear(hidden_dim, hidden_dim, key=key2),
            jax.nn.relu,
            nn.Linear(hidden_dim, output_dim, key=key3),
        ]

    def __call__(self, x):
        # ipdb.set_trace()
        if jnp.ndim(x) == 0:
            x = jnp.expand_dims(x, 0)
        for layer in self.layers:
            x = layer(x)
        x = jnp.squeeze(x)
        return x


class BalancePendulum:

    def __init__(self, input_dim, hidden_dim, output_dim, key, y0, t0, tf, h):

        self.optim = optax.adamw(1e-3)
        self.forcing_model = FCN(input_dim, hidden_dim, output_dim, key)
        self.ode_solver = partial(ODESolver, y0, t0, tf, h)
        self.tf = tf

    def _loss(self, forcing_model):
        """
        L2 norm on the pendulum trajectory for the last quarter of the simulation.
        # Penalises both the position and the velocity of the pendulum.
        Penalises the position of the pendulum.
        """
        _, y = self.ode_solver(forcing_model)
        quarter = len(y) // 4
        quarter += (
            len(y) // 16
        )  # Begin penalising a little before the period we actually care about
        # ipdb.set_trace()
        penalised_indices = [2, 3]
        loss = jnp.mean(y[-quarter:, penalised_indices] ** 2)
        # loss = jnp.mean(y[-quarter:, :] ** 2)
        return loss

    def train(self, n_steps=2000):
        opt_state = self.optim.init(eqx.filter(self.forcing_model, eqx.is_array))

        @eqx.filter_jit
        def step(opt_state, forcing_model):
            # ipdb.set_trace()
            loss_value, grads = eqx.filter_value_and_grad(self._loss)(forcing_model)
            updates, opt_state = self.optim.update(grads, opt_state, forcing_model)
            forcing_model = eqx.apply_updates(forcing_model, updates)
            return forcing_model, opt_state, loss_value

        for i in range(n_steps):
            self.forcing_model, opt_state, loss_value = step(
                opt_state, self.forcing_model
            )
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{n_steps}] loss: {loss_value}")


def task1():
    # Initial conditions
    y0 = jnp.array([0.0, 0.0, jnp.pi / 4, 0.0])
    t0 = 0.0
    tf = 5.0
    n_steps = 500
    h = (tf - t0) / n_steps

    def forcing(t):
        return 10 * jnp.sin(t)

    # Solve the ODE
    t, y = ODESolver(y0, t0, tf, h, forcing)
    plot(t, y, "1_1")
    animate(y, "1_1")


def task2():
    # Initial conditions
    y0 = jnp.array([0.0, 0.0, jnp.pi / 4, 0.0])
    t0 = 0.0
    tf = 5.0
    n_ode_steps = 500
    n_training_steps = 5000
    h = (tf - t0) / n_ode_steps
    solver = BalancePendulum(1, 64, 1, jr.key(0), y0, t0, tf, h)
    solver.train(n_training_steps)
    # solver.train(500)

    # Solve the ODE
    t, y = ODESolver(y0, t0, tf, h, solver.forcing_model)
    plot(t, y, "1_2", task_num=2, forcing=solver.forcing_model)
    animate(y, "1_2")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", "-t", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.task == 1:
            task1()
        else:
            task2()
    except Exception as e:
        print(e)
        ipdb.post_mortem()
