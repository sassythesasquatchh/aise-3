import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import ipdb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot_data(data, title=None):
    if title in ["1", "2"]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x = data["x"]
        t = data["t"]
        u = data["u"]

        # X, T = np.meshgrid(x, t)
        ax.plot_surface(x, t, u, cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        # ax.scatter(x, t, u, c=u, cmap="viridis")
        if title is not None:
            filename = "task2_plots/" + title + "_data.png"
            plt.savefig(filename, dpi=300)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_title("u")
        ax2.set_title("v")
        u = data["u"][:, :, 0]
        v = data["v"][:, :, 0]
        cax1 = ax1.imshow(
            u,
            vmin=np.min(data["u"]),
            vmax=np.max(data["u"]),
            cmap="viridis",
            animated=True,
        )
        cax2 = ax2.imshow(
            v,
            vmin=np.min(data["v"]),
            vmax=np.max(data["v"]),
            cmap="viridis",
            animated=True,
        )
        fig.colorbar(cax1, ax=ax1)
        fig.colorbar(cax2, ax=ax2)

        def update(frame):
            global data1, data2
            data1 = data["u"][:, :, frame]
            data2 = data["v"][:, :, frame]

            cax1.set_array(data1)
            cax2.set_array(data2)
            return cax1, cax2

        ani = FuncAnimation(fig, update, frames=data["u"].shape[-1], blit=True)
        ani.save("task2_plots/" + title + "_data.gif", writer="imagemagick", fps=20)


def plot_domain(data, title=None):

    if title in ["1", "2"]:
        plt.figure(figsize=(8, 6))

        x = data["x"]
        t = data["t"]

        plt.scatter(x, t, c="orange", s=0.1)
        plt.grid()
        plt.tight_layout()

        if title is not None:
            filename = "task2_plots/" + title + "_domain.png"

            plt.savefig(filename)
    else:
        plt.figure(figsize=(4, 4))
        t = data["t"][0, 0, :]
        plt.plot(t)
        plt.grid()
        if title is not None:
            filename = "task2_plots/" + title + "_t.png"

            plt.savefig(filename)


def test_uniform_grid(data, dataset=None):

    def print_grid_stats(deltas):
        print("Max delta:", np.max(deltas))
        print("Min delta:", np.min(deltas))
        print("Mean delta:", np.mean(deltas))
        print("Std dev of delta:", np.std(deltas))

    if dataset in [1, 2]:
        x = data["x"]
        t = data["t"]
        x = x[:, 0]
        t = t[0, :]

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        for deltas, axis in zip([np.diff(x), np.diff(t)], ["x", "t"]):
            if np.allclose(deltas, deltas[0]):
                print(f"Grid is uniform along {axis} dimension")
            else:
                print(f"Grid is not uniform along {axis} dimension")
                print_grid_stats(deltas)
        # assert np.allclose(np.diff(x), dx), "x is not uniformly spaced"
        # assert np.allclose(np.diff(t), dt), "t is not uniformly spaced"
    else:
        x = data["x"][:, 0, 0]
        y = data["y"][0, :, 0]
        t = data["t"][0, 0, :]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dt = t[1] - t[0]
        for deltas, axis in zip([np.diff(x), np.diff(y), np.diff(t)], ["x", "y", "t"]):
            if np.allclose(deltas, deltas[0]):
                print(f"Grid is uniform along {axis} dimension")
            else:
                print(f"Grid is not uniform along {axis} dimension")
                print_grid_stats(deltas)
        # assert np.allclose(np.diff(x), dx), "x is not uniformly spaced"
        # assert np.allclose(np.diff(y), dy), "y is not uniformly spaced"
        # assert np.allclose(np.diff(t), dt), "t is not uniformly spaced"


def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3

    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """

    n = u.size
    ux = np.zeros(n, dtype=np.complex64)

    if d == 1:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)

        ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
        ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
        return ux

    if d == 2:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2

        ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx**2
        ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx**2
        return ux

    if d == 3:
        for i in range(2, n - 2):
            ux[i] = (u[i + 2] / 2 - u[i + 1] + u[i - 1] - u[i - 2] / 2) / dx**3

        ux[0] = (-2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]) / dx**3
        ux[1] = (-2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]) / dx**3
        ux[n - 1] = (
            2.5 * u[n - 1]
            - 9 * u[n - 2]
            + 12 * u[n - 3]
            - 7 * u[n - 4]
            + 1.5 * u[n - 5]
        ) / dx**3
        ux[n - 2] = (
            2.5 * u[n - 2]
            - 9 * u[n - 3]
            + 12 * u[n - 4]
            - 7 * u[n - 5]
            + 1.5 * u[n - 6]
        ) / dx**3
        return ux


def PolyDiffPoint(u, x, deg=3, diff=1, index=None):
    """
    Same as above but now just looking at a single point

    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """

    n = len(x)
    if index == None:
        index = (n - 1) // 2

    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x, u, deg)

    # Take derivatives
    derivatives = []
    for d in range(1, diff + 1):
        derivatives.append(poly.deriv(m=d)(x[index]))

    return derivatives


def construct_linear_system(
    data, p_order=3, d_order=3, dataset=None, deriv_method="FD"
):
    """
    Construct the theta matrix from the data
    """
    if dataset in [1, 2]:
        x = data["x"][:, 0]
        t = data["t"][0, :]
        u = data["u"]

        ut = np.zeros_like(u)
        dt = np.mean(t[1:] - t[:-1])
        # loop over spatial dimension, to calculate how solution varies in time at each point
        for i in range(u.shape[0]):
            ut[i, :] = FiniteDiff(u[i, :], dt, 1)

        ut = np.reshape(ut, (-1, 1))

        derivs = []
        dx = np.mean(x[1:] - x[:-1])
        # loop over time dimension, to calculate spatial derivatives at each time point
        for order in range(1, d_order + 1):
            d = np.zeros_like(u)
            for i in range(u.shape[1]):
                d[:, i] = FiniteDiff(u[:, i], dx, order)
            derivs.append(np.reshape(d, (-1, 1)))

        u = np.reshape(u, (-1, 1))
        polys = [u**i for i in range(p_order + 1)]
        derivs = [np.ones_like(u)] + derivs

        derivs_names = ["", "u_x", "u_xx", "u_xxx"]
        polys_names = ["", "u", "u^2", "u^3"]
        polys_names = polys_names[: p_order + 1]
        derivs_names = derivs_names[: d_order + 1]
        theta = []
        feature_names = []
        for deriv, d_name in zip(derivs, derivs_names):
            for poly, p_name in zip(polys, polys_names):
                theta.append(poly * deriv)
                if d_name and p_name:
                    feature_name = f"{p_name}*{d_name}"
                elif d_name:
                    feature_name = d_name
                elif p_name:
                    feature_name = p_name
                else:
                    feature_name = "1"
                feature_names.append(feature_name)

        theta = np.hstack(theta)

        return ut, theta, feature_names

    else:
        x = data["x"][:, 0, 0]
        y = data["y"][0, :, 0]
        t = data["t"][0, 0, :]
        u = data["u"]
        v = data["v"]

        assert v.shape == u.shape, "u and v must have same shape"

        if deriv_method == "FD":

            ut = np.zeros_like(u)
            vt = np.zeros_like(v)
            dt = np.mean(t[1:] - t[:-1])
            # loop over spatial dimensions, to calculate how solution varies in time at each point
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    ut[i, j, :] = FiniteDiff(u[i, j, :], dt, 1)
                    vt[i, j, :] = FiniteDiff(v[i, j, :], dt, 1)

            # flatten over both spatial dimensions
            ut = np.reshape(ut, (-1, 1))
            vt = np.reshape(vt, (-1, 1))

            pure_u_derivs = []
            pure_v_derivs = []
            dx = np.mean(x[1:] - x[:-1])
            dy = np.mean(y[1:] - y[:-1])

            for sol, derivs in zip([u, v], [pure_u_derivs, pure_v_derivs]):
                for order in range(1, d_order + 1):
                    # dx
                    d = np.zeros_like(sol)
                    print("Calculating dx order ", order)
                    # loop over time
                    for t in range(sol.shape[-1]):
                        # loop over y
                        for y in range(sol.shape[1]):
                            d[:, y, t] = FiniteDiff(sol[:, y, t], dx, order)
                    derivs.append(d.copy())

                    # dy
                    d = np.zeros_like(sol)
                    print("Calculating dy order ", order)

                    for t in range(sol.shape[-1]):
                        for x in range(sol.shape[0]):
                            d[x, :, t] = FiniteDiff(sol[x, :, t], dy, order)
                    derivs.append(d.copy())

            mixed_u_derivs = []
            mixed_v_derivs = []
            if d_order > 1:
                for mixed_derivs, pure_derivs in zip(
                    [mixed_u_derivs, mixed_v_derivs], [pure_u_derivs, pure_v_derivs]
                ):
                    # dy dx
                    print("Calculating dy dx")
                    sol_dx = pure_derivs[0]
                    d = np.zeros_like(sol_dx)
                    for t in range(sol.shape[-1]):
                        for x in range(sol.shape[0]):
                            d[x, :, t] = FiniteDiff(sol_dx[x, :, t], dy, 1)
                    mixed_derivs.append(np.reshape(d, (-1, 1)))

            if d_order > 2:
                for mixed_derivs, pure_derivs in zip(
                    [mixed_u_derivs, mixed_v_derivs], [pure_u_derivs, pure_v_derivs]
                ):
                    # dy dx^2
                    print("Calculating dy dx^2")
                    sol_dxx = pure_derivs[2]
                    d = np.zeros_like(sol_dxx)
                    for t in range(sol.shape[-1]):
                        for x in range(sol.shape[0]):
                            d[x, :, t] = FiniteDiff(sol_dx[x, :, t], dy, 1)
                    mixed_derivs.append(np.reshape(d, (-1, 1)))

                    # dx dy^2
                    print("Calculating dx dy^2")
                    sol_dyy = pure_derivs[3]
                    d = np.zeros_like(sol_dyy)
                    for t in range(sol.shape[-1]):
                        for y in range(sol.shape[1]):
                            d[:, y, t] = FiniteDiff(sol_dyy[:, y, t], dx, 1)
                    mixed_derivs.append(np.reshape(d, (-1, 1)))

            derivs = (
                [np.ones_like(u)]
                + pure_u_derivs
                + pure_v_derivs
                + mixed_u_derivs
                + mixed_v_derivs
            )

        elif deriv_method == "PolyDiff":

            np.random.seed(0)

            num_xy = 5000  # needs to be very high to work with noise
            num_t = 30
            num_points = num_xy * num_t
            boundary = 5
            points = {}
            count = 0
            n = len(x)
            U = u
            V = v

            for p in range(num_xy):
                x = np.random.choice(np.arange(boundary, n - boundary), 1)[0]
                y = np.random.choice(np.arange(boundary, n - boundary), 1)[0]
                for t in range(num_t):
                    points[count] = [x, y, 6 * t + 10]
                    count = count + 1

            # Take up to second order derivatives.
            u = np.zeros((num_points, 1))
            v = np.zeros((num_points, 1))
            ut = np.zeros((num_points, 1))
            vt = np.zeros((num_points, 1))
            ux = np.zeros((num_points, 1))
            uy = np.zeros((num_points, 1))
            uxx = np.zeros((num_points, 1))
            uxy = np.zeros((num_points, 1))
            uyy = np.zeros((num_points, 1))
            vx = np.zeros((num_points, 1))
            vy = np.zeros((num_points, 1))
            vxx = np.zeros((num_points, 1))
            vxy = np.zeros((num_points, 1))
            vyy = np.zeros((num_points, 1))

            N = 2 * boundary - 1  # number of points to use in fitting
            Nt = N
            deg = 4  # degree of polynomial to use

            for p in points.keys():

                [x, y, t] = points[p]

                # value of function
                u[p] = U[x, y, t]
                v[p] = V[x, y, t]

                # time derivatives
                ut[p] = PolyDiffPoint(
                    U[x, y, t - (Nt - 1) // 2 : t + (Nt + 1) // 2],
                    np.arange(Nt) * dt,
                    deg,
                    1,
                )[0]
                vt[p] = PolyDiffPoint(
                    V[x, y, t - (Nt - 1) // 2 : t + (Nt + 1) // 2],
                    np.arange(Nt) * dt,
                    deg,
                    1,
                )[0]

                # spatial derivatives
                ux_diff = PolyDiffPoint(
                    U[x - (N - 1) // 2 : x + (N + 1) // 2, y, t],
                    np.arange(N) * dx,
                    deg,
                    2,
                )
                uy_diff = PolyDiffPoint(
                    U[x, y - (N - 1) // 2 : y + (N + 1) // 2, t],
                    np.arange(N) * dy,
                    deg,
                    2,
                )
                vx_diff = PolyDiffPoint(
                    V[x - (N - 1) // 2 : x + (N + 1) // 2, y, t],
                    np.arange(N) * dx,
                    deg,
                    2,
                )
                vy_diff = PolyDiffPoint(
                    V[x, y - (N - 1) // 2 : y + (N + 1) // 2, t],
                    np.arange(N) * dy,
                    deg,
                    2,
                )
                ux_diff_yp = PolyDiffPoint(
                    U[x - (N - 1) // 2 : x + (N + 1) // 2, y + 1, t],
                    np.arange(N) * dx,
                    deg,
                    2,
                )
                ux_diff_ym = PolyDiffPoint(
                    U[x - (N - 1) // 2 : x + (N + 1) // 2, y - 1, t],
                    np.arange(N) * dx,
                    deg,
                    2,
                )
                vx_diff_yp = PolyDiffPoint(
                    V[x - (N - 1) // 2 : x + (N + 1) // 2, y + 1, t],
                    np.arange(N) * dx,
                    deg,
                    2,
                )
                vx_diff_ym = PolyDiffPoint(
                    V[x - (N - 1) // 2 : x + (N + 1) // 2, y - 1, t],
                    np.arange(N) * dx,
                    deg,
                    2,
                )

                ux[p] = ux_diff[0]  # first spatial derivative in x
                uy[p] = uy_diff[0]  # first spatial derivative in y
                uxx[p] = ux_diff[1]  # second spatial derivative in x
                uyy[p] = uy_diff[1]  # second spatial derivative in y
                # central difference in y of the first spatial derivative in x
                uxy[p] = (ux_diff_yp[0] - ux_diff_ym[0]) / (2 * dy)

                vx[p] = vx_diff[0]  # first spatial derivative in x
                vy[p] = vy_diff[0]  # first spatial derivative in y
                vxx[p] = vx_diff[1]  # second spatial derivative in x
                vyy[p] = vy_diff[1]  # second spatial derivative in y
                # central difference in y of the first spatial derivative in x
                vxy[p] = (vx_diff_yp[0] - vx_diff_ym[0]) / (2 * dy)

                derivs = [np.ones_like(u), ux, uy, uxx, uyy, vx, vy, vxx, vyy, uxy, vxy]

        u = np.reshape(u, (-1, 1))
        v = np.reshape(v, (-1, 1))
        u_polys = [u**i for i in range(p_order + 1)]
        v_polys = [v**i for i in range(p_order + 1)]

        derivs = [np.reshape(d, (-1, 1)) for d in derivs]
        polys = []

        for i, u_poly in enumerate(u_polys):
            for j, v_poly in enumerate(v_polys):
                if i + j > p_order:
                    continue
                polys.append(u_poly * v_poly)

        if d_order == 2:
            derivs_names = [
                "",
                "u_x",
                "u_y",
                "u_xx",
                "u_yy",
                "v_x",
                "v_y",
                "v_xx",
                "v_yy",
                "u_xy",
                "v_xy",
            ]
        if p_order == 3:
            polys_names = [
                "",
                "v",
                "v^2",
                "v^3",
                "u",
                "uv",
                "uv^2",
                "u^2",
                "u^2v",
                "u^3",
            ]

        theta = []
        feature_names = []
        for deriv, d_name in zip(derivs, derivs_names):
            for poly, p_name in zip(polys, polys_names):
                theta.append(poly * deriv)
                if d_name and p_name:
                    feature_name = f"{p_name}*{d_name}"
                elif d_name:
                    feature_name = d_name
                elif p_name:
                    feature_name = p_name
                else:
                    feature_name = "1"
                feature_names.append(feature_name)

        theta = np.hstack(theta)

        return ut, vt, theta, feature_names


def subsample_linear_system(theta, ut, vt, subsample_rate=0.02):
    """
    Subsample the linear system to make it more tractable
    """
    n = theta.shape[0]
    subsample_idx = np.random.choice(n, int(n * subsample_rate), replace=False)
    theta = theta[subsample_idx, :]
    ut = ut[subsample_idx, :]
    vt = vt[subsample_idx, :]
    return theta, ut, vt


def STRidge(X0, y, lam, maxit, tol, normalize=2, print_results=False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Get the standard ridge esitmate
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y), rcond=None)[0]
    else:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    num_relevant = d
    biginds = np.where(abs(w) > tol)[0]

    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                # if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else:
                break
        biginds = new_biginds

        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = np.linalg.lstsq(
                X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)),
                X[:, biginds].T.dot(y),
                rcond=None,
            )[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    # ipdb.set_trace()
    if isinstance(biginds, np.ndarray):
        biginds = biginds.tolist()
    if biginds != []:
        w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w


def TrainSTRidge(
    R,
    Ut,
    lam,
    d_tol,
    maxit=25,
    STR_iters=10,
    l0_penalty=None,
    normalize=2,
    split=0.8,
    print_best_tol=False,
):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them
    using a loss function on a holdout set.

    Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
    not squared 2-norm.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0)  # for consistancy
    n, _ = R.shape
    train = np.random.choice(n, int(n * split), replace=False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train, :]
    TestR = R[test, :]
    TrainY = Ut[train, :]
    TestY = Ut[test, :]
    D = TrainR.shape[1]

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None:
        l0_penalty = 0.001 * np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D, 1))
    w_best = np.linalg.lstsq(TrainR, TrainY, rcond=None)[0]
    err_best = np.linalg.norm(
        TestY - TestR.dot(w_best), 2
    ) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(TrainR, TrainY, lam, STR_iters, tol, normalize=normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty * np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0, tol - 2 * d_tol])
            d_tol = 2 * d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol:
        print("Optimal tolerance:", tol_best)

    return w_best


def print_pde(w, rhs_description, ut="u_t"):
    pde = ut + " = "
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + " + "
            pde = (
                pde
                + "(%05f %+05fi)" % (w[i].real, w[i].imag)
                + rhs_description[i]
                + "\n   "
            )
            first = False
    print(pde)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", type=int, required=True)
    args = parser.parse_args()
    assert args.dataset in [1, 2, 3], "Invalid dataset number"
    data = np.load(f"data/{args.dataset}.npz")
    # plot_domain(data, str(args.dataset))
    test_uniform_grid(data, args.dataset)

    if args.dataset in [1, 2]:
        if args.dataset == 1:
            ut, theta, feature_names = construct_linear_system(
                data, p_order=3, d_order=3, dataset=args.dataset
            )
        else:
            ut, theta, feature_names = construct_linear_system(
                data, p_order=2, d_order=3, dataset=args.dataset
            )
        try:
            w = TrainSTRidge(theta, ut, 1e-5, 5)
        except Exception as e:
            print(e)
            ipdb.post_mortem()
        print_pde(w, feature_names)
    else:
        try:
            ut, vt, theta, feature_names = construct_linear_system(
                data, p_order=3, d_order=2, dataset=args.dataset
            )
            theta, ut, vt = subsample_linear_system(theta, ut, vt, subsample_rate=0.05)

            print("Solving for u")
            w_u = TrainSTRidge(theta, ut, 1e-5, 1)
            print("Solving for v")
            w_v = TrainSTRidge(theta, vt, 1e-5, 1)
        except Exception as e:
            print(e)
            ipdb.post_mortem()
        print("u:")
        print_pde(w_u, feature_names)
        print("v:")
        print_pde(w_v, feature_names)
