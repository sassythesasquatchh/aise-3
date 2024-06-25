import numpy as np
from task3_models import *
import copy
import ipdb
import scipy
from sklearn.preprocessing import StandardScaler


def random_function_sampler(function_class, n_points, n_functions):

    def put_in_range(fs):
        for i in range(len(fs)):
            scaler = StandardScaler()
            # ipdb.set_trace()
            fs[i] = scaler.fit_transform(fs[i].reshape(-1, 1)).reshape(-1)
        # scaler = StandardScaler()
        # fs = scaler.fit_transform(fs)
        return fs

    # Gaussian process
    if function_class == "gp":
        X = np.expand_dims(np.linspace(-1, 1, n_points), 1)
        sigma = rbf_kernel(X, X, 0.1)
        fs = np.random.multivariate_normal(np.zeros(n_points), sigma, n_functions)
        return put_in_range(fs.astype(np.float32))

    # Piecewise-linear
    elif function_class == "pw_linear":
        num_segments = 10

        fs = np.zeros((n_functions, n_points))
        for f in range(n_functions):
            breakpoints = np.sort(np.random.uniform(-1, 1, num_segments - 1))
            breakpoints = np.concatenate(([-1], breakpoints, [1]))
            ys = np.random.uniform(-3.0, 3.0, len(breakpoints))
            slopes = np.zeros(num_segments)
            for i in range(num_segments):
                slopes[i] = (ys[i + 1] - ys[i]) / (breakpoints[i + 1] - breakpoints[i])
            offsets = ys[:-1]
            for i, x in enumerate(np.linspace(-1, 1, n_points)):
                # ipdb.set_trace()
                final_seg = True
                for seg in range(num_segments):
                    if breakpoints[seg] <= x < breakpoints[seg + 1]:
                        fs[f, i] = slopes[seg] * (x - breakpoints[seg]) + offsets[seg]
                        final_seg = False
                if final_seg:
                    fs[f, i] = slopes[-1] * (x - breakpoints[-2]) + offsets[-1]
        return fs.astype(np.float32)

    # Chebyshev polynomials
    elif function_class == "chebyshev":
        n_coeffs = 20
        # X = np.expand_dims(np.linspace(-1, 1, n_points), 1)
        X = np.linspace(-1, 1, n_points)
        fs = np.zeros((n_functions, n_points))
        for i in range(n_functions):
            coeffs = 0.5 * np.random.randn(n_coeffs)
            fs[i, :] = np.polynomial.chebyshev.chebval(X, coeffs)
        return put_in_range(fs.astype(np.float32))

    else:
        raise ValueError("Unknown function class")


def rbf_kernel(x1, x2, sigma):
    return np.exp(-0.5 * scipy.spatial.distance.cdist(x1, x2, "sqeuclidean") / sigma**2)


def solve_1d_poisson(functions):
    N = len(functions[0])  # number of points including boundaries
    h = 2 / (N - 1)  # length of domain divided by number of segments
    N_interior = N - 2  # number of interior points
    A = (
        np.diag(2 * np.ones(N_interior))
        - np.diag(np.ones(N_interior - 1), 1)
        - np.diag(np.ones(N_interior - 1), -1)
    )
    A = A / h**2

    solutions = []
    for f in functions:
        # solve solution in the interior
        solution = np.linalg.solve(A, f[1:-1])
        # append the solution with the boundary conditions
        solutions.append(np.concatenate(([0], solution, [0])))
    return np.array(solutions, dtype=np.float32)


def build_dataset(function_class, n_functions=200):
    n_points = 500
    functions = random_function_sampler(function_class, n_points, n_functions)
    solutions = solve_1d_poisson(functions)
    return np.expand_dims(functions, 2), np.expand_dims(solutions, 2)


class PoissonDataset(Dataset):
    def __init__(self, functions, solutions):
        self.n_functions = len(functions)
        self.functions = functions
        self.solutions = solutions

    def __len__(self):
        return self.n_functions

    def __getitem__(self, idx):
        return torch.tensor(self.functions[idx]), torch.tensor(self.solutions[idx])


def train(model, train_data, n_epochs, val_data=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_loss = 0
    best_val_loss = np.inf
    best_train_loss = np.inf
    best_epoch = 0
    device = next(model.parameters()).device
    best_model = copy.deepcopy(model)
    for epoch in range(n_epochs):
        epoch_train_loss = 0
        model.train()
        for functions, solutions in train_data:
            functions = functions.to(device)
            solutions = solutions.to(device)

            optimizer.zero_grad()

            predicted_solutions = model(functions)
            loss = torch.mean((predicted_solutions - solutions) ** 2)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_data.dataset)

        if val_data is not None:
            with torch.no_grad():
                val_loss = 0
                model.eval()
                for functions, solutions in val_data:
                    functions = functions.to(device)
                    solutions = solutions.to(device)
                    predicted_solutions = model(functions)
                    loss = torch.mean((predicted_solutions - solutions) ** 2)
                    val_loss += loss.item()
                val_loss /= len(val_data.dataset)
        print(
            f"Epoch {epoch + 1}, Train Loss {epoch_train_loss}, Validation Loss {val_loss}"
        )
        if val_data is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model)
        else:
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model)
    best_loss = best_val_loss if val_data is not None else best_train_loss
    print(f"Best loss {best_loss} at epoch {best_epoch}")
    return best_model


def get_samples(model, function_class, device, n_samples=5):
    functions, solutions = build_dataset(function_class, 5)
    dataset = PoissonDataset(functions, solutions)
    data = DataLoader(dataset, batch_size=1)
    model.eval()
    outputs = []
    for function, solution in data:
        function = function.to(device)
        solution = solution.to(device)
        outputs.append(model(function).squeeze(0).cpu().detach().numpy())
    return functions, solutions, outputs


def save_results(results):
    with open("task3_plots/results.txt", "w") as f:
        for train_key in results.keys():
            f.write(f"Trained on {train_key}\n")
            for test_key in results[train_key].keys() & [
                "gp",
                "pw_linear",
                "chebyshev",
            ]:
                f.write(f"  Tested on {test_key}\n")
                f.write(
                    f"    Zero-shot loss: {results[train_key][test_key]['zero_shot_loss']}\n"
                )
                f.write(
                    f"    Finetuned loss: {results[train_key][test_key]['finetuned_loss']}\n"
                )
                f.write("\n")


def plot_dataset(functions, solutions, class_name):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    x = np.linspace(-1, 1, len(functions[0]))
    for i in range(5):
        axs[0].plot(x, functions[i], label=f"Function {i}")
        axs[1].plot(x, solutions[i], label=f"Solution {i}")
    axs[0].set_title("Functions")
    axs[1].set_title("Solutions")
    plt.tight_layout()
    # ipdb.set_trace()
    plt.savefig(f"task3_plots/{class_name}_dataset.png")


def plot_outputs(functions, solutions, outputs, class_name):
    num_subplots = len(functions)
    fig, axs = plt.subplots(num_subplots, 1, figsize=(2, 2 * num_subplots))
    x = np.linspace(-1, 1, len(functions[0]))
    for i in range(5):
        axs[i].plot(x, solutions[i], label="Ground truth")
        axs[i].plot(x, outputs[i], label="Prediction")
        axs[i].legend()
        # axs[1].plot(x, solutions[i], label=f"Solution {i}")
        # axs[2].plot(x, outputs[i].detach().numpy(), label=f"Output {i}")

    plt.tight_layout()

    plt.savefig(f"task3_plots/{class_name}_outputs.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    function_classes = ["gp", "pw_linear", "chebyshev"]
    cache = {function_class: dict() for function_class in function_classes}
    for function_class in function_classes:
        functions, solutions = build_dataset(function_class, 500)
        # ipdb.set_trace()
        dataset = PoissonDataset(functions, solutions)
        train_data = DataLoader(dataset, batch_size=32, shuffle=True)

        functions, solutions = build_dataset(function_class, 50)
        plot_dataset(functions, solutions, function_class)
        dataset = PoissonDataset(functions, solutions)
        val_data = DataLoader(dataset, batch_size=32, shuffle=True)
        # train_data, val_data = torch.utils.data.random_split(dataset, [160, 40])

        model = FNO1d(modes=16, width=64).to(device)
        model = train(model, train_data, val_data=val_data, n_epochs=100)
        functions, solutions, outputs = get_samples(
            model, function_class=function_class, device=device
        )
        plot_outputs(functions, solutions, outputs, function_class)

        functions, solutions = build_dataset(function_class, 100)
        # plot_dataset(functions, solutions, function_class)
        # dataset = PoissonDataset(functions, solutions)
        # eval_data = DataLoader(dataset, batch_size=32, shuffle=True)
        cache[function_class]["trained"] = model
        cache[function_class]["eval_data"] = (functions, solutions)
        # cache[function_class]["eval_data"] = eval_data
        # cache[function_class]["finetune_data"] = DataLoader(
        #     PoissonDataset(functions[:20, :], solutions[:20, :]),
        #     batch_size=16,
        #     shuffle=True,
        # )

    for function_class_trained in function_classes:
        for function_class_testing in function_classes:
            if function_class_trained == function_class_testing:
                continue
            # functions, solutions = build_dataset(function_class_testing, 100)
            # eval_dataset = PoissonDataset(functions, solutions)
            # eval_data = DataLoader(eval_dataset, batch_size=16, shuffle=False)
            model = copy.deepcopy(cache[function_class]["trained"])
            model.eval()
            zero_shot_loss = 0
            functions, solutions = copy.deepcopy(
                cache[function_class_testing]["eval_data"]
            )
            n_eval_functions = len(functions)
            n_val_functions = n_eval_functions // 8
            eval_data = DataLoader(
                PoissonDataset(
                    functions[:-n_val_functions], solutions[:-n_val_functions]
                ),
                batch_size=16,
                shuffle=True,
            )
            for functions, solutions in eval_data:
                functions = functions.to(device)
                solutions = solutions.to(device)
                predicted_solutions = model(functions)
                loss = torch.mean((predicted_solutions - solutions) ** 2)
                zero_shot_loss += loss.item()
            zero_shot_loss /= len(eval_data)
            cache[function_class_trained][function_class_testing] = {
                "zero_shot_loss": zero_shot_loss
            }

            # finetune_dataset = PoissonDataset(functions[:20, :], solutions[:20, :])
            # finetune_data = DataLoader(finetune_dataset, batch_size=16, shuffle=True)
            # finetune_data = cache[function_class]["finetune_data"]
            n_finetune = 20
            finetune_data = DataLoader(
                PoissonDataset(functions[:n_finetune], solutions[:n_finetune]),
                batch_size=16,
                shuffle=True,
            )
            val_data = DataLoader(
                PoissonDataset(
                    functions[-n_val_functions:], solutions[-n_val_functions:]
                ),
                batch_size=5,
                shuffle=True,
            )

            finetuned_model = train(
                model, train_data=finetune_data, val_data=val_data, n_epochs=1000
            )
            finetuned_model.eval()
            finetuned_loss = 0
            for functions, solutions in eval_data:
                functions = functions.to(device)
                solutions = solutions.to(device)
                predicted_solutions = model(functions)
                loss = torch.mean((predicted_solutions - solutions) ** 2)
                finetuned_loss += loss.item()

            finetuned_loss /= len(eval_data)
            cache[function_class_trained][function_class_testing][
                "finetuned_loss"
            ] = finetuned_loss

    save_results(cache)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        ipdb.post_mortem()
