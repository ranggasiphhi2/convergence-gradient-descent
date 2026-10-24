"""Manual derivatives for unregularized multinomial logistic regression.

Companion to gd_sgd_analysis.md. Parameter ordering is row-major:
theta = torch.cat((W.reshape(-1), b)).

Run offline derivative checks:
    python gd_sgd_verified.py
Also check a subset of the official Fashion-MNIST training set:
    python gd_sgd_verified.py --fashion-mnist --download

The default checks require PyTorch only. The optional dataset check also
requires a compatible torchvision installation. No training results are
claimed by these checks.
"""

import argparse
import json

import torch
import torch.nn.functional as F


def load_fashion_mnist(root="data", download=False):
    """Read raw uint8 tensors and normalize once; transforms are unnecessary."""
    from torchvision import datasets

    training_data = datasets.FashionMNIST(
        root=root, train=True, download=download
    )
    test_data = datasets.FashionMNIST(
        root=root, train=False, download=download
    )
    X = training_data.data.flatten(start_dim=1).to(torch.float32) / 255.0
    labels = training_data.targets.to(torch.long)
    Y = F.one_hot(labels, num_classes=10).to(X.dtype)
    X_test = test_data.data.flatten(start_dim=1).to(torch.float32) / 255.0
    test_labels = test_data.targets.to(torch.long)
    return X, Y, labels, X_test, test_labels


def initialize_parameters(d, K, *, dtype=torch.float32, device="cpu"):
    """Retain the draft's U[0, 1) initialization; seed the caller for repeats."""
    W = torch.rand(d, K, dtype=dtype, device=device)
    b = torch.rand(K, dtype=dtype, device=device)
    return W, b


def log_softmax_rows(Z):
    """Stable row-wise log probabilities without taking log(softmax(Z))."""
    shifted = Z - Z.amax(dim=1, keepdim=True)
    return shifted - torch.logsumexp(shifted, dim=1, keepdim=True)


def forward(X, Y, W, b):
    """Y must contain one-hot rows (or valid probability distributions)."""
    Z = X @ W + b
    log_P = log_softmax_rows(Z)
    P = log_P.exp()
    losses = -(Y * log_P).sum(dim=1)
    return P, losses


def objective(X, Y, W, b):
    return forward(X, Y, W, b)[1].mean()


def manual_gradient(X, Y, W, b):
    """Gradient of the MEAN loss; shapes are (d, K) and (K,)."""
    n = X.shape[0]
    P, _ = forward(X, Y, W, b)
    residual = P - Y
    grad_W = X.T @ residual / n
    grad_b = residual.mean(dim=0)
    return grad_W, grad_b


def logit_hessian_blocks(P):
    """Return C_i = Hessian of one sample loss wrt its K logits: (n, K, K).

    These blocks do not include the 1/n factor for the dataset mean loss.
    They are not the Hessian with respect to W and b.
    """
    return torch.diag_embed(P) - P.unsqueeze(2) * P.unsqueeze(1)


def pack_parameters(W, b):
    return torch.cat((W.reshape(-1), b.reshape(-1)))


def unpack_parameters(theta, d, K):
    return theta[: d * K].reshape(d, K), theta[d * K :].reshape(K)


def dense_parameter_hessian_tiny(X, W, b):
    """Exact Hessian of the mean loss, for SMALL validation problems only.

    Labels are unnecessary: the Hessian at fixed logits is independent of Y.
    The guard prevents accidentally constructing the Fashion-MNIST Hessian.
    """
    n, d = X.shape
    K = W.shape[1]
    parameter_count = (d + 1) * K
    if parameter_count > 1000:
        raise ValueError(
            "Dense Hessian is restricted to <= 1000 parameters; use the HVP."
        )
    A = torch.cat((X, X.new_ones(n, 1)), dim=1)
    P = log_softmax_rows(X @ W + b).exp()
    C = logit_hessian_blocks(P)
    # H4[r,j,s,l] = mean_i A[i,r] A[i,s] C[i,j,l].
    # Row-major reshape groups (r,j) as row and (s,l) as column.
    H4 = torch.einsum("ir,is,ijl->rjsl", A, A, C) / n
    return H4.reshape(parameter_count, parameter_count)


def hessian_vector_product(X, W, b, V_W, v_b):
    """Exact H @ v without a dense Hessian or per-sample K x K blocks.

    v uses the same row-major order as pack_parameters(V_W, v_b).
    The result is split into weight and bias parts.
    """
    n = X.shape[0]
    P = log_softmax_rows(X @ W + b).exp()
    R = X @ V_W + v_b
    Q = P * (R - (P * R).sum(dim=1, keepdim=True))
    return X.T @ Q / n, Q.mean(dim=0)


def verify_derivatives():
    """Compare manual formulas with independent autograd and finite differences."""
    reports = []
    for seed, (n, d, K) in enumerate([(7, 3, 4), (1, 2, 3), (9, 4, 2)]):
        torch.manual_seed(seed)
        dtype = torch.float64
        X = torch.randn(n, d, dtype=dtype)
        labels = torch.randint(K, (n,))
        Y = F.one_hot(labels, num_classes=K).to(dtype)
        W, b = initialize_parameters(d, K, dtype=dtype)
        # Include tied logits; a stable implementation must handle them.
        if seed == 1:
            W.zero_()
            b.zero_()

        theta = pack_parameters(W, b)

        def reference(t):
            w_t, b_t = unpack_parameters(t, d, K)
            return F.cross_entropy(X @ w_t + b_t, labels, reduction="mean")

        theta_autograd = theta.clone().requires_grad_(True)
        value_autograd = reference(theta_autograd)
        grad_autograd, = torch.autograd.grad(value_autograd, theta_autograd)
        grad_manual = pack_parameters(*manual_gradient(X, Y, W, b))
        torch.testing.assert_close(objective(X, Y, W, b), value_autograd)
        assert manual_gradient(X, Y, W, b)[0].shape == (d, K)
        torch.testing.assert_close(
            grad_manual, grad_autograd, rtol=1e-10, atol=1e-12
        )

        H_manual = dense_parameter_hessian_tiny(X, W, b)
        H_autograd = torch.autograd.functional.hessian(reference, theta)
        torch.testing.assert_close(H_manual, H_autograd, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(H_manual, H_manual.T, rtol=0, atol=1e-12)
        minimum_eigenvalue = torch.linalg.eigvalsh(H_manual).min().item()
        assert minimum_eigenvalue >= -1e-12

        v = torch.randn_like(theta)
        v = v / torch.linalg.vector_norm(v)
        V_W, v_b = unpack_parameters(v, d, K)
        hv_manual = pack_parameters(*hessian_vector_product(X, W, b, V_W, v_b))
        torch.testing.assert_close(hv_manual, H_autograd @ v, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(hv_manual, H_manual @ v, rtol=1e-10, atol=1e-12)

        epsilon = 1e-5
        W_plus, b_plus = unpack_parameters(theta + epsilon * v, d, K)
        W_minus, b_minus = unpack_parameters(theta - epsilon * v, d, K)
        grad_plus = pack_parameters(*manual_gradient(X, Y, W_plus, b_plus))
        grad_minus = pack_parameters(*manual_gradient(X, Y, W_minus, b_minus))
        hv_finite_difference = (grad_plus - grad_minus) / (2 * epsilon)
        torch.testing.assert_close(
            hv_manual, hv_finite_difference, rtol=1e-6, atol=1e-9
        )
        directional_finite_difference = (
            objective(X, Y, W_plus, b_plus)
            - objective(X, Y, W_minus, b_minus)
        ) / (2 * epsilon)
        torch.testing.assert_close(
            grad_manual @ v, directional_finite_difference, rtol=1e-6, atol=1e-9
        )

        # A common class bias shift is an exact null direction.
        shift = pack_parameters(torch.zeros_like(W), torch.ones_like(b))
        torch.testing.assert_close(H_manual @ shift, torch.zeros_like(shift),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(objective(X, Y, W, b + 17),
                                   objective(X, Y, W, b), rtol=1e-12, atol=1e-12)

        P, _ = forward(X, Y, W, b)
        C = logit_hessian_blocks(P)
        Z = X @ W + b
        H_Z_autograd = torch.autograd.functional.hessian(
            lambda z: F.cross_entropy(z.reshape(n, K), labels), Z.reshape(-1)
        )
        H_Z_manual = torch.block_diag(*C.unbind(dim=0)) / n
        torch.testing.assert_close(H_Z_manual, H_Z_autograd, rtol=1e-10, atol=1e-12)
        reports.append({
            "shape_n_d_K": [n, d, K],
            "gradient_max_abs_error": (grad_manual - grad_autograd).abs().max().item(),
            "hessian_max_abs_error": (H_manual - H_autograd).abs().max().item(),
            "hvp_finite_difference_max_abs_error":
                (hv_manual - hv_finite_difference).abs().max().item(),
            "minimum_hessian_eigenvalue": minimum_eigenvalue,
        })

    # A naive log(softmax) calculation can produce inf or NaN here.
    extreme_Z = torch.tensor([[1000.0, -1000.0, 0.0],
                              [-1000.0, 1000.0, 0.0]], dtype=torch.float64)
    extreme_labels = torch.tensor([1, 0])
    extreme_Y = F.one_hot(extreme_labels, num_classes=3).to(torch.float64)
    stable_loss = -(extreme_Y * log_softmax_rows(extreme_Z)).sum(dim=1)
    assert torch.isfinite(stable_loss).all()
    torch.testing.assert_close(stable_loss,
                               F.cross_entropy(extreme_Z, extreme_labels, reduction="none"))
    return {"torch_version": torch.__version__, "status": "passed",
            "cases": reports, "extreme_logits_losses": stable_loss.tolist()}


def verify_fashion_mnist(root="data", download=False):
    """Check actual 784-feature data; use an HVP, never a dense Hessian."""
    torch.manual_seed(42)
    X, Y, labels, X_test, test_labels = load_fashion_mnist(root, download)
    X_small = X[:128].to(torch.float64)
    Y_small = Y[:128].to(torch.float64)
    labels_small = labels[:128]
    d, K = X.shape[1], Y.shape[1]
    W, b = initialize_parameters(d, K, dtype=torch.float64)
    theta = pack_parameters(W, b)

    def reference(t):
        w_t, b_t = unpack_parameters(t, d, K)
        return F.cross_entropy(X_small @ w_t + b_t, labels_small)

    theta_auto = theta.clone().requires_grad_(True)
    gradient_auto, = torch.autograd.grad(reference(theta_auto), theta_auto)
    gradient_manual = pack_parameters(*manual_gradient(X_small, Y_small, W, b))
    torch.testing.assert_close(gradient_manual, gradient_auto, rtol=1e-9, atol=1e-11)
    v = torch.randn_like(theta)
    v /= torch.linalg.vector_norm(v)
    V_W, v_b = unpack_parameters(v, d, K)
    hv_manual = pack_parameters(*hessian_vector_product(X_small, W, b, V_W, v_b))
    _, hv_auto = torch.autograd.functional.hvp(reference, theta, v)
    torch.testing.assert_close(hv_manual, hv_auto, rtol=1e-9, atol=1e-11)
    return {
        "status": "passed", "training_shape": list(X.shape),
        "test_shape": list(X_test.shape), "checked_training_samples": len(X_small),
        "parameter_count": theta.numel(),
        "gradient_max_abs_error": (gradient_manual - gradient_auto).abs().max().item(),
        "hvp_max_abs_error": (hv_manual - hv_auto).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fashion-mnist", action="store_true")
    parser.add_argument("--download", action="store_true",
                        help="Allow torchvision to download Fashion-MNIST.")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    report = {"synthetic_checks": verify_derivatives()}
    if args.fashion_mnist:
        report["fashion_mnist_checks"] = verify_fashion_mnist(
            args.data_root, args.download
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
