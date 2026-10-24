# Fashion-MNIST: Objective, Gradients, and Hessian of a Linear Softmax Classifier

This note studies **multinomial logistic regression**, implemented as one affine layer followed by a softmax function, with no hidden layers and no regularization. The goal is to establish a mathematically consistent baseline for subsequent GD and SGD analysis.

The derivations distinguish derivatives with respect to **logits** from derivatives with respect to the **trainable parameters**. The accompanying file `gd_sgd_verified.py` implements the formulas and checks them against PyTorch autograd and finite differences.

**Current scope:** the model, objective, first- and second-order derivatives, convexity, and derivative verification. This version does not report completed GD/SGD convergence experiments or training accuracy.

## 1. Notation and dimensions

Vectors are column vectors in the mathematics. Samples are stored as rows in the data and logit matrices. Lowercase letters denote scalars or vectors, and uppercase letters denote matrices; indices make the distinction explicit.

| Symbol | Meaning | Dimension or range |
| --- | --- | --- |
| $n$ | Number of training samples | $60{,}000$ |
| $d$ | Number of features per image | $28\times28=784$ |
| $K$ | Number of classes | $10$ |
| $i$ | Sample index | $1,\ldots,n$ |
| $r,s$ | Feature indices | $1,\ldots,d$ |
| $j,\ell$ | Class indices | $1,\ldots,K$ |
| $T$ | Raw training images | $\{0,\ldots,255\}^{n\times28\times28}$ |
| $x_i$ | Flattened and normalized image | $[0,1]^d$ |
| $X$ | Matrix with row $i$ equal to $x_i^\top$ | $[0,1]^{n\times d}$ |
| $c_i$ | Class label in mathematical notation | $\{1,\ldots,K\}$ |
| $y_i$ | One-hot target, with $y_{i,j}=\mathbf{1}_{\{j=c_i\}}$ | $\{0,1\}^K$, $\mathbf{1}_K^\top y_i=1$ |
| $Y$ | Matrix with row $i$ equal to $y_i^\top$ | $\{0,1\}^{n\times K}$ |
| $W$ | Trainable weight matrix | $\mathbb{R}^{d\times K}$ |
| $b$ | Trainable bias vector | $\mathbb{R}^K$ |
| $z_i$ | Logit vector for sample $i$ | $\mathbb{R}^K$ |
| $Z$ | Matrix with row $i$ equal to $z_i^\top$ | $\mathbb{R}^{n\times K}$ |
| $p_i$ | Predicted class probabilities for sample $i$ | $(0,1)^K$, $\mathbf{1}_K^\top p_i=1$ |
| $P$ | Matrix with row $i$ equal to $p_i^\top$ | $(0,1)^{n\times K}$ |
| $\mathbf{1}_q$ | Vector of ones | $\mathbb{R}^q$ |

The mathematical class labels are one-based; the Python labels are zero-based, from `0` to `K - 1`. All norms of vectors below are Euclidean. Matrix parameter gradients use the Frobenius inner product, $\langle A,B\rangle_F=\operatorname{tr}(A^\top B)$.

## 2. Load and preprocess Fashion-MNIST

Fashion-MNIST supplies separate training and test sets, with $60{,}000$ and $10{,}000$ images, respectively. The objective below uses only the training set. The test set is reserved for evaluation.

For each image, flatten its $28\times28$ pixels and divide the raw integer intensities by $255$:

$$
x_i=\frac{\operatorname{flatten}(T_i)}{255}\in[0,1]^d.
$$

The code reads `dataset.data` directly. Dataset transforms such as `ToTensor()` are applied when retrieving samples through `dataset[i]`, not when directly accessing `.data`; therefore, the conversion and normalization are explicit here.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
```

```python
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
```

```python
# Set download=True for the first run if the dataset is not already present.
X, Y, labels, X_test, test_labels = load_fashion_mnist(
    root="data", download=True
)
n, d = X.shape
K = Y.shape[1]

assert X.shape == (60000, 784)
assert Y.shape == (60000, 10)
assert X_test.shape == (10000, 784)
assert torch.all(Y.sum(dim=1) == 1)
```

The one-hot matrix is convenient for the derivation. An implementation can instead keep integer labels to avoid storing it, but this note retains the original one-hot formulation.

## 3. Affine model and row-wise softmax

For fixed parameters $(W,b)$, define

$$
z_i=W^\top x_i+b,
\qquad
Z=XW+\mathbf{1}_n b^\top.
$$

In Python, `X @ W + b` broadcasts `b` over samples. The stored shape of `b` remains `(K,)`; broadcasting does not modify the tensor's shape.

The draft's initialization is retained: independently sample the entries of $W$ and $b$ uniformly from $[0,1)$. This initialization is a modeling choice, not an assumption required for the derivative formulas or convexity.

```python
def initialize_parameters(d, K, *, dtype=torch.float32, device="cpu"):
    """Retain the draft's U[0, 1) initialization; seed the caller for repeats."""
    W = torch.rand(d, K, dtype=dtype, device=device)
    b = torch.rand(K, dtype=dtype, device=device)
    return W, b
```

```python
W, b = initialize_parameters(d, K, dtype=X.dtype, device=X.device)
Z = X @ W + b
assert Z.shape == (n, K)
```

For one sample, define $\sigma:\mathbb{R}^K\to(0,1)^K$ by

$$
\sigma(z_i)_j=p_{i,j}
=\frac{\exp(z_{i,j})}{\sum_{\ell=1}^K\exp(z_{i,\ell})}.
$$

The corresponding matrix map $\Sigma:\mathbb{R}^{n\times K}\to(0,1)^{n\times K}$ applies $\sigma$ independently to each row, so $P=\Sigma(Z)$. Each row of $P$ sums to one.

**Flattening the dataset does not turn this into one softmax over $nK$ entries.** Every normalization denominator still includes only the $K$ classes of its own sample.

For numerical stability, introduce a separate shifted quantity

$$
m_i=\max_j z_{i,j},\qquad
\widetilde z_i=z_i-m_i\mathbf{1}_K.
$$

Softmax is invariant to this shift. The mathematical model continues to use the original logits $Z=XW+\mathbf{1}_n b^\top$; shifting is an evaluation technique.

## 4. Per-sample loss and scalar objective

With the target $y_i$ fixed, the per-sample negative log-likelihood is

$$
\ell_i:\mathbb{R}^K\to\mathbb{R},\qquad
\ell_i(z_i)
=-\sum_{j=1}^K y_{i,j}\log p_{i,j}
=-\log p_{i,c_i}.
$$

Using $\sum_j y_{i,j}=1$ gives the equivalent expression

$$
\boxed{
\ell_i(z_i)=\log\left(\sum_{j=1}^K e^{z_{i,j}}\right)-y_i^\top z_i.
}
$$

For finite logits and $K\ge2$, $0<p_{i,c_i}<1$, so $\ell_i(z_i)>0$ in exact arithmetic. In particular, zero is a lower bound. Floating-point evaluation may round a sufficiently small loss to zero.

Distinguish three objects:

$$
\mathcal{L}(Z)=\begin{bmatrix}\ell_1(z_1)\\\vdots\\\ell_n(z_n)\end{bmatrix}
\in\mathbb{R}^n,
\qquad
\Phi(Z)=\frac1n\sum_{i=1}^n\ell_i(z_i),
$$

$$
\boxed{
f(W,b)=\Phi(XW+\mathbf{1}_n b^\top)
=\frac1n\sum_{i=1}^n f_i(W,b),
\qquad
f_i(W,b)=\ell_i(W^\top x_i+b).
}
$$

$\mathcal{L}$ is vector-valued, whereas $\Phi$ and $f$ are scalar-valued. Training minimizes $f$ over $(W,b)$; the logits are determined by the fixed data and these shared parameters.

### Stable implementation

Computing `torch.log(P)` after forming probabilities can produce `-inf` when a probability underflows to zero. Multiplying this by a zero one-hot entry can then produce `NaN`. Compute log-probabilities directly instead:

$$
\log p_{i,j}=\widetilde z_{i,j}
-\log\left(\sum_{\ell=1}^K e^{\widetilde z_{i,\ell}}\right).
$$

```python
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
```

```python
P, losses = forward(X, Y, W, b)
f_value = losses.mean()

assert losses.shape == (n,)
assert f_value.ndim == 0
torch.testing.assert_close(P.sum(dim=1), torch.ones_like(P[:, 0]))
torch.testing.assert_close(f_value, F.cross_entropy(X @ W + b, labels))
```

PyTorch's `F.cross_entropy` expects logits for this call. The comparison uses no class weighting, label smoothing, or ignored labels, so its mean reduction matches $f$ exactly in real arithmetic.

## 5. First-order derivatives

### 5.1 Softmax Jacobian for one sample

Temporarily omit the sample index and write $p=\sigma(z)$. For $j,\ell\in\{1,\ldots,K\}$,

$$
\frac{\partial p_j}{\partial z_\ell}
=\begin{cases}
p_j(1-p_j),&j=\ell,\\
-p_jp_\ell,&j\ne\ell.
\end{cases}
=p_j(\delta_{j\ell}-p_\ell).
$$

Thus,

$$
\boxed{J_\sigma(z)=\operatorname{Diag}(p)-pp^\top\in\mathbb{R}^{K\times K}.}
$$

Here $\delta_{j\ell}$ is the Kronecker delta. Use **Jacobian** notation $J_\sigma$ (or $D\sigma$) for the derivative of this vector-valued function. Its entries are not all positive: the off-diagonal entries are negative for finite logits.

### 5.2 Loss gradient with respect to logits

For one sample, the chain rule gives

$$
\begin{aligned}
\frac{\partial\ell_i}{\partial z_{i,\ell}}
&=-\sum_{j=1}^K y_{i,j}\frac{1}{p_{i,j}}
\frac{\partial p_{i,j}}{\partial z_{i,\ell}}\\
&=-\sum_{j=1}^K y_{i,j}(\delta_{j\ell}-p_{i,\ell})\\
&=-y_{i,\ell}+p_{i,\ell}\sum_{j=1}^K y_{i,j}\\
&=p_{i,\ell}-y_{i,\ell}.
\end{aligned}
$$

Therefore,

$$
\boxed{\nabla_{z_i}\ell_i(z_i)=p_i-y_i\in\mathbb{R}^K,}
\qquad
\boxed{\nabla_Z\Phi(Z)=\frac1n(P-Y)\in\mathbb{R}^{n\times K}.}
$$

These gradients can have negative components even though the losses are nonnegative. A nonnegative loss does not imply a nonnegative gradient.

### 5.3 Gradients with respect to weights and bias

Since $z_{i,j}=\sum_{r=1}^d x_{i,r}w_{r,j}+b_j$,

$$
\frac{\partial f}{\partial w_{r,j}}
=\frac1n\sum_{i=1}^n x_{i,r}(p_{i,j}-y_{i,j}),
\qquad
\frac{\partial f}{\partial b_j}
=\frac1n\sum_{i=1}^n(p_{i,j}-y_{i,j}).
$$

In matrix form,

$$
\boxed{\nabla_W f(W,b)=\frac1n X^\top(P-Y)\in\mathbb{R}^{d\times K},}
$$

$$
\boxed{\nabla_b f(W,b)=\frac1n(P-Y)^\top\mathbf{1}_n\in\mathbb{R}^{K}.}
$$

```python
def manual_gradient(X, Y, W, b):
    """Gradient of the MEAN loss; shapes are (d, K) and (K,)."""
    n = X.shape[0]
    P, _ = forward(X, Y, W, b)
    residual = P - Y
    grad_W = X.T @ residual / n
    grad_b = residual.mean(dim=0)
    return grad_W, grad_b
```

```python
grad_W, grad_b = manual_gradient(X, Y, W, b)
assert grad_W.shape == W.shape  # (784, 10)
assert grad_b.shape == b.shape  # (10,)
```

The original expression `(X.T @ (P - Y)).mean(dim=0)` is incorrect: matrix multiplication has already summed over samples. Applying `.mean(dim=0)` afterward averages over the **feature** dimension, removes that dimension, and returns a vector of length $K$. Divide the matrix product by $n$ instead.

## 6. Second-order derivatives

### 6.1 Hessian with respect to one sample's logits

Because the target is fixed,

$$
\boxed{
C_i:=\nabla^2_{z_i}\ell_i(z_i)
=J_\sigma(z_i)
=\operatorname{Diag}(p_i)-p_i p_i^\top
\in\mathbb{R}^{K\times K}.
}
$$

At fixed logits, $C_i$ does not depend on the target $y_i$. Its entries are

$$
(C_i)_{j\ell}=p_{i,j}(\delta_{j\ell}-p_{i,\ell}).
$$

```python
def logit_hessian_blocks(P):
    """Return C_i = Hessian of one sample loss wrt its K logits: (n, K, K).

    These blocks do not include the 1/n factor for the dataset mean loss.
    They are not the Hessian with respect to W and b.
    """
    return torch.diag_embed(P) - P.unsqueeze(2) * P.unsqueeze(1)
```

```python
C = logit_hessian_blocks(P)
assert C.shape == (n, K, K)
```

The original code

```python
C_original_formula = torch.diag_embed(P) - torch.einsum("ij,il->ijl", P, P)
```

is **correct for the stack $(C_1,\ldots,C_n)$**. It is not yet the parameter Hessian of $f(W,b)$, and it does not include the factor $1/n$ associated with the mean loss.

### 6.2 Hessian of the mean loss with respect to all logits

Define row-wise stacking by

$$
\operatorname{rvec}(Z)=
\begin{bmatrix}z_1\\z_2\\\vdots\\z_n\end{bmatrix}\in\mathbb{R}^{nK}.
$$

Each sample loss depends only on that sample's logits. Consequently,

$$
\boxed{
\nabla^2_{\operatorname{rvec}(Z)}\Phi(Z)
=\frac1n\operatorname{blockdiag}(C_1,\ldots,C_n).
}
$$

The Jacobian of the row-wise softmax is the same block-diagonal structure **without** $1/n$. There are no cross-sample derivative blocks.

For Fashion-MNIST, the dense logit Hessian would have shape $600{,}000\times600{,}000$. It should never be materialized for this model. The small blocks are sufficient for mathematical analysis; the implementation below avoids even storing those blocks when computing a Hessian–vector product.

### 6.3 Parameter ordering

A matrix formula for the parameter Hessian requires an explicit ordering of the parameter entries. Use **row-major** order, matching `W.reshape(-1)`:

$$
\theta=
\begin{bmatrix}\operatorname{rvec}(W)\\b\end{bmatrix}
=(w_{1,1},\ldots,w_{1,K},w_{2,1},\ldots,w_{d,K},b_1,\ldots,b_K)^\top
\in\mathbb{R}^{q},
\qquad q=(d+1)K.
$$

Write $f(\theta)$ for the same objective expressed in these coordinates. Define augmented data and parameters by

$$
a_i=\begin{bmatrix}x_i\\1\end{bmatrix},\qquad
A=[X\ \mathbf{1}_n]\in\mathbb{R}^{n\times(d+1)},\qquad
\Theta=\begin{bmatrix}W\\b^\top\end{bmatrix}.
$$

Then $Z=A\Theta$, $\theta=\operatorname{rvec}(\Theta)$, and

$$
z_i=B_i\theta,\qquad
B_i=a_i^\top\otimes I_K\in\mathbb{R}^{K\times q}.
$$

Here $\otimes$ denotes the Kronecker product. A column-wise vectorization convention would yield a differently ordered Hessian; mixing conventions changes which entry corresponds to which parameter pair.

### 6.4 Full parameter Hessian and its blocks

The logits are affine functions of $\theta$, so their second derivatives vanish. The second-order chain rule therefore gives

$$
\boxed{
H:=\nabla_\theta^2f(\theta)
=\frac1n\sum_{i=1}^n B_i^\top C_i B_i
=\frac1n\sum_{i=1}^n(a_i a_i^\top)\otimes C_i.
}
$$

Equivalently,

$$
H=\begin{bmatrix}H_{WW}&H_{Wb}\\H_{bW}&H_{bb}\end{bmatrix},
$$

where

$$
\begin{aligned}
H_{WW}&=\frac1n\sum_{i=1}^n(x_i x_i^\top)\otimes C_i
&&\in\mathbb{R}^{dK\times dK},\\
H_{Wb}&=\frac1n\sum_{i=1}^n x_i\otimes C_i
&&\in\mathbb{R}^{dK\times K},\\
H_{bW}&=H_{Wb}^\top
&&\in\mathbb{R}^{K\times dK},\\
H_{bb}&=\frac1n\sum_{i=1}^n C_i
&&\in\mathbb{R}^{K\times K}.
\end{aligned}
$$

For direct comparison with the scalar chain rule,

$$
\begin{aligned}
\frac{\partial^2 f}{\partial w_{r,j}\,\partial w_{s,\ell}}
&=\frac1n\sum_{i=1}^n x_{i,r}x_{i,s}(C_i)_{j\ell},\\
\frac{\partial^2 f}{\partial w_{r,j}\,\partial b_\ell}
&=\frac1n\sum_{i=1}^n x_{i,r}(C_i)_{j\ell},\\
\frac{\partial^2 f}{\partial b_j\,\partial b_\ell}
&=\frac1n\sum_{i=1}^n(C_i)_{j\ell}.
\end{aligned}
$$

The original expression $X^\top C_iX$ is not dimensionally valid: $X^\top$ has shape $d\times n$, while $C_i$ has shape $K\times K$. The class and feature dimensions must both be represented, as they are in the formulas above.

### 6.5 Dense Hessian for small verification problems

For Fashion-MNIST, $q=(784+1)10=7{,}850$, so the full parameter Hessian has $61{,}622{,}500$ entries: about $235$ MiB in `float32` or $470$ MiB in `float64`, before intermediate tensors or autograd storage. It is much smaller than the logit Hessian, but repeatedly constructing it is unnecessary for GD or SGD.

The following function is restricted to small problems. It constructs every parameter block in the ordering defined above.

```python
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
```

Do not construct a per-sample tensor of shape $(n,q,q)$: that would be far larger than even the full aggregated Hessian.

### 6.6 Hessian–vector product without forming the Hessian

Let $v=(\operatorname{rvec}(V_W),v_b)$ be a parameter direction. Set

$$
R=XV_W+\mathbf{1}_n v_b^\top,
\qquad
q_i=C_i r_i,
$$

where $r_i$ and $q_i$ are the column vectors represented by row $i$ of $R$ and $Q$. Since

$$
C_i r_i=p_i\odot r_i-p_i(p_i^\top r_i),
$$

we can evaluate each row without building a $K\times K$ matrix:

$$
Q_{i,:}=P_{i,:}\odot
\left(R_{i,:}-\left(\sum_{j=1}^K P_{i,j}R_{i,j}\right)\mathbf{1}_K^\top\right).
$$

The weight and bias parts of $Hv$ are then

$$
\boxed{(Hv)_W=\frac1n X^\top Q,\qquad
(Hv)_b=\frac1n Q^\top\mathbf{1}_n.}
$$

```python
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
```

```python
V_W = torch.randn_like(W)
v_b = torch.randn_like(b)
Hv_W, Hv_b = hessian_vector_product(X, W, b, V_W, v_b)

assert Hv_W.shape == W.shape
assert Hv_b.shape == b.shape
```

This is an **exact** Hessian–vector product in real arithmetic, not a finite-difference approximation. Its leading arithmetic cost is $O(ndK)$, with $O(nK+dK)$ additional storage beyond the input data and parameters; it does not allocate $O(q^2)$ Hessian storage. If necessary, the same sums can be accumulated over data chunks and divided by the total sample count.

## 7. Convexity and implications

### 7.1 Convexity of the parameter objective

A function $f:\mathbb{R}^q\to\mathbb{R}$ is convex if, for all $\theta,\eta\in\mathbb{R}^q$ and $t\in[0,1]$,

$$
f(t\theta+(1-t)\eta)\le t f(\theta)+(1-t)f(\eta).
$$

Here $f$ is twice continuously differentiable. It suffices to show that its Hessian is positive semidefinite everywhere.

For any $u\in\mathbb{R}^K$, let $\bar u=\sum_j p_{i,j}u_j$. Then

$$
\begin{aligned}
u^\top C_i u
&=\sum_{j=1}^Kp_{i,j}u_j^2-
\left(\sum_{j=1}^Kp_{i,j}u_j\right)^2\\
&=\sum_{j=1}^Kp_{i,j}(u_j-\bar u)^2\ge0.
\end{aligned}
$$

Thus $C_i\succeq0$. For any parameter direction $v\in\mathbb{R}^q$,

$$
v^\top H v
=\frac1n\sum_{i=1}^n(B_i v)^\top C_i(B_i v)\ge0.
$$

Therefore, $f$ is **jointly convex in $(W,b)$**. Softmax itself is a nonlinear vector-valued map; the convexity claim concerns the scalar cross-entropy objective composed with the affine model.

### 7.2 Not strictly or strongly convex in the full parameter space

Softmax is invariant to adding the same scalar to every class logit. In particular, for every $t\in\mathbb{R}$,

$$
f(W,b+t\mathbf{1}_K)=f(W,b).
$$

This is a nonconstant line in parameter space on which the objective is constant. Hence $f$ is not strictly convex, and cannot be strongly convex on the full space $\mathbb{R}^q$.

Equivalently, $C_i\mathbf{1}_K=0$, so the nonzero direction $v=(0,\mathbf{1}_K)$ satisfies $Hv=0$ at every parameter value. Strong convexity would require $v^\top Hv\ge\mu\|v\|_2^2$ for some $\mu>0$, which is impossible for this direction.

More generally, adding the same vector to every column of $W$ also leaves the predicted probabilities unchanged. These symmetries explain why a correct Hessian can be singular; singularity is not itself an implementation error.

### 7.3 Lower bound and existence of an optimizer

Since each loss is nonnegative,

$$
f(W,b)\ge0.
$$

This establishes a lower bound; it does **not** establish that $\inf_{W,b}f(W,b)=0$, or that a finite minimizer exists. If every sample's logits were independent optimization variables, each true-class logit could be increased separately to approach zero loss. Here the shared linear model constrains them through $Z=XW+\mathbf{1}_n b^\top$.

For example, if some $(W,b)$ strictly separates every training sample in the multiclass sense, scaling those parameters makes the loss tend to zero, while no finite parameter vector attains zero. No separability or minimizer-existence claim is made here for Fashion-MNIST. Any later convergence theorem involving an optimizer $\theta^\star$ must state its existence as an assumption or justify it.

## 8. Verification and reproducibility

The accompanying script keeps the main gradient and Hessian–vector product formulas manual. Autograd is used as an independent checker. Its reference objective is `F.cross_entropy`, rather than a differentiation of the manual gradient formula.

With a compatible PyTorch installation, run the small offline checks:

```bash
python gd_sgd_verified.py
```

To also check derivatives on actual Fashion-MNIST data, using $128$ training samples with all $784$ features:

```bash
python gd_sgd_verified.py --fashion-mnist --download --data-root data
```

Omit `--download` if the dataset is already present. The dataset check requires torchvision; the offline checks require PyTorch only. These commands print their results and do not train a classifier.

The checks cover:

1. The scalar objective and both parameter gradients against autograd.
2. Every entry of the dense parameter Hessian on small synthetic problems.
3. The block-diagonal logit Hessian, including its $1/n$ scaling.
4. The Hessian–vector product against dense Hessian multiplication.
5. The Hessian–vector product against central differences of the gradient, and a directional derivative against central differences of the objective.
6. Hessian symmetry, nonnegative eigenvalues up to roundoff, and the common-bias-shift null direction.
7. Stable loss evaluation for logits of magnitude $1{,}000$, and a case with tied logits.

Checks use `float64` to reduce numerical error. Small negative eigenvalues close to machine precision are treated as floating-point noise, not evidence of nonconvexity. Numerical agreement supports the implementation; the positive-semidefiniteness proof establishes convexity for all parameter values.

### Recorded verification results

Executed on 18 October 2026 using Python 3.12.14, PyTorch 2.14.1+cpu, and torchvision 0.29.1+cpu. The derivative comparisons use CPU tensors in `float64`.

| Check | Data used | Maximum absolute error | Result |
| --- | --- | --- | --- |
| Parameter gradient vs. autograd | Three synthetic cases: $(n,d,K)=(7,3,4),(1,2,3),(9,4,2)$ | $4.86\times10^{-17}$ | Passed |
| Full parameter Hessian vs. autograd | Same three small cases | $1.11\times10^{-16}$ | Passed |
| HVP vs. finite differences of the gradient | Same three small cases; step $10^{-5}$ | $5.72\times10^{-12}$ | Passed |
| Parameter gradient vs. autograd | First 128 Fashion-MNIST training images, all 784 features | `1.665e-16` | Passed |
| HVP vs. autograd | Same 128 images; all 7,850 parameters | `5.204e-18` | Passed |

All 15 Python code blocks in this document were also executed successfully in sequence, including the forward pass, gradient, stacked logit Hessians, and HVP on the full 60,000-sample training set. Numerical comparisons against autograd on Fashion-MNIST used only the stated 128-sample subset. The full parameter Hessian was verified only on the small synthetic problems. No dense Fashion-MNIST Hessian, complete training run, or empirical convergence comparison is claimed.

## 9. Main corrections relative to the original draft

| Original issue | Correction |
| --- | --- |
| Flattened softmax treated as one normalization over all entries | Softmax remains per sample; the global Jacobian is block diagonal. |
| Loss vector, scalar mean loss, and parameter objective shared inconsistent notation | Separate $\mathcal{L}(Z)$, $\Phi(Z)$, and $f(W,b)$. |
| Positive codomains were assigned to gradients and Jacobians | Derivatives take values in real vector or matrix spaces and may have negative entries. |
| Weight gradient reduced to shape $(K,)$ | Use `X.T @ (P - Y) / n`, of shape $(d,K)$. |
| `log(softmax)` could underflow | Compute log-probabilities using shifted logits and `logsumexp`. |
| Stacked $K\times K$ blocks labeled as the full parameter Hessian | Identify them as $C_i=\nabla^2\ell_i$ with respect to logits. |
| Expressions such as $X^\top C_iX$ mixed class and sample dimensions | Derive $H_{WW}$, both mixed blocks, and $H_{bb}$ under one explicit parameter ordering. |
| Full Hessian construction was left incomplete | Provide a small-problem dense implementation and a scalable exact Hessian–vector product. |
| Convexity section was unfinished | Complete the positive-semidefinite Hessian proof and explain the absence of strict/strong convexity. |

## 10. Implementation references

The derivative and convexity arguments are derived above. The following official documentation specifies the library operations used for loading data and independent checks:

- [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — logits, labels, and mean reduction.
- [PyTorch: torch.logsumexp](https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html) — numerically stabilized log-sum-exp evaluation.
- [PyTorch: torch.autograd.functional.hessian](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html) — Hessian of a scalar-valued function.
- [PyTorch: torch.autograd.functional.hvp](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.hvp.html) — independent Hessian–vector product check.
- [Torchvision: FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) — dataset loading and train/test selection.

\[
A=
\begin{bmatrix}
a_1^\top\\
a_2^\top\\
\vdots\\
a_n^\top
\end{bmatrix}
=
\begin{bmatrix}X & \mathbf{1}_n\end{bmatrix}.
\]
