# PIINs vs Numerical Methods: A Comparative Study for Solving PDEs

This project presents a comparative study between traditional high-order numerical schemes and Physics-Informed Implicit Neural Networks (PIINs) for solving partial differential equations (PDEs). We implement both approaches for elliptic, parabolic, and hyperbolic equations and evaluate them on accuracy, efficiency, and stability.

---

## Key Points

- **Objective**: To develop and compare high-order numerical schemes and PIIN-based SciML models for solving PDEs.
- **Focus**: Elliptic (Poisson), Parabolic (Heat), and Hyperbolic (Wave) equations.
- **Approaches**: Finite Difference Methods (FDM) vs Physics-Informed Implicit Neural Networks.
- **Analysis**: Error evaluation, convergence analysis, and runtime comparisons.
- **Goal**: Understand the trade-offs between traditional numerical solvers and SciML-based solvers.

---

## Governing Equations (Operator Form)

### Heat Equation
$$
\mathcal{N}[u](x, t) := \frac{\partial u}{\partial t} - \alpha \nabla^2 u(x, t) - f(x, t) = 0
$$

### Poisson Equation
$$
\mathcal{N}[u] (x) := -\nabla^2 u(x) - f(x) = 0
$$

### Wave Equation
$$
\mathcal{N}[u](x, t) := \frac{\partial^2 u}{\partial t^2} - c^2 \nabla^2 u(x, t) - f(x, t) = 0
$$

---

## Methods

### 1. Classical Numerical Methods

Used for solving elliptic, parabolic, and hyperbolic PDEs via deterministic schemes.

**Key Techniques:**

- **Second-Order Finite Difference Method (FDM)**
- **Fourth-Order Finite Difference Method** (classical or compact)
- **Explicit Scheme** – FTCS (Forward Time Centered Space)
- **Implicit Scheme** – BTCS (Backward Time Centered Space)
- **Crank-Nicolson Scheme** – Semi-implicit time discretization
- **Central Difference Scheme** – For spatial derivatives in hyperbolic PDEs
- **Leapfrog Method** – For second-order hyperbolic equations
- **For solving system of linear equation**:
 - **Thomas Algorithm** – Efficient solution of tridiagonal systems
 - **Sparse Matrix Solvers** – Direct or iterative approaches

#### Example: Second-Order Finite Difference Approximation
$$
\frac{d^2 u}{dx^2}=\frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} + \mathcal{O}(h^2)
$$

---

### 2. Physics-Informed Implicit Neural Networks (PIINs)

A deep learning-based approach to approximate the solution of PDEs by minimizing residuals.

**Key Techniques:**

- **Physics-Informed Neural Networks (PINNs)**
- **Physics-Informed Implicit Neural Networks (PIINs)**
- **Automatic Differentiation** – For computing spatial and temporal derivatives
- **Collocation Method** – Sampling interior and boundary points
- **Loss Function Decomposition**:
  - **PDE Residual Loss** – $\mathcal{L}_{res}$
  - **Initial Condition Loss** – $\mathcal{L}_{IC}$
  - **Boundary Condition Loss** – $\mathcal{L}_{BC}$
- **Optimization Algorithms**:
  - **Adam Optimizer**
  - **L-BFGS Optimizer**

### 2. Physics-Informed Implicit Neural Networks (PIINs)

A neural network $u_\theta(x, t)$ is trained to satisfy the PDE and boundary conditions by minimizing the residual loss.

**Loss Function:**
$L(\theta) = L_{res} + L_{BC} + L_{IC}$


- $L_{res}$: PDE residual using automatic differentiation
- $L_{BC}$: Boundary condition loss
- $L_{IC}$: Initial condition loss

Training is performed using optimizers like Adam or L-BFGS. The solution is approximated over a collocation point grid.

---

## Comparative Summary

| PDE Type   | Classical Method                     | PIIN Approach                               |
|------------|--------------------------------------|---------------------------------------------|
| Elliptic   | FDM + Sparse Solvers                 | Physics loss + Boundary constraints         |
| Parabolic  | Explicit/Implicit/Crank-Nicolson     | PDE loss + IC/BC enforced during training   |
| Hyperbolic | Central Difference / Leapfrog        | Second-order time loss + autograd           |

---

## Results

- Error surface plots
- Convergence graphs
- Runtime comparisons

## Contributors

- [Mamta Saini](https://github.com/mamtasaini003)
- [Manoj Solanki](https://github.com/manojms3063)
