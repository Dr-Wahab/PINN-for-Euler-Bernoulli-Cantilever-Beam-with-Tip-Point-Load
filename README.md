## Option 1 — Full GitHub README Post

# PINN for Euler–Bernoulli Cantilever Beam with Tip Point Load (Deflection, Moment, Shear)

This repo demonstrates a **Physics-Informed Neural Network (PINN)** solution for a classic structural mechanics benchmark: an **Euler–Bernoulli cantilever beam** subjected to a **point load at the free end**.
The goal is to learn the **deflection field** (w(x)) and recover **rotation**, **bending moment**, and **shear force** using automatic differentiation—without requiring labeled displacement data.

---

## Problem Setup

**Beam:** prismatic cantilever, length (L), rigidity (EI)
**Load:** point load (P) at the free end (x=L)

### Governing equation (interior)

For no distributed load in the span:

[
EI,w''''(x)=0 \quad (0<x<L)
]

### Boundary conditions

* **Fixed end** at (x=0):
  [
  w(0)=0,\qquad w'(0)=0
  ]

* **Free end** at (x=L) with tip load:
  [
  w''(L)=0,\qquad -EI,w'''(L)=P
  ]

> Note: The sign convention is explicitly controlled in the code (downward load can be defined as negative).

---

## PINN Method (What the Network Learns)

A neural network (w_\theta(x)) approximates the deflection. Using **automatic differentiation**, we compute:

* (w'(x)), (w''(x)), (w'''(x)), (w''''(x))

Then we minimize a composite objective:

[
\mathcal{L}=\lambda_f,\mathrm{MSE}\big(w''''(x)\big);+;\lambda_{bc},\mathrm{MSE}(\text{BC residuals})
]

### Engineering outputs (derived from (w(x)))

* Rotation:
  [
  \theta(x)=w'(x)
  ]
* Bending moment:
  [
  M(x)=-EI,w''(x)
  ]
* Shear force:
  [
  V(x)=-EI,w'''(x)
  ]

---

## Stability / “Production-Grade” Implementation Choices

This repo includes standard PINN best practices for higher-order PDEs:

* **Nondimensional coordinate:** (\xi=x/L)
* **Hard enforcement of clamp BCs** using a trial function:
  [
  \bar{w}(\xi)=\xi^2,N_\theta(\xi)
  ]
  This guarantees (w(0)=0) and (w'(0)=0) by construction.
* **float64** to stabilize 4th derivatives
* **Two-stage optimization:** Adam → L-BFGS (for residual polishing)
* Consistent sign convention across **(w, M, V)**

---

## Analytical Validation (QA Gate)

Closed-form solution for the cantilever + tip load:

[
w(x)=\frac{P x^2(3L-x)}{6EI}
]

Shear and moment profiles:
[
V(x)=P \quad (\text{constant}),\qquad M(x)=P(L-x)
]

The script plots:

* PINN vs analytical **deflection**
* PINN vs analytical **moment**
* PINN vs analytical **shear**

and prints numeric checks such as tip deflection, fixed-end moment, and constant shear.

---

## How to Run

```bash
python pinn_cantilever_tipload.py
```

Outputs:

* deflection (w(x))
* rotation (\theta(x))
* bending moment (M(x))
* shear force (V(x))

---

## Use Cases

This example is a clean template for extending PINNs to:

* non-uniform loads (q(x))
* variable stiffness (EI(x))
* beam systems with sparse sensor supervision (hybrid PINNs)
* dynamic beam PDEs (time-dependent PINNs)

---

## References

* Euler–Bernoulli beam theory (standard structural mechanics texts)
* PINNs: Raissi, Perdikaris, Karniadakis (foundational PINN literature)

---

If you use or extend this repo, feel free to cite or fork.

---

## Option 2 — Short GitHub Post (Quick Intro)

# PINN: Cantilever Beam with Tip Load (Euler–Bernoulli)

This repository implements a **Physics-Informed Neural Network (PINN)** for a **cantilever Euler–Bernoulli beam** with a **tip point load**. The model learns the deflection field (w(x)) by minimizing:

* the PDE residual (EI,w''''(x)=0) (interior)
* boundary conditions at the clamp and free end

Using automatic differentiation, the code also recovers:

* (M(x)=-EI,w''(x))
* (V(x)=-EI,w'''(x))

The solution is validated against the analytical formula:
[
w(x)=\frac{P x^2(3L-x)}{6EI}
]

Run:

```bash
python pinn_cantilever_tipload.py
```

