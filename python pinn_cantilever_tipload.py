import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0) NUMERICAL SETTINGS
# ============================================================
torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float64)  # critical for high-order derivatives
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ============================================================
# 1) BEAM PARAMETERS (EDIT THESE)
# ============================================================
L = 2.0        # m
E = 200e9      # Pa
I = 8e-6       # m^4

P = 10e3       # N (magnitude you want)
P_load = -abs(P)   # DOWNWARD load negative (set +abs(P) for upward)

EI = E * I
Pmag = abs(P_load)

# positive scale
w_ref = Pmag * L**3 / EI

print(f"EI     = {EI:.3e} N·m^2")
print(f"P_load = {P_load:.3e} N")
print(f"w_ref  = {w_ref:.3e} m")

# correct nondimensional shear BC target at xi=1
# -|P| * wbar'''(1) = P_load  =>  wbar'''(1) = -P_load/|P|
wbar3_target = -P_load / Pmag
print("Target wbar'''(1) =", float(wbar3_target))


# ============================================================
# 2) PINN MODEL: wbar(xi) with hard clamp BCs
# wbar(0)=0 and wbar'(0)=0 are satisfied exactly by construction.
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xi):
        # trial function
        return (xi**2) * self.net(xi)

model = MLP(hidden=64, depth=4).to(device)


# ============================================================
# 3) AUTODIFF UTILITIES
# ============================================================
def d(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

def wbar_and_derivs(xi):
    """
    Returns wbar, dwbar/dxi, d2wbar/dxi2, d3wbar/dxi3, d4wbar/dxi4
    """
    xi = xi.clone().detach().requires_grad_(True)
    w0 = model(xi)
    w1 = d(w0, xi)
    w2 = d(w1, xi)
    w3 = d(w2, xi)
    w4 = d(w3, xi)
    return w0, w1, w2, w3, w4


# ============================================================
# 4) TRAINING POINTS
# ============================================================
N_f = 6000
xi_f = torch.rand(N_f, 1, device=device).clamp(1e-4, 1 - 1e-4)
xi_1 = torch.tensor([[1.0]], device=device)  # free end


# ============================================================
# 5) LOSS FUNCTION
# PDE (interior): wbar''''(xi) = 0
# Tip BCs: wbar''(1)=0 and wbar'''(1)=wbar3_target
# ============================================================
lam_pde = 1.0
lam_tip = 300.0

def compute_losses():
    # PDE residual
    _, _, _, _, w4 = wbar_and_derivs(xi_f)
    loss_pde = torch.mean(w4**2)

    # Tip BCs
    _, _, w2_1, w3_1, _ = wbar_and_derivs(xi_1)
    loss_tip = torch.mean(w2_1**2) + torch.mean((w3_1 - wbar3_target)**2)

    return loss_pde, loss_tip


# ============================================================
# 6) TRAINING: ADAM -> LBFGS
# ============================================================
adam = optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(1, 6001):
    adam.zero_grad()
    lpde, ltip = compute_losses()
    loss = lam_pde * lpde + lam_tip * ltip
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    adam.step()

    if epoch % 500 == 0:
        print(f"[Adam] Epoch {epoch:4d} | Total {loss.item():.3e} | PDE {lpde.item():.3e} | Tip {ltip.item():.3e}")

lbfgs = optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=600,
    history_size=50,
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()
    lpde, ltip = compute_losses()
    loss = lam_pde * lpde + lam_tip * ltip
    loss.backward()
    return loss

final_loss = lbfgs.step(closure)
print("[LBFGS] Final loss:", float(final_loss))


# ============================================================
# 7) POST-PROCESSING (PHYSICAL UNITS)
# IMPORTANT: V is computed from dM/dx (stable), not from w'''
# ============================================================
# Use xi as grad-enabled tensor to compute dM/dx
xi = torch.linspace(0.0, 1.0, 401, device=device).view(-1, 1).requires_grad_(True)

wbar, wbar1, wbar2, wbar3, _ = wbar_and_derivs(xi)

# coordinates
x_torch = L * xi

# physical fields
w_torch = w_ref * wbar
theta_torch = (w_ref / L) * wbar1

# w''(x) = (w_ref/L^2) * wbar''(xi)
w_xx_torch = (w_ref / L**2) * wbar2

# Moment convention:
# M = -EI * w''(x)
M_torch = -EI * w_xx_torch

# Shear computed robustly from V = dM/dx
V_torch = torch.autograd.grad(
    M_torch, x_torch, grad_outputs=torch.ones_like(M_torch), create_graph=False
)[0]

# Convert to numpy
x = x_torch.detach().cpu().numpy().ravel()
w = w_torch.detach().cpu().numpy().ravel()
theta = theta_torch.detach().cpu().numpy().ravel()
M = M_torch.detach().cpu().numpy().ravel()
V = V_torch.detach().cpu().numpy().ravel()


# ============================================================
# 8) ANALYTICAL SOLUTION (SAME SIGN CONVENTION)
# w(x) = P_load x^2 (3L-x) / (6EI)
# M(x) = P_load (L-x)
# V(x) = P_load (constant)
#
# This matches the governing relation: V = dM/dx
# ============================================================
w_true = (P_load * x**2 * (3.0 * L - x)) / (6.0 * EI)
M_true = P_load * (L - x)
V_true = np.ones_like(x) * P_load

print("MSE(w) =", np.mean((w - w_true) ** 2))
print(f"Tip deflection PINN/TRUE: {w[-1]:.6e} / {P_load*L**3/(3*EI):.6e}")
print(f"M(0) PINN/TRUE: {M[0]:.3e} / {P_load*L:.3e}")
print(f"V(x) PINN/TRUE: {V.mean():.3e} / {P_load:.3e}  (V should be constant)")


# ============================================================
# 9) PLOTS
# ============================================================
plt.figure()
plt.plot(x, w, label="PINN w(x)")
plt.plot(x, w_true, "--", label="Analytical w(x)")
plt.xlabel("x (m)")
plt.ylabel("w (m)")
plt.title("Cantilever beam (tip load): deflection")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(x, M, label="PINN M(x)=-EI w''(x)")
plt.plot(x, M_true, "--", label="Analytical M(x)")
plt.xlabel("x (m)")
plt.ylabel("M (N·m)")
plt.title("Bending moment (linear)")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(x, V, label="PINN V(x)=dM/dx (robust)")
plt.plot(x, V_true, "--", label="Analytical V(x)")
plt.xlabel("x (m)")
plt.ylabel("V (N)")
plt.title("Shear force (constant)")
plt.grid(True)
plt.legend()

plt.show()

