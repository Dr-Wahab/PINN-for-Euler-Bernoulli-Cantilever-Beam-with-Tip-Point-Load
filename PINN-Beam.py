import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------------
# 0) settings
# ----------------------------
torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ----------------------------
# 1) beam inputs
# ----------------------------
L = 2.0
E = 200e9
I = 8e-6
P = 10e3

# Choose physical load sign here:
P_load = -abs(P)   # downward
# P_load = +abs(P) # upward

EI = E * I
Pmag = abs(P_load)

# Scaling (positive)
w_ref = Pmag * L**3 / EI

print(f"EI     = {EI:.3e} N·m^2")
print(f"P_load = {P_load:.3e} N")
print(f"w_ref  = {w_ref:.3e} m")

# ----------------------------
# 2) PINN: wbar(xi) with hard clamp BCs
# wbar(0)=0 and wbar'(0)=0 enforced by construction: wbar = xi^2*N(xi)
# ----------------------------
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xi):
        return (xi**2) * self.net(xi)

model = MLP().to(device)

def d(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

def derivs(xi):
    xi = xi.clone().detach().requires_grad_(True)
    w  = model(xi)
    w1 = d(w, xi)
    w2 = d(w1, xi)
    w3 = d(w2, xi)
    w4 = d(w3, xi)
    return w, w1, w2, w3, w4

# ----------------------------
# 3) collocation + boundary points
# ----------------------------
N_f = 5000
xi_f = torch.rand(N_f, 1, device=device).clamp(1e-4, 1-1e-4)
xi1  = torch.tensor([[1.0]], device=device)

# ----------------------------
# 4) correct nondimensional tip BC target
# From: -|P| * wbar'''(1) = P_load  => wbar'''(1) = -P_load/|P|
wbar3_target = -P_load / Pmag  # = +1 for downward, -1 for upward
print("wbar'''(1) target =", float(wbar3_target))

lam_pde = 1.0
lam_tip = 200.0

def losses():
    # PDE: wbar'''' = 0
    _, _, _, _, w4 = derivs(xi_f)
    lpde = torch.mean(w4**2)

    # Tip: wbar''(1)=0 and wbar'''(1)=wbar3_target
    _, _, w2_1, w3_1, _ = derivs(xi1)
    ltip = torch.mean(w2_1**2) + torch.mean((w3_1 - wbar3_target)**2)
    return lpde, ltip

# ----------------------------
# 5) train: Adam -> LBFGS
# ----------------------------
adam = optim.Adam(model.parameters(), lr=3e-4)
for epoch in range(1, 5001):
    adam.zero_grad()
    lpde, ltip = losses()
    loss = lam_pde*lpde + lam_tip*ltip
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    adam.step()
    if epoch % 500 == 0:
        print(f"[Adam] Epoch {epoch:4d} | Total {loss.item():.3e} | PDE {lpde.item():.3e} | Tip {ltip.item():.3e}")

lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, history_size=50, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad()
    lpde, ltip = losses()
    loss = lam_pde*lpde + lam_tip*ltip
    loss.backward()
    return loss
final_loss = lbfgs.step(closure)
print("[LBFGS] Final loss:", float(final_loss))

# ----------------------------
# 6) post-process to physical units
# ----------------------------
xi = torch.linspace(0, 1, 401, device=device).view(-1, 1)
wbar, wbar1, wbar2, wbar3, _ = derivs(xi)

x = (L * xi).detach().cpu().numpy().ravel()

# physical fields (NO extra sign multipliers!)
w = (w_ref * wbar).detach().cpu().numpy().ravel()
theta = ((w_ref / L) * wbar1).detach().cpu().numpy().ravel()

M = (-EI * (w_ref / L**2) * wbar2).detach().cpu().numpy().ravel()
V = (-EI * (w_ref / L**3) * wbar3).detach().cpu().numpy().ravel()

# ----------------------------
# 7) analytical (same P_load)
# ----------------------------
x_t = torch.tensor(x, device=device).view(-1, 1)
w_true = (P_load * x_t**2 * (3*L - x_t)) / (6*EI)   # deflection sign follows P_load
M_true = P_load * (L - x_t)                         # moment
V_true = P_load * torch.ones_like(x_t)              # shear constant

w_true = w_true.detach().cpu().numpy().ravel()
M_true = M_true.detach().cpu().numpy().ravel()
V_true = V_true.detach().cpu().numpy().ravel()

print("MSE(w) =", np.mean((w - w_true)**2))
print("V_tip PINN/TRUE =", V[-1], V_true[-1])
print("M_fixed PINN/TRUE =", M[0], M_true[0])

# ----------------------------
# 8) plots
# ----------------------------
plt.figure()
plt.plot(x, w, label="PINN w(x)")
plt.plot(x, w_true, "--", label="Analytical w(x)")
plt.xlabel("x (m)"); plt.ylabel("w (m)")
plt.title("Cantilever beam (tip load): deflection")
plt.grid(True); plt.legend()

plt.figure()
plt.plot(x, M, label="PINN M(x)=-EI w''")
plt.plot(x, M_true, "--", label="Analytical M(x)")
plt.xlabel("x (m)"); plt.ylabel("M (N·m)")
plt.title("Bending moment (linear)")
plt.grid(True); plt.legend()

plt.figure()
plt.plot(x, V, label="PINN V(x)=-EI w'''")
plt.plot(x, V_true, "--", label="Analytical V(x)")
plt.xlabel("x (m)"); plt.ylabel("V (N)")
plt.title("Shear force (constant)")
plt.grid(True); plt.legend()

plt.show()