# pip install torch numpy
import math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 1) Model: C_theta (MLP)
# ---------------------------
class EmbedNet(nn.Module):
    def __init__(self, L, K, hidden=(256, 128), unit_norm=True):
        super().__init__()
        layers = []
        d_prev = L
        for d in hidden:
            layers += [nn.Linear(d_prev, d), nn.ReLU()]
            d_prev = d
        layers += [nn.Linear(d_prev, K)]
        self.net = nn.Sequential(*layers)
        self.unit_norm = unit_norm

    def forward(self, x):
        z = self.net(x)
        if self.unit_norm:
            z = F.normalize(z, p=2, dim=1)  # constrain norms if desired
        return z

# ---------------------------
# 2) Full Frobenius loss
#    (use when N is modest)
# ---------------------------
def fit_full(A, B, K, hidden=(256,128), unit_norm=True, lr=1e-3, wd=1e-4,
             steps=5000, device="cpu", exclude_diag=True):
    """
    A: (N,N) symmetric torch tensor
    B: (N,L) torch tensor
    """
    N, L = B.shape
    model = EmbedNet(L, K, hidden, unit_norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    A = A.to(device)
    B = B.to(device)
    mask = None
    if exclude_diag:
        mask = ~torch.eye(N, dtype=torch.bool, device=device)

    best = math.inf
    best_state = None

    for t in range(steps):
        opt.zero_grad()
        Z = model(B)            # (N,K)
        S = Z @ Z.t()           # (N,N)
        if mask is None:
            loss = F.mse_loss(S, A)
        else:
            diff = (S - A)[mask]
            loss = (diff * diff).mean()
        loss.backward()
        opt.step()

        if loss.item() < best:
            best, best_state = loss.item(), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (t+1) % 500 == 0:
            print(f"[full] step {t+1}/{steps} | loss={loss.item():.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ---------------------------------------
# 3) Pair-sampled loss (scales to big N)
# ---------------------------------------
class PairDataset(Dataset):
    def __init__(self, A, B, num_pairs_per_epoch=200_000, exclude_diag=True, seed=0):
        self.A = A
        self.B = B
        self.N = A.shape[0]
        self.exclude_diag = exclude_diag
        rng = np.random.default_rng(seed)
        # Pre-sample index pairs for one epoch; you can resample each epoch if you want
        I = rng.integers(0, self.N, size=(num_pairs_per_epoch,))
        J = rng.integers(0, self.N, size=(num_pairs_per_epoch,))
        if exclude_diag:
            mask = I != J
            I, J = I[mask], J[mask]
        self.pairs = np.stack([I, J], axis=1)

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return i, j, self.A[i, j]

def fit_pairs(A, B, K, hidden=(256,128), unit_norm=True, lr=2e-3, wd=1e-4,
              epochs=10, batch_size=4096, num_pairs_per_epoch=200_000,
              device="cpu"):
    """
    Minimizes E[(<z_i,z_j> - A_ij)^2] over sampled (i,j) pairs.
    """
    N, L = B.shape
    model = EmbedNet(L, K, hidden, unit_norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Tensors to device once
    A = A.to(device)
    B = B.to(device)

    # Precompute for quick row fetch
    B_cpu = B.detach().cpu()

    for ep in range(epochs):
        ds = PairDataset(A.detach().cpu(), B_cpu, num_pairs_per_epoch=num_pairs_per_epoch, exclude_diag=True, seed=ep)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        running = 0.0
        for i_idx, j_idx, aij in dl:
            i_idx = i_idx.to(device)
            j_idx = j_idx.to(device)
            aij = aij.to(device)

            # Pull rows and embed
            Bi = B[i_idx]     # (B,L)
            Bj = B[j_idx]     # (B,L)
            Zi = model(Bi)    # (B,K)
            Zj = model(Bj)    # (B,K)

            # Predicted similarity: dot product
            sij = (Zi * Zj).sum(dim=1)  # (B,)
            loss = F.mse_loss(sij, aij)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * len(i_idx)

        print(f"[pairs] epoch {ep+1}/{epochs} | loss={running/len(ds):.6f}")

    return model

# ---------------------------
# 4) Usage example
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, L, K = 200, 50, 8

    # Toy data
    B_np = np.random.randn(N, L).astype(np.float32)
    # Target similarity A (symmetric). Example: cosine similarity of B, just for demo.
    Bn = B_np / (np.linalg.norm(B_np, axis=1, keepdims=True) + 1e-9)
    A_np = Bn @ Bn.T

    A = torch.tensor(A_np, dtype=torch.float32)
    B = torch.tensor(B_np, dtype=torch.float32)

    # Small N -> full loss; large N -> use fit_pairs
    model = fit_full(A, B, K, hidden=(256,128), unit_norm=True, lr=1e-3, wd=1e-4, steps=3000)

    # Get embeddings and reconstruction
    with torch.no_grad():
        Z = model(B)                 # (N,K)
        S = Z @ Z.t()                # approx of A
        mse = F.mse_loss(S, A).item()
        print(f"Final full-matrix MSE: {mse:.6f}")

    # For large N: uncomment to train with pairs
    # model = fit_pairs(A, B, K, epochs=10)