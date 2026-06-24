import torch
import torch.nn as nn
import torch.nn.functional as F
import random

BASE       = "/home/project/Documents/Maharshi/gem/"
DATA_PATH  = BASE + "icec_training_data.pt"
SAVE_PATH  = BASE + "icec_model.pt"
STEPS      = 3000
LR         = 1e-3
LOG_EVERY  = 100
BATCH_SIZE = 8

class ICEC(nn.Module):
    """Input-Conditioned Error Corrector.
    Takes (t, summary) and outputs a low-rank correction to the
    noise prediction shaped [C, H, W]."""
    def __init__(self, n_channels=4, hidden_dim=128):
        super().__init__()
        # Input: timestep embedding (32) + summary (8) = 40
        self.t_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 32),
        )
        # Summary is 8-dim (mean + std per channel)
        self.net = nn.Sequential(
            nn.Linear(32 + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            # Output: per-channel scalar + per-channel bias
            # We'll use a "shift + scale" correction
            # scale: n_channels, bias: n_channels
            nn.Linear(hidden_dim, n_channels * 2),
        )
        # Learnable mixing weight — starts at 0 so no correction initially
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, t, summary, q_pred):
        """
        t:       (B,) scalar timestep
        summary: (B, 8)
        q_pred:  (B, C, H, W) quantized UNet prediction
        Returns: (B, C, H, W) corrected prediction
        """
        t_norm = (t.float() / 1000.0).unsqueeze(-1)  # (B, 1)
        t_emb  = self.t_embed(t_norm)                # (B, 32)
        x      = torch.cat([t_emb, summary], dim=-1) # (B, 40)
        out    = self.net(x)                          # (B, 2*C)
        C      = q_pred.shape[1]
        scale  = out[:, :C].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        bias   = out[:, C:].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Correction: scale and shift applied per channel
        correction = q_pred * scale + bias

        return q_pred + self.alpha * correction, correction

# ── Load data ─────────────────────────────────────────────────────────
print("Loading training data...")
data = torch.load(DATA_PATH, map_location="cpu")
print(f"  {len(data)} samples")

# ── Model ─────────────────────────────────────────────────────────────
model = ICEC(n_channels=4, hidden_dim=128).to("cuda")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):.0f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=STEPS, eta_min=LR/10
)

# ── Training ──────────────────────────────────────────────────────────
print(f"\nTraining ICEC for {STEPS} steps | batch={BATCH_SIZE}\n")
model.train()

best_loss  = float('inf')
best_state = None

for step in range(1, STEPS + 1):
    batch_idx = random.sample(range(len(data)), BATCH_SIZE)
    ts       = torch.tensor([data[i]['t'] for i in batch_idx],
                            device="cuda")
    summary  = torch.stack([data[i]['summary'] for i in batch_idx]
                           ).to("cuda")
    residual = torch.stack([data[i]['residual'] for i in batch_idx]
                           ).to("cuda")

    # We need q_pred to predict scale+bias on. Use residual as proxy
    # (it's added to q_pred to get fp_pred, and the correction is meant
    # to be close to residual when alpha=1)
    # To train: make correction ≈ residual, then alpha learns optimal mix
    dummy_q_pred = torch.zeros_like(residual)  # we'll replace logic

    # Actually simpler: train correction directly to match residual
    t_norm = (ts.float() / 1000.0).unsqueeze(-1)
    t_emb  = model.t_embed(t_norm)
    x      = torch.cat([t_emb, summary], dim=-1)
    out    = model.net(x)
    C      = residual.shape[1]

    # Output per-channel scalar + spatial mean — the simplest useful form
    pred_scale = out[:, :C]        # (B, C) — per-channel scale
    pred_bias  = out[:, C:]        # (B, C) — per-channel additive

    # Target: decompose residual into per-channel scale*input + bias
    # Simplification: just predict the per-channel mean of residual
    target_mean = residual.mean(dim=[2,3])  # (B, C)
    target_std  = residual.std(dim=[2,3])   # (B, C)

    loss_mean = F.mse_loss(pred_bias,  target_mean)
    loss_std  = F.mse_loss(pred_scale, target_std)
    loss = loss_mean + loss_std

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    lr_sched.step()

    loss_val = loss.item()
    if loss_val < best_loss:
        best_loss  = loss_val
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if step % LOG_EVERY == 0:
        print(f"  step {step:5d}/{STEPS} | "
              f"loss={loss_val:.6f} | best={best_loss:.6f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

# ── Save ──────────────────────────────────────────────────────────────
torch.save(best_state, SAVE_PATH)
print(f"\nBest loss: {best_loss:.6f}")
print(f"Saved: {SAVE_PATH}")