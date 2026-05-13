"""Regenerate /tmp/llama_probe_full/rope_{cos,sin}.npy at 1024 positions
for LLaMA-3.1-8B (matches setup_probe_data.py logic but pure numpy,
no HF model load — doesn't disturb the running sweep on GPU 0)."""
import numpy as np

HEAD_DIM = 128
ROPE_BASE = 500000.0
POSITIONS = 1024  # headroom beyond 512

# Standard LLaMA-3.1 NTK-aware scaling
FACTOR = 8.0
LOW_FREQ_FACTOR = 1.0
HIGH_FREQ_FACTOR = 4.0
OLD_MAX = 8192.0

inv_freq = 1.0 / (ROPE_BASE ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
low_wavelen = OLD_MAX / LOW_FREQ_FACTOR
high_wavelen = OLD_MAX / HIGH_FREQ_FACTOR
wavelens = 2 * np.pi / inv_freq
smooth = (OLD_MAX / wavelens - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
is_high = wavelens < high_wavelen
is_low = wavelens > low_wavelen
scaled = inv_freq.copy()
scaled = np.where(is_low, inv_freq / FACTOR, scaled)
mid = (~is_low) & (~is_high)
scaled = np.where(mid, (1 - smooth) * inv_freq / FACTOR + smooth * inv_freq, scaled)
inv_freq = scaled

pos = np.arange(POSITIONS, dtype=np.float64)
freqs = np.einsum("i,j->ij", pos, inv_freq)
emb = np.concatenate([freqs, freqs], axis=-1)
np.save("/tmp/llama_probe_full/rope_cos.npy", np.cos(emb))
np.save("/tmp/llama_probe_full/rope_sin.npy", np.sin(emb))
print(f"rope_cos/sin shape = ({POSITIONS}, {HEAD_DIM})")
