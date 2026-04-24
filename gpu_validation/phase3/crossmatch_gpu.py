# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
GPU-accelerated SN-void crossmatch with batched nearest-void search.

Never stores the full (N_sn, N_void) distance matrix — batches over
void chunks and updates best_dnorm incrementally per SN.
"""

import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def compute_environment_gpu(sn_pos, void_pos, void_r, chunk_size=10000):
    """Batched GPU nearest-void search.

    Args:
        sn_pos: (N_sn, 3) array — SN comoving positions in Mpc/h
        void_pos: (N_void, 3) array — void center positions in Mpc/h
        void_r: (N_void,) array — void radii in Mpc/h
        chunk_size: number of voids per batch (controls VRAM usage)

    Returns:
        d_signed: (N_sn,) signed distance to nearest void boundary
                  negative = inside void, positive = outside
        nearest_idx: (N_sn,) index into void_pos of nearest void
        in_void: (N_sn,) boolean, True if SN is inside any void
    """
    if not HAS_GPU:
        print("  [WARNING] CuPy not available, falling back to CPU")
        return compute_environment_cpu(sn_pos, void_pos, void_r)

    N_sn = len(sn_pos)
    N_void = len(void_pos)

    # Transfer SN positions to GPU once
    sn_gpu = cp.asarray(sn_pos, dtype=cp.float64)

    best_dnorm = cp.full(N_sn, cp.inf, dtype=cp.float64)
    best_idx = cp.full(N_sn, -1, dtype=cp.int32)
    best_dsigned = cp.full(N_sn, cp.inf, dtype=cp.float64)

    for i_start in range(0, N_void, chunk_size):
        i_end = min(i_start + chunk_size, N_void)
        vp_chunk = cp.asarray(void_pos[i_start:i_end], dtype=cp.float64)
        vr_chunk = cp.asarray(void_r[i_start:i_end], dtype=cp.float64)

        # Pairwise distances: (N_sn, chunk_len)
        diff = sn_gpu[:, None, :] - vp_chunk[None, :, :]
        dist = cp.sqrt(cp.sum(diff ** 2, axis=2))
        dnorm = dist / vr_chunk[None, :]

        # Best within this chunk
        chunk_best_local = cp.argmin(dnorm, axis=1)
        chunk_best_dnorm = dnorm[cp.arange(N_sn), chunk_best_local]

        # Update global best where this chunk is better
        update_mask = chunk_best_dnorm < best_dnorm
        if cp.any(update_mask):
            best_dnorm[update_mask] = chunk_best_dnorm[update_mask]
            best_idx[update_mask] = (i_start + chunk_best_local[update_mask]).astype(cp.int32)

            # d_signed for the new best entries
            best_dist = dist[cp.arange(N_sn), chunk_best_local]
            best_vr = vr_chunk[chunk_best_local]
            new_dsigned = (best_dist - best_vr) / best_vr
            best_dsigned[update_mask] = new_dsigned[update_mask]

    in_void = best_dsigned < 0.0

    return best_dsigned.get(), best_idx.get(), in_void.get()


def compute_environment_cpu(sn_pos, void_pos, void_r, chunk_size=10000):
    """CPU reference implementation — same batched algorithm with NumPy.

    Used for cross-checking GPU results.
    """
    N_sn = len(sn_pos)
    N_void = len(void_pos)

    best_dnorm = np.full(N_sn, np.inf)
    best_idx = np.full(N_sn, -1, dtype=np.int32)
    best_dsigned = np.full(N_sn, np.inf)

    for i_start in range(0, N_void, chunk_size):
        i_end = min(i_start + chunk_size, N_void)
        vp_chunk = void_pos[i_start:i_end]
        vr_chunk = void_r[i_start:i_end]

        # Pairwise distances: (N_sn, chunk_len)
        diff = sn_pos[:, None, :] - vp_chunk[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        dnorm = dist / vr_chunk[None, :]

        chunk_best_local = np.argmin(dnorm, axis=1)
        chunk_best_dnorm = dnorm[np.arange(N_sn), chunk_best_local]

        update_mask = chunk_best_dnorm < best_dnorm
        if np.any(update_mask):
            best_dnorm[update_mask] = chunk_best_dnorm[update_mask]
            best_idx[update_mask] = (i_start + chunk_best_local[update_mask]).astype(np.int32)

            best_dist = dist[np.arange(N_sn), chunk_best_local]
            best_vr = vr_chunk[chunk_best_local]
            new_dsigned = (best_dist - best_vr) / best_vr
            best_dsigned[update_mask] = new_dsigned[update_mask]

    in_void = best_dsigned < 0.0

    return best_dsigned, best_idx, in_void


def crosscheck_gpu_cpu(sn_pos, void_pos, void_r, label=""):
    """Run both GPU and CPU crossmatch and assert results match.

    Returns GPU results if they match, raises AssertionError otherwise.
    """
    d_gpu, idx_gpu, inv_gpu = compute_environment_gpu(sn_pos, void_pos, void_r)
    d_cpu, idx_cpu, inv_cpu = compute_environment_cpu(sn_pos, void_pos, void_r)

    d_match = np.allclose(d_gpu, d_cpu, atol=1e-6, rtol=1e-6)
    idx_match = np.array_equal(idx_gpu, idx_cpu)
    inv_match = np.array_equal(inv_gpu, inv_cpu)

    status = "PASS" if (d_match and idx_match and inv_match) else "FAIL"
    print(f"  [CPU crosscheck {label}] d_signed: {d_match}, idx: {idx_match}, "
          f"in_void: {inv_match} => {status}")

    if not d_match:
        max_diff = np.max(np.abs(d_gpu - d_cpu))
        print(f"    max |d_gpu - d_cpu| = {max_diff:.2e}")

    assert d_match and idx_match and inv_match, \
        f"GPU/CPU crossmatch mismatch for {label}"

    return d_gpu, idx_gpu, inv_gpu
