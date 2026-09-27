import os
import sys
import time
import h5py
import numpy as np
import scipy.ndimage as ndimage
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor

from GENESIS_write import (
    parse_config_file,
    parse_beamline_file,
    config_to_string,
    beamline_to_string,
)

# ----------------------------
# User / runtime parameters
# ----------------------------

# Small transverse smoothing strength as a fraction of slice rms size.
# 0.03 ~ 0.08 is usually reasonable.
XY_SMOOTHING = 0.05

# Cap process count to avoid oversubscription.
MAX_WORKERS_CAP = 32


# ----------------------------
# Low-discrepancy / random helper
# ----------------------------

def HaltonRandomNumber(dims, nb_pts):
    hArr = np.empty(nb_pts * dims, dtype=np.float64)
    pArr = np.empty(nb_pts, dtype=np.float64)

    Primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
        67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
        139, 149, 151, 157, 163
    ]

    log_nb_pts = np.log(nb_pts + 1)
    for i in range(dims):
        b = Primes[i]
        n = int(np.ceil(log_nb_pts / np.log(b)))
        for t in range(n):
            pArr[t] = b ** (-(t + 1))
        for j in range(nb_pts):
            d = j + 1
            s = (d % b) * pArr[0]
            for t in range(1, n):
                d = d // b
                s += (d % b) * pArr[t]
            hArr[j * dims + i] = s
    return hArr.reshape(nb_pts, dims)


# ----------------------------
# H5 reader
# ----------------------------

def read_h5_file(h5_filename):
    datasets = {}
    with h5py.File(h5_filename, 'r') as h5file:
        def extract_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[:]
        h5file.visititems(extract_datasets)
    return datasets


# ----------------------------
# Slice generation
# ----------------------------

def _slice_worker(slice_number):
    return SliceCalculate(
        z_hlt[:, 1], slice_number, StepZ, NumberOfSlices,
        f_Z, Non_Zero_Z,
        Num_Of_Slice_Particles, minz, XY_SMOOTHING, RandomHaltonSequence
    )


def SliceCalculate(z_hlt, i, StepZ, NumberOfSlices,
                   f_Z, Non_Zero_Z,
                   Num_Of_Slice_Particles, minz, JDFSmoothing, RandomHaltonSequence):
    """
    Direct raw-slice 6D generator with small smooth transverse displacement.

    Important preserved logic:
    - particles are sampled from the ORIGINAL quiet slice i
    - only x and y are given a small smooth offset
    - quiet z placement ZZZ_in is kept for slice attachment and later ordering
    """
    z_center = slice_centers[i]
    ZZZ_in = z_center + StepZ * z_hlt
    NoOfElec = (Non_Zero_Z) * (f_Z(z_center) / NumberOfSlices) / Num_Of_Slice_Particles

    out = np.empty((Num_Of_Slice_Particles, 7), dtype=np.float64)

    idx_src = slice_src_indices.get(i, None)

    # Fallback if source slice is empty
    if idx_src is None or idx_src.size == 0:
        if global_weight_sum > 0:
            x_mid = global_x_mean
            y_mid = global_y_mean
            px_mid = global_px_mean
            py_mid = global_py_mean
            pz_mid = global_pz_mean
        else:
            x_mid = y_mid = px_mid = py_mid = pz_mid = 0.0

        out[:, 0] = x_mid
        out[:, 1] = y_mid
        out[:, 2] = ZZZ_in
        out[:, 3] = NoOfElec
        out[:, 4] = px_mid
        out[:, 5] = py_mid
        out[:, 6] = pz_mid
        return out

    x_slice = mA_X[idx_src]
    y_slice = mA_Y[idx_src]
    px_slice = mA_PX[idx_src]
    py_slice = mA_PY[idx_src]
    pz_slice = mA_PZ[idx_src]
    w_slice = mA_WGHT[idx_src]

    good = (
        np.isfinite(x_slice) &
        np.isfinite(y_slice) &
        np.isfinite(px_slice) &
        np.isfinite(py_slice) &
        np.isfinite(pz_slice) &
        np.isfinite(w_slice) &
        (w_slice > 0.0)
    )

    x_slice = x_slice[good]
    y_slice = y_slice[good]
    px_slice = px_slice[good]
    py_slice = py_slice[good]
    pz_slice = pz_slice[good]
    w_slice = w_slice[good]

    if x_slice.size == 0:
        out[:, 0] = global_x_mean
        out[:, 1] = global_y_mean
        out[:, 2] = ZZZ_in
        out[:, 3] = NoOfElec
        out[:, 4] = global_px_mean
        out[:, 5] = global_py_mean
        out[:, 6] = global_pz_mean
        return out

    p = w_slice / np.sum(w_slice)
    cdf = np.cumsum(p)
    cdf[-1] = 1.0

    mx = np.sum(p * x_slice)
    my = np.sum(p * y_slice)
    sx = np.sqrt(np.sum(p * (x_slice - mx) ** 2))
    sy = np.sqrt(np.sum(p * (y_slice - my) ** 2))
    sx = max(sx, 1e-16)
    sy = max(sy, 1e-16)

    smooth_x = float(JDFSmoothing) * sx if JDFSmoothing is not None else 0.0
    smooth_y = float(JDFSmoothing) * sy if JDFSmoothing is not None else 0.0
    clip_x = 3.0 * smooth_x
    clip_y = 3.0 * smooth_y

    u_sel = RandomHaltonSequence[:, 0]
    u_dx = RandomHaltonSequence[:, 1]
    u_dy = RandomHaltonSequence[:, 2]

    idx = np.searchsorted(cdf, u_sel, side='right')
    idx = np.minimum(idx, x_slice.size - 1)

    dx = np.clip((2.0 * u_dx - 1.0) * smooth_x, -clip_x, clip_x)
    dy = np.clip((2.0 * u_dy - 1.0) * smooth_y, -clip_y, clip_y)

    out[:, 0] = x_slice[idx] + dx
    out[:, 1] = y_slice[idx] + dy
    out[:, 2] = ZZZ_in
    out[:, 3] = NoOfElec
    out[:, 4] = px_slice[idx]
    out[:, 5] = py_slice[idx]
    out[:, 6] = pz_slice[idx]
    return out


# ----------------------------
# Shot noise
# ----------------------------

def add_shot_noise_to_quiet_z(
    Z0,
    Ne0,
    cell_length,
    rng=None,
    keep_local_mean=True,
    max_dtheta=None,
    n_harmonics=4,
    n_grid=1024,
    min_pdf=1e-12,
    z_reference=None,
):
    """
    Add shot noise to quiet longitudinal positions in a GENESIS-friendly way.

    Preserved interface:
    - returns Z_new and info
    - noise is applied in wavelength-sized cells

    Important implementation choice:
    - quiet slice attachment is preserved in the main program by sorting and
      grouping with Full_Z0, while the loaded microscopic phase is carried by
      dtheta / theta_new.
    """
    if rng is None:
        rng = np.random.default_rng()

    Z0 = np.asarray(Z0, dtype=np.float64)
    Ne0 = np.asarray(Ne0, dtype=np.float64)

    if Z0.ndim != 1 or Ne0.ndim != 1 or len(Z0) != len(Ne0):
        raise ValueError("Z0 and Ne0 must be 1D arrays of the same length.")
    if np.any(Ne0 <= 0):
        raise ValueError("All Ne0 must be positive.")
    if cell_length <= 0:
        raise ValueError("cell_length must be positive.")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1.")
    if n_grid < 16:
        raise ValueError("n_grid must be >= 16.")

    if z_reference is None:
        z_reference = Z0.min()

    cell_idx = np.floor((Z0 - z_reference) / cell_length).astype(np.int64)

    Z_new = Z0.copy()
    dtheta = np.zeros_like(Z0)
    theta_new_all = np.zeros_like(Z0)
    theta_old_all = np.zeros_like(Z0)

    dtheta_grid = 2.0 * np.pi / n_grid
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_grid + 1)
    theta_centers = theta_edges[:-1] + 0.5 * dtheta_grid

    harmonics = np.arange(1, n_harmonics + 1, dtype=np.float64)[:, None]
    exp_table = np.exp(-1j * harmonics * theta_centers[None, :])

    order = np.argsort(cell_idx, kind='mergesort')
    cell_sorted = cell_idx[order]

    if cell_sorted.size == 0:
        info = {
            "dtheta": dtheta,
            "theta_old": theta_old_all,
            "theta_new": theta_new_all,
            "cell_idx": cell_idx,
            "b_targets": np.zeros((0, n_harmonics), dtype=np.complex128),
            "Nreal_cells": np.zeros(0, dtype=np.float64),
        }
        return Z_new, info

    unique_cells, starts, counts = np.unique(
        cell_sorted, return_index=True, return_counts=True
    )

    n_cells = int(cell_idx.max()) + 1
    b_targets = np.zeros((n_cells, n_harmonics), dtype=np.complex128)
    Nreal_cells = np.zeros(n_cells, dtype=np.float64)

    for icell, start, count in zip(unique_cells, starts, counts):
        block = order[start:start + count]

        wc = Ne0[block]
        zc = Z0[block]
        n_macro = block.size
        Nreal = wc.sum()

        if n_macro == 0 or Nreal <= 0:
            continue

        Nreal_cells[icell] = Nreal

        b_vec = (
            rng.normal(size=n_harmonics) + 1j * rng.normal(size=n_harmonics)
        ) / np.sqrt(2.0 * Nreal)
        b_targets[icell, :] = b_vec

        rho = 1.0 + 2.0 * np.real(np.sum(b_vec[:, None] * exp_table, axis=0))
        rho = np.maximum(rho, min_pdf)

        pdf = rho / np.sum(rho)
        cdf_edges = np.concatenate(([0.0], np.cumsum(pdf)))
        cdf_edges[-1] = 1.0

        perm = rng.permutation(n_macro)
        idx_perm = block[perm]
        w_perm = wc[perm]
        z_perm = zc[perm]

        z_cell_start = z_reference + icell * cell_length

        # quiet phase in this wavelength cell
        th0_perm = 2.0 * np.pi * (z_perm - z_cell_start) / cell_length
        th0_perm = np.mod(th0_perm, 2.0 * np.pi)

        # weighted quiet-start quantiles
        cumw = np.cumsum(w_perm)
        u = (cumw - 0.5 * w_perm) / Nreal
        u = np.mod(u + rng.random(), 1.0)

        sort_u = np.argsort(u)
        theta_sorted = np.interp(u[sort_u], cdf_edges, theta_edges)
        theta_sorted = np.mod(theta_sorted, 2.0 * np.pi)

        theta_perm = np.empty_like(theta_sorted)
        theta_perm[sort_u] = theta_sorted

        # Optional weighted mean preservation in periodic phase sense
        if keep_local_mean:
            dth_perm = np.angle(np.exp(1j * (theta_perm - th0_perm)))
            dth_mean = np.average(dth_perm, weights=w_perm)
            theta_perm = np.mod(theta_perm - dth_mean, 2.0 * np.pi)

        if max_dtheta is not None:
            dth_perm = np.angle(np.exp(1j * (theta_perm - th0_perm)))
            dth_perm = np.clip(dth_perm, -abs(max_dtheta), abs(max_dtheta))
            theta_perm = np.mod(th0_perm + dth_perm, 2.0 * np.pi)

        z_new_cell = z_cell_start + (cell_length / (2.0 * np.pi)) * theta_perm

        Z_new[idx_perm] = z_new_cell
        theta_old_all[idx_perm] = th0_perm
        theta_new_all[idx_perm] = theta_perm
        dtheta[idx_perm] = np.angle(np.exp(1j * (theta_perm - th0_perm)))

    info = {
        "dtheta": dtheta,
        "theta_old": theta_old_all,
        "theta_new": theta_new_all,
        "cell_idx": cell_idx,
        "b_targets": b_targets,
        "Nreal_cells": Nreal_cells,
    }
    return Z_new, info


# ----------------------------
# Weighted statistics
# ----------------------------

def _normalize_weights(w):
    w = np.asarray(w, dtype=np.float64)
    s = np.sum(w)
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return w / s


def weighted_mean_n(x, wn):
    return np.sum(wn * x)


def weighted_mean_vec_n(X, wn):
    return np.sum(X * wn[:, None], axis=0)


def weighted_var_n(x, wn, reg=0.0):
    mu = np.sum(wn * x)
    return np.sum(wn * (x - mu) ** 2) + reg


def weighted_cov_xy_n(x, y, wn):
    mx = np.sum(wn * x)
    my = np.sum(wn * y)
    return np.sum(wn * (x - mx) * (y - my))


def weighted_cov_n(X, wn, reg=0.0):
    mu = np.sum(X * wn[:, None], axis=0)
    Xc = X - mu
    C = (Xc * wn[:, None]).T @ Xc
    if reg > 0:
        C = C + reg * np.eye(C.shape[0])
    return C


def _make_psd(C, eps=1e-14):
    C = 0.5 * (C + C.T)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, eps, None)
    return (evecs * evals) @ evecs.T


def _sqrtm_psd(C, eps=1e-14):
    C = _make_psd(C, eps=eps)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, eps, None)
    return (evecs * np.sqrt(evals)) @ evecs.T


def _invsqrtm_psd(C, eps=1e-14):
    C = _make_psd(C, eps=eps)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, eps, None)
    return (evecs * (1.0 / np.sqrt(evals))) @ evecs.T


# ----------------------------
# Covariance rematch
# ----------------------------

def _rematch_one_pair_fixed_q(q_new, p0, wn, q_src, p_src, ws, reg=1e-14):
    mq_new = weighted_mean_n(q_new, wn)
    qnc = q_new - mq_new
    var_q_new = weighted_var_n(q_new, wn, reg=reg)

    mp_t = weighted_mean_n(p_src, ws)
    var_p_t = weighted_var_n(p_src, ws, reg=reg)
    cov_qp_t = weighted_cov_xy_n(q_src, p_src, ws)

    beta_t = cov_qp_t / var_q_new

    mp0 = weighted_mean_n(p0, wn)
    cov_qp_0 = weighted_cov_xy_n(q_new, p0, wn)
    beta_0 = cov_qp_0 / var_q_new

    r0 = p0 - mp0 - beta_0 * qnc
    var_r0 = weighted_var_n(r0, wn, reg=reg)

    var_r_t = var_p_t - beta_t ** 2 * var_q_new
    if var_r_t < reg:
        var_r_t = reg

    alpha = np.sqrt(var_r_t / var_r0)
    p_new = mp_t + beta_t * qnc + alpha * r0
    return p_new


def _rematch_full_fixed_q(q_new, p0, wn, q_src, p_src, ws, reg=1e-14):
    mu_q_new = weighted_mean_vec_n(q_new, wn)
    Qn = q_new - mu_q_new
    Sqq_new = weighted_cov_n(q_new, wn, reg=reg)

    mu_p_t = weighted_mean_vec_n(p_src, ws)
    mu_q_t = weighted_mean_vec_n(q_src, ws)
    Qt = q_src - mu_q_t
    Pt = p_src - mu_p_t

    Sqp_t = (Qt * ws[:, None]).T @ Pt
    Spp_t = weighted_cov_n(p_src, ws, reg=reg)

    B_t = np.linalg.solve(Sqq_new, Sqp_t)

    mu_p0 = weighted_mean_vec_n(p0, wn)
    P0c = p0 - mu_p0
    Sqp_0 = (Qn * wn[:, None]).T @ P0c
    B_0 = np.linalg.solve(Sqq_new, Sqp_0)

    R0 = p0 - mu_p0 - Qn @ B_0
    Srr_0 = weighted_cov_n(R0, wn, reg=reg)

    Srr_t = Spp_t - B_t.T @ Sqq_new @ B_t
    Srr_t = _make_psd(Srr_t, eps=reg)

    A = _sqrtm_psd(Srr_t, eps=reg) @ _invsqrtm_psd(Srr_0, eps=reg)
    R_new = R0 @ A.T

    p_new = mu_p_t + Qn @ B_t + R_new
    return p_new


def group_indices_by_id(ids):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return {}

    order = np.argsort(ids, kind='mergesort')
    ids_sorted = ids[order]
    unique_ids, starts, counts = np.unique(ids_sorted, return_index=True, return_counts=True)

    groups = {}
    for sid, s0, cnt in zip(unique_ids, starts, counts):
        groups[int(sid)] = order[s0:s0 + cnt]
    return groups


def covariance_rematch_interpolated_momenta(
    x_new, y_new, z_new,
    px0, py0, pz0,
    w_new,
    x_src, y_src, z_src,
    px_src, py_src, pz_src,
    w_src,
    slice_id_new,
    slice_id_src,
    mode="pairwise",
    reg=1e-14,
    min_particles_new=8,
    min_particles_src=8,
):
    if mode not in ("pairwise", "full"):
        raise ValueError("mode must be 'pairwise' or 'full'")

    px_new = px0.copy()
    py_new = py0.copy()
    pz_new = pz0.copy()

    new_groups = group_indices_by_id(slice_id_new)
    src_groups = group_indices_by_id(slice_id_src)

    if len(new_groups) == 0 or len(src_groups) == 0:
        return px_new, py_new, pz_new

    common_slices = np.intersect1d(
        np.fromiter(new_groups.keys(), dtype=np.int64),
        np.fromiter(src_groups.keys(), dtype=np.int64),
    )

    for sid in common_slices:
        idxn = new_groups[int(sid)]
        idxs = src_groups[int(sid)]

        if idxn.size < min_particles_new or idxs.size < min_particles_src:
            continue

        xn = x_new[idxn]
        yn = y_new[idxn]
        zn = z_new[idxn]
        p0x = px0[idxn]
        p0y = py0[idxn]
        p0z = pz0[idxn]
        wn = np.asarray(w_new[idxn], dtype=np.float64)

        xs = x_src[idxs]
        ys = y_src[idxs]
        zs = z_src[idxs]
        psx = px_src[idxs]
        psy = py_src[idxs]
        psz = pz_src[idxs]
        ws = np.asarray(w_src[idxs], dtype=np.float64)

        sum_wn = wn.sum()
        sum_ws = ws.sum()
        if sum_wn <= 0 or sum_ws <= 0:
            continue

        wn /= sum_wn
        ws /= sum_ws

        if (
            np.var(xn) < 1e-30 or np.var(yn) < 1e-30 or np.var(zn) < 1e-30 or
            np.var(xs) < 1e-30 or np.var(ys) < 1e-30 or np.var(zs) < 1e-30
        ):
            continue

        if mode == "pairwise":
            px_new[idxn] = _rematch_one_pair_fixed_q(xn, p0x, wn, xs, psx, ws, reg=reg)
            py_new[idxn] = _rematch_one_pair_fixed_q(yn, p0y, wn, ys, psy, ws, reg=reg)
            pz_new[idxn] = _rematch_one_pair_fixed_q(zn, p0z, wn, zs, psz, ws, reg=reg)
        else:
            q_new = np.column_stack([xn, yn, zn])
            p0 = np.column_stack([p0x, p0y, p0z])
            q_src = np.column_stack([xs, ys, zs])
            p_src = np.column_stack([psx, psy, psz])

            p_rem = _rematch_full_fixed_q(q_new, p0, wn, q_src, p_src, ws, reg=reg)
            px_new[idxn] = p_rem[:, 0]
            py_new[idxn] = p_rem[:, 1]
            pz_new[idxn] = p_rem[:, 2]

    return px_new, py_new, pz_new


# ----------------------------
# Output writer
# ----------------------------

def write_sliced_h5(
    filename,
    x, y, theta, npx, npy, gamma, weight,
    e_ch, c, StepZ, Pi, n_cpu,
    n_part,
    slice_prefix="slice",
    start_index=1,
):
    x = np.asarray(x)
    y = np.asarray(y)
    theta = np.asarray(theta)
    npx = np.asarray(npx)
    npy = np.asarray(npy)
    gamma = np.asarray(gamma)
    weight = np.asarray(weight)

    n_total = x.size
    for arr, name in [
        (y, "y"), (theta, "theta"), (npx, "npx"),
        (npy, "npy"), (gamma, "gamma"), (weight, "weight")
    ]:
        if arr.size != n_total:
            raise ValueError(f"{name} has size {arr.size}, but x has size {n_total}")

    if n_total % n_part != 0:
        raise ValueError(f"Total length {n_total} is not divisible by n_part={n_part}")

    n_slices = n_total // n_part

    X = x.reshape(n_slices, n_part)
    Y = y.reshape(n_slices, n_part)
    THETA = theta.reshape(n_slices, n_part)
    NPX = npx.reshape(n_slices, n_part)
    NPY = npy.reshape(n_slices, n_part)
    G = gamma.reshape(n_slices, n_part)
    WEIGHT = weight.reshape(n_slices, n_part)

    current = np.sum(WEIGHT, axis=1) * e_ch / StepZ * c
    extra_slices = (-n_slices) % n_cpu
    main_input['time']['slen'] = f'{StepZ * (n_slices + extra_slices)}'

    with h5py.File(filename, "w") as f:
        f.attrs["n_slices"] = n_slices
        f.attrs["n_part"] = n_part
        f.create_dataset("one4one", data=np.array([0]))
        f.create_dataset("refposition", data=np.array([0]))
        f.create_dataset("slicecount", data=np.array([n_slices + extra_slices]))
        f.create_dataset("slicelength", data=np.array([StepZ]))
        f.create_dataset("slicespacing", data=np.array([StepZ]))
        f.create_dataset("beamletsize", data=np.array([4]))

        for i in range(n_slices):
            g = f.create_group(f"{slice_prefix}{i + start_index:06d}")
            g.create_dataset("current", data=np.array([current[i]]))
            g.create_dataset("x", data=X[i])
            g.create_dataset("y", data=Y[i])
            g.create_dataset("theta", data=THETA[i] - (2 * i + 1) * Pi)
            g.create_dataset("px", data=NPX[i])
            g.create_dataset("py", data=NPY[i])
            g.create_dataset("gamma", data=G[i])

        if extra_slices > 0:
            for i in np.arange(extra_slices) + n_slices:
                g = f.create_group(f"{slice_prefix}{i + start_index:06d}")
                g.create_dataset("current", data=np.array([current[-1]]))
                g.create_dataset("x", data=X[-1])
                g.create_dataset("y", data=Y[-1])
                g.create_dataset("theta", data=THETA[-1] - (2 * n_slices - 1) * Pi)
                g.create_dataset("px", data=NPX[-1])
                g.create_dataset("py", data=NPY[-1])
                g.create_dataset("gamma", data=G[-1])


# ----------------------------
# Optional diagnostics
# ----------------------------

def measure_bunching_by_lambda_block(theta, weight, n_part, SlicesMultiplyFactor, m=1):
    """
    Diagnostic helper:
    compute bunching harmonic b_m over wavelength-sized blocks
    after final GENESIS ordering.
    """
    theta = np.asarray(theta, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    n_total = theta.size
    if n_total % n_part != 0:
        raise ValueError("theta.size must be divisible by n_part")

    n_slices = n_total // n_part
    theta_s = theta.reshape(n_slices, n_part)
    w_s = weight.reshape(n_slices, n_part)

    n_blocks = n_slices // SlicesMultiplyFactor
    if n_blocks == 0:
        return np.zeros(0, dtype=np.complex128)

    vals = np.empty(n_blocks, dtype=np.complex128)
    for ib in range(n_blocks):
        sl0 = ib * SlicesMultiplyFactor
        sl1 = (ib + 1) * SlicesMultiplyFactor
        th = theta_s[sl0:sl1].ravel()
        ww = w_s[sl0:sl1].ravel()
        sw = np.sum(ww)
        if sw <= 0:
            vals[ib] = 0.0j
        else:
            vals[ib] = np.sum(ww * np.exp(-1j * m * th / SlicesMultiplyFactor)) / sw
    return vals


# ----------------------------
# Main
# ----------------------------

if __name__ == '__main__':

    import PARAMS_JDF

    try:
        lambda_u = PARAMS_JDF.lambda_u
    except Exception:
        lambda_u = 0.03

    try:
        n_cpu = PARAMS_JDF.n_cpu
    except Exception:
        n_cpu = 8

    try:
        avg_sigr = PARAMS_JDF.AvgBeamSize
    except Exception:
        avg_sigr = 1e-5

    try:
        a_u = PARAMS_JDF.a_u
    except Exception:
        a_u = 1.0121809

    try:
        SlicesMultiplyFactor = PARAMS_JDF.SlicesMultiplyFactor
    except Exception:
        SlicesMultiplyFactor = 10

    try:
        Num_Of_Slice_Particles = PARAMS_JDF.NumOfSliceParticles
    except Exception:
        Num_Of_Slice_Particles = 800

    try:
        S_factor = PARAMS_JDF.BeamStretchFactor
    except Exception:
        S_factor = 0.0

    try:
        OUT_DIR = PARAMS_JDF.OUT_DIR
    except Exception:
        OUT_DIR = 'run_test'
        
    if len(sys.argv) == 2:
        file_name_in = sys.argv[1]
    elif getattr(PARAMS_JDF, "RAWFile", None) not in (None, ""):
        file_name_in = PARAMS_JDF.RAWFile
    else:
        print("Please specify source beam file !!!")
        print("Usage: JDF_NLIST <Input File Name>")
        sys.exit(1)
    
    print("Processing file:", file_name_in)

    try:
        nseeds = PARAMS_JDF.NumOfSeeds
    except Exception:
        nseeds = 1
    try:
        iseed = PARAMS_JDF.StartOfSeeds
    except Exception:
        iseed = 0

    
    iout = read_h5_file(file_name_in)

    Pi = np.pi
    c = 299792458.0
    m = 9.11e-31
    e_ch = 1.602e-19

    mask = (iout['s'] >= -999) & (iout['s'] <= 999)
    mA_X = iout['x'][mask]
    mA_Y = iout['y'][mask]
    mA_Z = iout['s'][mask]
    mA_Z = mA_Z - np.min(mA_Z)
    mA_DXDZ = iout['dxdz'][mask]
    mA_DYDZ = iout['dydz'][mask]
    mA_GAMMA = iout['gamma'][mask]
    mA_DSDZ = np.sqrt(1 - 1 / (mA_GAMMA ** 2) - mA_DXDZ ** 2 - mA_DYDZ ** 2)
    mA_WGHT = iout['weight'][mask]

    mA_PX = mA_GAMMA * mA_DXDZ * (m * c)
    mA_PY = mA_GAMMA * mA_DYDZ * (m * c)
    mA_PZ = mA_GAMMA * mA_DSDZ * (m * c)

    p_tot = np.sqrt(mA_PX ** 2 + mA_PY ** 2 + mA_PZ ** 2)
    gamma = np.sqrt(1 + (p_tot / (m * c)) ** 2)
    gamma_0 = 11130  # keep your current choice

    lambda_r = (lambda_u / (2 * gamma_0 ** 2)) * (1 + a_u ** 2)

    mask_XLAMD = (mA_Z <= np.max(mA_Z) // lambda_r * lambda_r)
    mA_X = mA_X[mask_XLAMD]
    mA_Y = mA_Y[mask_XLAMD]
    mA_Z = mA_Z[mask_XLAMD]
    mA_DXDZ = mA_DXDZ[mask_XLAMD]
    mA_DYDZ = mA_DYDZ[mask_XLAMD]
    mA_GAMMA = mA_GAMMA[mask_XLAMD]
    mA_DSDZ = mA_DSDZ[mask_XLAMD]
    mA_WGHT = mA_WGHT[mask_XLAMD]
    mA_PX = mA_PX[mask_XLAMD]
    mA_PY = mA_PY[mask_XLAMD]
    mA_PZ = mA_PZ[mask_XLAMD]

    size_x = np.max(mA_X) - np.min(mA_X)
    size_y = np.max(mA_Y) - np.min(mA_Y)
    size_z = np.max(mA_Z) - np.min(mA_Z)
    minz, maxz = np.min(mA_Z), np.max(mA_Z)

    print('Size of sample X,Y,THETA = ', size_x, size_y, size_z)

    binnumber_Z = int(size_z // lambda_r)
    binnumber_Z = max(binnumber_Z, 2)

    # Quiet longitudinal placement
    z_hlt = 0.5 - HaltonRandomNumber(2, Num_Of_Slice_Particles)

    print('User defined parameters:')
    print('Saved in:', OUT_DIR)
    print('lambda_u = ', lambda_u)
    print('a_u = ', a_u)
    print('lambda_r = ', lambda_r)
    print('Slices per wavelength = ', SlicesMultiplyFactor)
    print('Current / Density sampling in THETA =', binnumber_Z)
    print('Stretching factor in THETA = ', S_factor)

    NumberOfSlices = int(
        SlicesMultiplyFactor * (
            (max(mA_Z) + S_factor * size_z) - (min(mA_Z) - S_factor * size_z)
        ) / lambda_r
    ) + 1

    Hz, edges_Z = np.histogram(
        mA_Z,
        bins=binnumber_Z,
        density=False,
        weights=mA_WGHT,
        range=(min(mA_Z) - S_factor * size_z, max(mA_Z) + S_factor * size_z)
    )
    Hz = ndimage.gaussian_filter(Hz, 2.0)

    Non_Zero_Z = float(np.count_nonzero(Hz))
    x0_Z = np.linspace(
        0.5 * (edges_Z[0] + edges_Z[1]),
        0.5 * (edges_Z[binnumber_Z] + edges_Z[binnumber_Z - 1]),
        binnumber_Z
    )
    y0_Z = Hz
    f_Z = interpolate.PchipInterpolator(x0_Z, y0_Z)

    StepZ = lambda_r / SlicesMultiplyFactor
    slice_centers = np.arange(
        minz + 0.5 * StepZ,
        NumberOfSlices * StepZ - 0.5 * StepZ + 0.1 * StepZ,
        StepZ
    )

    # Source quiet-slice IDs
    slice_id_src = np.floor((mA_Z - minz) / StepZ).astype(np.int64)
    slice_src_indices = group_indices_by_id(slice_id_src)

    # Global fallback means
    w_all = np.clip(mA_WGHT, 0.0, None)
    global_weight_sum = np.sum(w_all)
    if global_weight_sum > 0:
        global_x_mean = np.sum(mA_X * w_all) / global_weight_sum
        global_y_mean = np.sum(mA_Y * w_all) / global_weight_sum
        global_px_mean = np.sum(mA_PX * w_all) / global_weight_sum
        global_py_mean = np.sum(mA_PY * w_all) / global_weight_sum
        global_pz_mean = np.sum(mA_PZ * w_all) / global_weight_sum
    else:
        global_x_mean = 0.0
        global_y_mean = 0.0
        global_px_mean = 0.0
        global_py_mean = 0.0
        global_pz_mean = 0.0

    main_input = parse_config_file("genesis4.in")
    lattice_input = parse_beamline_file("genesis4.lat")
    lattice_input['UND']['params']['aw'] = f'{a_u}'
    main_input['setup']['lambda0'] = f'{StepZ}'
    main_input['setup']['gamma0'] = f'{gamma_0}'
    main_input['setup']['npart'] = f'{int(Num_Of_Slice_Particles)}'

    print('Executing main loop.')
    start = time.time()

    for rnd_idx in np.arange(iseed, iseed+nseeds, 1):

        rng = np.random.default_rng(rnd_idx)

        # preserve parameter name and logic style
        u_sel = rng.random(Num_Of_Slice_Particles)
        u_xy = HaltonRandomNumber(2, Num_Of_Slice_Particles)
        RandomHaltonSequence = np.column_stack((u_sel, u_xy))

        slice_list = []
        SliceCal_start = time.time()

        for slice_number in range(NumberOfSlices):
            ZZZ = slice_centers[slice_number]
            NoOfElec = (Non_Zero_Z) * (f_Z(ZZZ) / NumberOfSlices) / Num_Of_Slice_Particles
            if NoOfElec > 0:
                slice_list.append(slice_number)

        n_workers = min(MAX_WORKERS_CAP, os.cpu_count() or 1, max(1, len(slice_list)))
        chunksize = max(1, len(slice_list) // max(1, 4 * n_workers))

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            result2 = list(ex.map(_slice_worker, slice_list, chunksize=chunksize))

        SliceCal_end = time.time()

        print('Starting rearranging array...')
        all_data = np.concatenate(result2, axis=0)

        Full_X0 = all_data[:, 0]
        Full_Y0 = all_data[:, 1]
        Full_Z0 = all_data[:, 2]
        Full_Ne0 = all_data[:, 3]
        Full_PX0 = all_data[:, 4]
        Full_PY0 = all_data[:, 5]
        Full_PZ0 = all_data[:, 6]

        Rearrange_end = time.time()

        main_input['sponrad']['seed'] = int(rnd_idx)
        main_input['importbeam']['file'] = f'../../BeamInputs/seed_{int(rnd_idx)}.h5'
        outdir = OUT_DIR + f"/Planar_AW{a_u}/{rnd_idx}"
        beamdir = OUT_DIR + f"/Planar_AW{a_u}/BeamInputs"
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(beamdir, exist_ok=True)

        print('Adding noise...')
        rng = np.random.default_rng(rnd_idx)
        Full_Z_noise, noise_info = add_shot_noise_to_quiet_z(
            Z0=Full_Z0,
            Ne0=Full_Ne0,
            cell_length=lambda_r,
            rng=rng,
            keep_local_mean=True,
            max_dtheta=None,
            n_harmonics=4,
            n_grid=1024,
            z_reference=minz,
        )

        AddNoise_end = time.time()

        # IMPORTANT PRESERVED LOGIC:
        # keep particles attached to ORIGINAL quiet GENESIS slices
        print('Covariance Rematch...')
        slice_id_new = np.floor((Full_Z0 - minz) / StepZ).astype(np.int64)

        # Use quiet z in rematch so longitudinal noise only affects phase loading
        Full_PX, Full_PY, Full_PZ = covariance_rematch_interpolated_momenta(
            x_new=Full_X0,
            y_new=Full_Y0,
            z_new=Full_Z0,
            px0=Full_PX0,
            py0=Full_PY0,
            pz0=Full_PZ0,
            w_new=Full_Ne0,
            x_src=mA_X,
            y_src=mA_Y,
            z_src=mA_Z,
            px_src=mA_PX,
            py_src=mA_PY,
            pz_src=mA_PZ,
            w_src=mA_WGHT,
            slice_id_new=slice_id_new,
            slice_id_src=slice_id_src,
            mode="full",
            reg=1e-14
        )

        CovRematch_end = time.time()

        # IMPORTANT PRESERVED LOGIC:
        # final ordering follows quiet Full_Z0
        sort_idx = np.argsort(Full_Z0, kind='mergesort')

        Full_X = Full_X0[sort_idx]
        Full_Y = Full_Y0[sort_idx]
        Full_Z_quiet = Full_Z0[sort_idx]
        Full_Z = Full_Z_noise[sort_idx]   # kept for diagnostics / compatibility
        Full_PX = Full_PX[sort_idx]
        Full_PY = Full_PY[sort_idx]
        Full_PZ = Full_PZ[sort_idx]
        Full_Ne = Full_Ne0[sort_idx]

        Full_NPX = Full_PX / (m * c)
        Full_NPY = Full_PY / (m * c)
        Full_NPZ = Full_PZ / (m * c)
        Full_GAMMA = np.sqrt(1 + Full_NPX ** 2 + Full_NPY ** 2 + Full_NPZ ** 2)

        # GENESIS-friendly final phase:
        # quiet slice phase + wavelength-cell microscopic phase perturbation
        Full_dtheta = noise_info["dtheta"][sort_idx]
        Full_THETA_quiet = Full_Z_quiet * 2.0 * Pi / StepZ
        Full_THETA = Full_THETA_quiet + SlicesMultiplyFactor * Full_dtheta

        write_sliced_h5(
            f"{beamdir}/seed_{int(rnd_idx)}.h5",
            Full_X, Full_Y, Full_THETA, Full_NPX, Full_NPY, Full_GAMMA, Full_Ne,
            e_ch=e_ch, c=c, StepZ=StepZ, Pi=Pi, n_cpu=n_cpu,
            n_part=Num_Of_Slice_Particles
        )

        with open(outdir + "/genesis4.in", "w") as f:
            f.write(config_to_string(main_input))

        with open(outdir + "/genesis4.lat", "w") as f:
            f.write(beamline_to_string(lattice_input))

        # optional startup diagnostic
        try:
            b1 = measure_bunching_by_lambda_block(
                Full_THETA, Full_Ne, Num_Of_Slice_Particles, SlicesMultiplyFactor, m=1
            )
            if b1.size > 0:
                print("Measured <|b1|^2> =", np.mean(np.abs(b1) ** 2))
        except Exception as exc:
            print("Bunching diagnostic skipped:", exc)

        Written_end = time.time()

        print('Time of Slice Calculation: ', SliceCal_end - SliceCal_start)
        print('Time of Rearrange: ', Rearrange_end - SliceCal_end)
        print('Time of Add Noise: ', AddNoise_end - Rearrange_end)
        print('Time of Rematch: ', CovRematch_end - AddNoise_end)
        print('Time of Written: ', Written_end - CovRematch_end)
        print(f'Seed {rnd_idx} written into GENESIS Beam format')

    end = time.time()
    print('Time of work: ', end - start)