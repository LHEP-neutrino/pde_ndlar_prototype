#!/usr/bin/env python3
import glob
import os
import csv
import yaml
import numpy as np
import pandas as pd

import h5flow
from sklearn.cluster import DBSCAN


# =========================
# Output directories
# =========================
PLOT_DIR = "pde_plots"
SAVE_DIR = "pde_files"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# Geometry helpers
# =========================
def extract_tpc_bounds(file_obj):
    mod_bounds = np.array(file_obj["geometry_info"].attrs["module_RO_bounds"])
    max_drift = file_obj["geometry_info"].attrs["max_drift_distance"]
    tpc_bounds = []
    for mod in mod_bounds:
        x_min, y_min, z_min = mod[0]
        x_max, y_max, z_max = mod[1]
        # Two TPC boundaries per module:
        tpc_bounds.append(((x_max - max_drift, y_min, z_min), (x_max, y_max, z_max)))
        tpc_bounds.append(((x_min, y_min, z_min), (x_min + max_drift, y_max, z_max)))
    return np.array(tpc_bounds)


def flipYZ(coords):
    coords = np.array(coords)
    if coords.ndim == 1:
        x, y, z = coords
        return np.array([x, z, y])
    return coords[:, [0, 2, 1]]


def get_plate_corners(det_id, tpc_shift, geom_dict):
    # ACLs are every modulo 4 -> shape_key convention you used
    shape_key = 0 if (det_id % 4) == 0 else 1 # ==3 for FSD!
    offs_min = np.array(geom_dict["geom"][shape_key]["min"], float)
    offs_max = np.array(geom_dict["geom"][shape_key]["max"], float)

    xmin, ymin, _ = offs_min
    xmax, ymax, _ = offs_max
    local_rect = np.array(
        [[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0]]
    )

    det_ctr_local = np.array(geom_dict["det_center"][det_id], float)
    return local_rect + det_ctr_local + tpc_shift


# =========================
# Physics / math helpers
# =========================
def solid_angle_rectangle(x_s, y_s, z_s, x0, x1, y0, y1, z_det):
    z = np.abs(z_det - z_s)
    omega = 0.0
    for i, xi in enumerate([x0, x1]):
        for j, yj in enumerate([y0, y1]):
            sign = (-1) ** (i + j)
            dx = xi - x_s
            dy = yj - y_s
            r = np.sqrt(dx**2 + dy**2 + z**2)
            if z != 0:
                omega += sign * np.arctan((dx * dy) / (z * r))
    return omega


def line_fit_3d_segment_midpoints(x, y, z, seg_length):
    if len(x) < 2:
        return [], [], [], []

    pts = np.column_stack((x, y, z))
    centroid = pts.mean(axis=0)

    _, _, vt = np.linalg.svd(pts - centroid)
    direction = vt[0]
    direction /= np.linalg.norm(direction)

    dx = pts[:, 0] - centroid[0]
    dy = pts[:, 1] - centroid[1]
    dz = pts[:, 2] - centroid[2]
    t = dx * direction[0] + dy * direction[1] + dz * direction[2]

    t_min, t_max = t.min(), t.max()
    total_length = t_max - t_min
    if total_length <= 0:
        raise ValueError("All points are identical or degenerate.")

    t_edges = np.arange(t_min, t_max, seg_length)
    if t_edges[-1] < t_max:
        t_edges = np.append(t_edges, t_max)

    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:])

    x_mid = centroid[0] + t_mid * direction[0]
    y_mid = centroid[1] + t_mid * direction[1]
    z_mid = centroid[2] + t_mid * direction[2]

    step_length = np.diff(t_edges)
    return x_mid, y_mid, z_mid, step_length


def compute_light_fraction_from_line_midpoints(
    x_mid, y_mid, z_mid, step_length, x0, x1, y0, y1, z_det
):
    """
    Compute expected light for a rectangular detector from a pre-defined
    straight track, given by segment midpoints and lengths, using:
        N_gamma = Y_gamma * dEdx_muon * L * exp(-d/3000)

    where d is the 3D distance from the segment midpoint to the center of the
    lighttrap rectangle (at z_det).

    Returns
    -------
    total_detected, total_produced, fraction_detected
    """
    # Constants
    Y_gamma = 24000  # photons per MeV at 500 V/cm
    dEdx_muon = 2.1  # MeV/cm, MIP energy loss in LAr
    att_len = 3000.0  # cm

    total_produced = 0.0
    total_detected = 0.0

    # center of the light trap rectangle (at plane z_det)
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    zc = z_det

    for xm, ym, zm, L in zip(x_mid, y_mid, z_mid, step_length):
        if L <= 0:
            continue

        # distance from emission point to trap center
        d = np.sqrt((xm - xc) ** 2 + (ym - yc) ** 2 + (zm - zc) ** 2)

        # photons produced in this segment with exponential reduction
        N_gamma = Y_gamma * dEdx_muon * L * np.exp(-d / att_len)
        total_produced += N_gamma

        # solid angle from this segment to the rectangle
        omega = solid_angle_rectangle(xm, ym, zm, x0, x1, y0, y1, z_det)
        if omega < 0:
            omega = 0.0

        frac = omega / (4 * np.pi)
        total_detected += frac * N_gamma

    fraction = total_detected / total_produced if total_produced > 0 else 0.0
    return total_detected, total_produced, fraction


def min_range_baseline(array, segment_size=15, num_segments=40, num_means=4):
    indices = np.arange(num_segments + 1) * segment_size
    start_indices = indices[:-1]

    segment_range = np.arange(segment_size)
    index_array = start_indices[:, None] + segment_range

    sliced_data = array[..., index_array]

    ranges = np.abs(np.ptp(sliced_data, axis=-1))
    means = np.mean(sliced_data, axis=-1)

    mask_zero = ranges != 0
    ranges = np.where(mask_zero, ranges, np.nan)
    means = np.where(mask_zero, means, np.nan)

    smallest_ordering = np.argsort(ranges, axis=-1)
    sorted_means = np.take_along_axis(means, smallest_ordering, axis=-1)

    average_mean = np.mean(sorted_means[..., 1:num_means], axis=-1)
    rms = np.sqrt(
        np.mean(
            np.square(
                np.take_along_axis(ranges, smallest_ordering[..., :num_means], axis=-1)
            ),
            axis=-1,
        )
    )
    return average_mean, rms


def ransac_line_3d(
    x, y, z, distance_threshold=1.0, min_inliers=10, max_trials=1000, random_state=None
):
    rng = np.random.default_rng(random_state)
    pts = np.column_stack((x, y, z))
    N = pts.shape[0]
    if N < 2:
        raise ValueError("Not enough points for RANSAC line fit")

    best_inliers = None
    best_count = 0

    for _ in range(max_trials):
        i1, i2 = rng.choice(N, size=2, replace=False)
        p0 = pts[i1]
        p1 = pts[i2]
        v = p1 - p0
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        v /= norm

        diff = pts - p0
        proj = diff @ v
        closest = np.outer(proj, v)
        dist = np.linalg.norm(diff - closest, axis=1)

        inliers = dist < distance_threshold
        count = np.count_nonzero(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers
            if best_count >= max(min_inliers, 0.9 * N):
                break

    if best_inliers is None or best_count < max(2, min_inliers):
        best_inliers = np.ones(N, dtype=bool)

    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid)
    direction = vt[0]
    direction /= np.linalg.norm(direction)

    return centroid, direction, best_inliers


def line_midpoints_from_model(centroid, direction, pts, seg_length):
    diff = pts - centroid
    t = diff @ direction
    t_min, t_max = t.min(), t.max()

    t_edges = np.arange(t_min, t_max, seg_length)
    if t_edges[-1] < t_max:
        t_edges = np.append(t_edges, t_max)

    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:])
    step_len = np.diff(t_edges)

    mid = centroid + np.outer(t_mid, direction)
    return mid[:, 0], mid[:, 1], mid[:, 2], step_len


# =========================
# TPC processing (from global line)
# =========================
def process_tpc_from_line(
    tpc_idx,
    x_mid,
    y_mid,
    z_mid,
    step_length,
    tpc_bounds,
    geom_data,
    det_positions_local,
):
    bounds = tpc_bounds[tpc_idx]
    lower, upper = np.array(bounds[0]), np.array(bounds[1])

    mask_geom = (
        (x_mid >= lower[0])
        & (x_mid <= upper[0])
        & (y_mid >= lower[1])
        & (y_mid <= upper[1])
        & (z_mid >= lower[2])
        & (z_mid <= upper[2])
    )

    mask_zsafe = (z_mid > lower[2]) & (z_mid < upper[2])
    mask_tpc = mask_geom & mask_zsafe

    if not np.any(mask_tpc):
        x_t = np.array([])
        y_t = np.array([])
        z_t = np.array([])
        L_t = np.array([])
    else:
        x_t = x_mid[mask_tpc]
        y_t = y_mid[mask_tpc]
        z_t = z_mid[mask_tpc]
        L_t = step_length[mask_tpc]

    tpc_shift = np.mean(tpc_bounds[tpc_idx], axis=0)

    det_rects = []
    for det_id in range(det_positions_local.shape[0]):
        corners = get_plate_corners(det_id, tpc_shift, geom_data)
        cf = flipYZ(corners)
        det_rects.append((cf[0][0], cf[0][2], cf[2][0], cf[2][2]))

    (x0_b, y0_b, z0_b), (x1_b, y1_b, z1_b) = tpc_bounds[tpc_idx]
    my_detector_z = [z0_b, z1_b]

    tolerance = 0.05
    unique_dets = []
    for det in det_rects:
        if not any(np.allclose(det, ud, atol=tolerance) for ud in unique_dets):
            unique_dets.append(det)

    results = []
    for z_det in my_detector_z:
        for x0, y0, x1, y1 in unique_dets:
            det, prod, frac = compute_light_fraction_from_line_midpoints(
                x_t, y_t, z_t, L_t, x0, x1, y0, y1, z_det
            )
            results.append((x0, y0, x1, y1, z_det, det, prod, frac))

    return results


def process_tpc_from_line_LTcrossing(
    tpc_idx,
    x_mid,
    y_mid,
    z_mid,
    step_length,
    tpc_bounds,
    geom_data,
    det_positions_local,
    z_safe=65,
    z_cross_tol=5,
):
    bounds = tpc_bounds[tpc_idx]
    lower, upper = np.array(bounds[0]), np.array(bounds[1])

    mask_geom = (
        (x_mid >= lower[0])
        & (x_mid <= upper[0])
        & (y_mid >= lower[1])
        & (y_mid <= upper[1])
        & (z_mid >= lower[2])
        & (z_mid <= upper[2])
    )
    mask_zsafe = (z_mid > lower[2]) & (z_mid < upper[2])
    mask_tpc = mask_geom & mask_zsafe

    if not np.any(mask_tpc):
        x_t = np.array([])
        y_t = np.array([])
        z_t = np.array([])
        L_t = np.array([])
    else:
        x_t = x_mid[mask_tpc]
        y_t = y_mid[mask_tpc]
        z_t = z_mid[mask_tpc]
        L_t = step_length[mask_tpc]

    tpc_shift = np.mean(tpc_bounds[tpc_idx], axis=0)

    det_rects = []
    for det_id in range(det_positions_local.shape[0]):
        corners = get_plate_corners(det_id, tpc_shift, geom_data)
        cf = flipYZ(corners)
        det_rects.append((cf[0][0], cf[0][2], cf[2][0], cf[2][2]))

    (x0_b, y0_b, z0_b), (x1_b, y1_b, z1_b) = tpc_bounds[tpc_idx]
    my_detector_z = [z0_b, z1_b]

    tolerance = 0.05
    unique_dets = []
    for det in det_rects:
        if not any(np.allclose(det, ud, atol=tolerance) for ud in unique_dets):
            unique_dets.append(det)

    results = []
    for z_det in my_detector_z:
        for x0, y0, x1, y1 in unique_dets:
            crosses_mask = (
                (x_t >= x0)
                & (x_t <= x1)
                & (y_t >= y0)
                & (y_t <= y1)
                & (np.abs(z_t - z_det) <= z_cross_tol)
            )
            if np.any(crosses_mask):
                det = np.nan
                prod = np.nan
                frac = np.nan
            else:
                det, prod, frac = compute_light_fraction_from_line_midpoints(
                    x_t, y_t, z_t, L_t, x0, x1, y0, y1, z_det
                )

            results.append((x0, y0, x1, y1, z_det, det, prod, frac))

    return results


# =========================
# Worker: process ONE (csv,hdf5) pair
# =========================
def worker_process_file(
    *,
    csv_file: str,
    hdf5_path: str,
    hdf5_name: str,
    geom_data: dict,
    tpc_bounds: np.ndarray,
    det_positions_local: np.ndarray,
):
    """
    Process a single CSV (muon selections) + matching HDF5 file.
    Writes one output CSV: ./pde_files/pde_<hdf5_name>.csv
    """

    df_mu = pd.read_csv(csv_file)
    p_events = df_mu["ev_id"].astype(int).tolist()

    # Global accumulators (initialized after first event)
    PE_meas_tot = None
    PE_meas_noLT_tot = None
    PE_exp_tot = None
    PE_exp_noLT_tot = None

    det_chan = geom_data["det_chan"]  # {tpc: {trap: [channels...]}}
    tpc_list = [0, 1, 2, 3, 4, 5, 6, 7]

    ordered_pairs = [
        (int(tpc), int(trap))
        for tpc in tpc_list
        for trap in sorted(det_chan[tpc].keys(), key=int)
    ]
    trap_index = {pair: i for i, pair in enumerate(ordered_pairs)}
    trap_summaries_flat = [
        (tpc * 40 + trap, tpc, trap, list(map(int, det_chan[tpc][trap])))
        for tpc, trap in ordered_pairs
    ]
    n_traps = len(ordered_pairs)

    with h5flow.data.H5FlowDataManager(hdf5_path, "r") as f:
        for event in p_events:
            # ---------------------------------
            # Global muon hits for this event
            # ---------------------------------
            my_muon_hits = f["charge/events", "charge/calib_prompt_hits", event]
            x = my_muon_hits["x"].reshape(-1)
            y = my_muon_hits["y"].reshape(-1)
            z = my_muon_hits["z"].reshape(-1)

            coords = np.column_stack((x, y, z))

            # ---------------------------------
            # Global DBSCAN: keep only largest cluster
            # ---------------------------------
            db = DBSCAN(eps=15.0, min_samples=3)
            labels = db.fit_predict(coords)

            mask_non_noise = labels != -1
            if np.any(mask_non_noise):
                labels_nn = labels[mask_non_noise]
                uniq, counts = np.unique(labels_nn, return_counts=True)
                main_label = uniq[np.argmax(counts)]
                cluster_mask = labels == main_label
            else:
                cluster_mask = np.zeros_like(labels, dtype=bool)

            # ---------------------------------
            # RANSAC line on the main cluster (fallback PCA)
            # ---------------------------------
            seg_length = 0.1  # cm
            if np.sum(cluster_mask) >= 2:
                x_c = x[cluster_mask]
                y_c = y[cluster_mask]
                z_c = z[cluster_mask]

                centroid, direction, _inliers_r = ransac_line_3d(
                    x_c,
                    y_c,
                    z_c,
                    distance_threshold=1.0,
                    min_inliers=5,
                    max_trials=500,
                    random_state=42,
                )
                pts_c = np.column_stack((x_c, y_c, z_c))
                x_mid, y_mid, z_mid, step_length = line_midpoints_from_model(
                    centroid, direction, pts_c, seg_length
                )
            else:
                x_mid, y_mid, z_mid, step_length = line_fit_3d_segment_midpoints(
                    x, y, z, seg_length
                )

            # ---------------------------------
            # Expected light per trap (this event)
            # ---------------------------------
            all_results_normal = []
            all_results_noLT = []

            for tpc_idx in range(len(tpc_bounds)):
                res_norm = process_tpc_from_line(
                    tpc_idx,
                    x_mid,
                    y_mid,
                    z_mid,
                    step_length,
                    tpc_bounds,
                    geom_data,
                    det_positions_local,
                )
                res_noLT = process_tpc_from_line_LTcrossing(
                    tpc_idx,
                    x_mid,
                    y_mid,
                    z_mid,
                    step_length,
                    tpc_bounds,
                    geom_data,
                    det_positions_local,
                )
                all_results_normal.extend(res_norm)
                all_results_noLT.extend(res_noLT)

            detected_all_normal = np.array(
                [r[5] for r in all_results_normal], dtype=float
            )
            detected_all_noLT = np.array([r[5] for r in all_results_noLT], dtype=float)
            
            if event % 4917 == 0:
                plot_event_example(detected_all_normal, all_results_normal, 1, event, x_mid, y_mid, z_mid, "expected PE")
                plot_event_example(detected_all_noLT, all_results_noLT, 1, event, x_mid, y_mid, z_mid, "expected PE no LT crossing")

            # ---------------------------------
            # Measured PE sums per trap (this event)
            # ---------------------------------
            sum_hits = f["charge/events", "light/events", "light/sum_hits", event]

            light_trap_pe_sums = np.zeros(n_traps, dtype=float)

            # Take first entry (bug in flow triples all entries?)
            tpcs = sum_hits["tpc"][0, 0]
            traps = sum_hits["det"][0, 0]
            sums = sum_hits["sum"][0, 0]

            for tpc, trap, pe_sum in zip(tpcs, traps, sums):
                idx = trap_index[(int(tpc), int(trap))]
                light_trap_pe_sums[idx] = float(pe_sum)

            pe_meas_evt = np.asarray(light_trap_pe_sums, dtype=float)
            pe_exp_evt = np.asarray(detected_all_normal, dtype=float)
            pe_exp_noLT_evt_raw = np.asarray(detected_all_noLT, dtype=float)
            
            # ---------------------------------
            # Initialize global accumulators once
            # ---------------------------------
            if PE_meas_tot is None:
                n_traps = len(pe_meas_evt)
                PE_meas_tot = np.zeros(n_traps, dtype=float)
                PE_meas_noLT_tot = np.zeros(n_traps, dtype=float)
                PE_exp_tot = np.zeros(n_traps, dtype=float)
                PE_exp_noLT_tot = np.zeros(n_traps, dtype=float)

            # Defensive NaN cleanup
            pe_meas_evt = np.nan_to_num(pe_meas_evt, nan=0.0, posinf=0.0, neginf=0.0)
            pe_exp_evt = np.nan_to_num(pe_exp_evt, nan=0.0, posinf=0.0, neginf=0.0)

            # no-LT logic
            mask_noLT = np.isfinite(pe_exp_noLT_evt_raw) & (pe_exp_noLT_evt_raw > 0)

            pe_exp_noLT_evt = np.where(mask_noLT, pe_exp_noLT_evt_raw, 0.0)
            pe_meas_noLT_evt = np.where(mask_noLT, pe_meas_evt, 0.0)

            if event % 4917 == 0:
                plot_event_example(pe_meas_evt, all_results_normal, 1, event, x, y, z, "measured PE")
                plot_event_example(pe_meas_noLT_evt, all_results_noLT, 1, event, x, y, z, "measured PE no LT crossing")

            # Accumulate over events
            PE_meas_tot += pe_meas_evt
            PE_exp_tot += pe_exp_evt
            PE_exp_noLT_tot += pe_exp_noLT_evt
            PE_meas_noLT_tot += pe_meas_noLT_evt

    # Final totals
    PE_meas = PE_meas_tot
    PE_meas_noLTcr = PE_meas_noLT_tot
    PE_exp = PE_exp_tot
    PE_exp_noLTcr = PE_exp_noLT_tot

    # Write CSV
    output_file = os.path.join(SAVE_DIR, f"pde_{hdf5_name}.csv")

    titles = [
        "file_name",
        "global_trap_id",
        "tpc",
        "trap",
        "ch_list",
        "PE_meas",
        "PE_meas_noLTcr",
        "PE_exp",
        "PE_exp_noLTcr",
    ]

    with open(output_file, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(titles)

        for entry, meas, meas_noLT, exp_, exp_noLT in zip(
            trap_summaries_flat, PE_meas, PE_meas_noLTcr, PE_exp, PE_exp_noLTcr
        ):
            global_trap_id = entry[0]
            tpc = entry[1]
            trap = entry[2]
            ch_list = entry[3]

            writer.writerow(
                [
                    hdf5_name,
                    global_trap_id,
                    tpc,
                    trap,
                    ch_list,
                    meas,
                    meas_noLT,
                    exp_,
                    exp_noLT,
                ]
            )

    print(f"[rank worker] Wrote {len(PE_meas)} traps to '{output_file}'.")

# ===================
# Plot event display
# ===================
    
# def plot_event_example(pde_avg, all_results, max_events, event, x, y, z, title):
#     import matplotlib.pyplot as plt
#     from matplotlib.cm import ScalarMappable
#     from matplotlib.colors import Normalize
#     from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#     import numpy as np

#     # Normalize or set to zero if all zero
#     pde_percent = np.zeros_like(pde_avg)
#     if np.any(pde_avg > 0):
#         pde_percent = pde_avg/np.max(pde_avg) * 100.0

#     # Set up colormap and normalization range
#     vmin = np.min(pde_percent)
#     vmax = np.max(pde_percent)
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     cmap = plt.cm.ocean_r

#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot rectangles
#     for result, value in zip(all_results, pde_percent):
#         x0, y0, x1, y1, z_det, *_ = result
#         corners = [
#             [x0, z_det, y0],
#             [x1, z_det, y0],
#             [x1, z_det, y1],
#             [x0, z_det, y1]
#         ]
#         color = cmap(norm(value))  # Map actual PDE % through normalized colormap
#         rect = Poly3DCollection([corners], color=color, alpha=0.5)
#         ax.add_collection3d(rect)
#     ax.plot(x, z, y, color='b', marker='o', markersize=2, ls='none', alpha=0.8, label='Muon track')
#     # Colorbar setup — use the same normalization!
#     sm = ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])  # Just to make matplotlib happy
#     cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
#     cbar.set_label('Normalized intensity (%)')

#     # Axes setup
#     ax.set_xlabel('X (cm)')
#     ax.set_ylabel('Z (cm)')
#     ax.set_zlabel('Y (cm)')
#     ax.set_xlim(-70, 70)
#     ax.set_ylim(-70, 70)
#     ax.set_zlim(-70, 70)
#     ax.set_title(f'Light per light trap (normalized to 1)')

#     plt.savefig(os.path.join("pde_plots/", f'pde_{title}_{event}.png'))
#     plt.close()
def plot_event_example(pde_avg, all_results, max_events, event, x, y, z, title):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import os

    # -----------------------------
    # Normalize (same logic)
    # -----------------------------
    pde_percent = np.zeros_like(pde_avg, dtype=float)
    if np.any(pde_avg > 0):
        pde_percent = pde_avg / np.max(pde_avg) * 100.0

    vmin = float(np.min(pde_percent))
    vmax = float(np.max(pde_percent))

    # -----------------------------
    # Build rectangle meshes
    # -----------------------------
    meshes = []

    for result, value in zip(all_results, pde_percent):
        x0, y0, x1, y1, z_det, *_ = result

        # Rectangle corners (same orientation as matplotlib)
        xs = [x0, x1, x1, x0]
        ys = [z_det, z_det, z_det, z_det]
        zs = [y0, y0, y1, y1]

        # Two triangles per rectangle
        i = [0, 0]
        j = [1, 2]
        k = [2, 3]

        meshes.append(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=i,
                j=j,
                k=k,
                intensity=[value] * 4,
                colorscale="cividis",
                cmin=vmin,
                cmax=vmax,
                opacity=0.5,
                showscale=False
            )
        )

    # -----------------------------
    # Muon track
    # -----------------------------
    track = go.Scatter3d(
        x=x,
        y=z,
        z=y,
        mode="markers",
        marker=dict(
            size=3,
            color="blue",
            opacity=0.8
        ),
        name="Muon track"
    )

    # -----------------------------
    # Dummy trace for colorbar
    # -----------------------------
    colorbar_trace = go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode="markers",
        marker=dict(
            size=0.1,
            color=[vmin, vmax],
            colorscale="cividis",
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title="Normalized intensity (%)"
            ),
        ),
        showlegend=False
    )

    # -----------------------------
    # Figure
    # -----------------------------
    fig = go.Figure(data=[*meshes, track, colorbar_trace])

    fig.update_layout(
        title="Light per light trap (normalized to 1)",
        scene=dict(
            xaxis=dict(title="X (cm)", range=[-70, 70]),
            yaxis=dict(title="Z (cm)", range=[-70, 70]),
            zaxis=dict(title="Y (cm)", range=[-70, 70]),
            aspectmode="cube",
        ),
        width=900,
        height=800,
        showlegend=True
    )

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs("pde_plots", exist_ok=True)
    outfile = os.path.join("pde_plots", f"pde_{title}_{event}.png")
    # pio.get_chrome()
    pio.write_image(fig, outfile)

# =========================
# Main: rank → files → geometry → worker
# =========================
# def main():
#     # ---- SLURM rank ----
#     try:
#         rank = int(os.environ["SLURM_PROCID"])
#     except KeyError:
#         print("SLURM_PROCID not set, defaulting to 1")
#         rank = 1

#     # ---- Input locations ----
#     directory_selected_muons = "/global/cfs/cdirs/dune/users/wermelinger/FSD/PDE_study/FSD_display_muons_v7/"
#     mu_file_pattern = "*.csv"
#     file_list_muons = sorted(glob.glob(os.path.join(directory_selected_muons, mu_file_pattern)))

#     directory1 = "/global/cfs/cdirs/dune/www/data/FSD/reflows/v7/flow/cosmics/07Nov2024/"
#     directory2 = "/global/cfs/cdirs/dune/www/data/FSD/reflows/v7/flow/cosmics/08Nov2024/"
#     file_pattern = "*.hdf5"
#     file_list1 = sorted(glob.glob(os.path.join(directory1, file_pattern)))
#     file_list2 = sorted(glob.glob(os.path.join(directory2, file_pattern)))
#     file_list = file_list1 + file_list2

#     if rank < 0 or rank >= len(file_list_muons):
#         print(f"Rank {rank}: no CSV to process (len(file_list_muons)={len(file_list_muons)}), exiting.")
#         return

#     # Fast lookup: basename -> full path
#     hdf5_map = {os.path.basename(path): path for path in file_list}

#     csv_file = file_list_muons[rank]
#     csv_basename = os.path.basename(csv_file)
#     hdf5_name = csv_basename.replace(".csv", "")
#     hdf5_path = hdf5_map.get(hdf5_name)

#     if hdf5_path is None:
#         print(f"Rank {rank}: WARNING: No matching HDF5 file found for {hdf5_name}. Exiting.")
#         return

#     print(f"[rank {rank}] CSV : {csv_file}")
#     print(f"[rank {rank}] HDF5: {hdf5_path}")

#     # ---- Load geometry (once per rank) ----
#     geom_yaml = "/global/cfs/cdirs/dune/users/kunzmann/FSD/flow_james/ndlar_flow/data/fsd_flow/light_module_desc-6.0.1.yaml"
#     with open(geom_yaml, "r") as geom_f:
#         geom_data = yaml.safe_load(geom_f)

#     det_positions_list = []
#     for det_id in sorted(geom_data["det_center"].keys()):
#         x, y, z = geom_data["det_center"][det_id]
#         det_positions_list.append([x, y, z])
#     det_positions_local = np.array(det_positions_list)

#     # Take an example hdf5 file to extract TPC bounds
#     sample_hdf5_file = "/global/cfs/cdirs/dune/www/data/FSD/reflows/v6/flow/cosmics/07Nov2024/packet-0020113-2024_11_07_19_44_23_CET.FLOW.hdf5"
#     with h5flow.data.H5FlowDataManager(sample_hdf5_file, "r") as file_obj:
#         tpc_bounds = extract_tpc_bounds(file_obj)

#     # ---- Call worker for this rank's file ----
#     worker_process_file(
#         csv_file=csv_file,
#         hdf5_path=hdf5_path,
#         hdf5_name=hdf5_name,
#         geom_data=geom_data,
#         tpc_bounds=tpc_bounds,
#         det_positions_local=det_positions_local,
#     )


def main():
    # ---- SLURM rank ----
    try:
        rank = int(os.environ["SLURM_PROCID"])
    except Exception:
        print("SLURM_PROCID not set (or invalid), defaulting to 0")
        rank = 0

    # ---- Input locations ----
    directory_selected_muons = "/global/cfs/cdirs/dune/users/wermelinger/2x2/PDE_study/run1_2x2_muons_first100/"
    mu_file_pattern = "*.csv"
    file_list_muons = sorted(
        glob.glob(os.path.join(directory_selected_muons, mu_file_pattern))
    )

    directory1 = "/global/cfs/cdirs/dune/www/data/2x2/reflows/v11/flow/beam/july8_2024/nominal_hv/"
    # directory2 = "/global/cfs/cdirs/dune/www/data/FSD/reflows/v7/flow/cosmics/08Nov2024/"
    file_pattern = "*.FLOW.hdf5"
    file_list1 = sorted(glob.glob(os.path.join(directory1, file_pattern)))
    # file_list2 = sorted(glob.glob(os.path.join(directory2, file_pattern)))
    file_list = file_list1[0:100]  # + file_list2

    # ---- Guard: extra ranks do nothing ----
    if rank < 0 or rank >= len(file_list_muons):
        print(
            f"Rank {rank}: no CSV to process (len(file_list_muons)={len(file_list_muons)}), exiting."
        )
        return
    
    # ---- Match CSV -> HDF5 ----
    hdf5_map = {os.path.basename(path): path for path in file_list}

    csv_file = file_list_muons[rank]
    csv_basename = os.path.basename(csv_file)
    hdf5_name = csv_basename.replace(".csv", "")
    hdf5_path = hdf5_map.get(hdf5_name)

    if hdf5_path is None:
        print(
            f"Rank {rank}: WARNING: No matching HDF5 file found for {hdf5_name}. Exiting."
        )
        return

    print(f"[rank {rank}] CSV : {csv_file}")
    print(f"[rank {rank}] HDF5: {hdf5_path}")

    # ---- Protected block: geometry, bounds, worker ----
    try:
        # Geometry
        geom_yaml = "/global/homes/m/mnuland/backtracking/light_module_desc-5.0.0.yaml"
        with open(geom_yaml, "r") as geom_f:
            geom_data = yaml.safe_load(geom_f)

        det_positions_list = []
        for det_id in sorted(geom_data["det_center"].keys()):
            x, y, z = geom_data["det_center"][det_id]
            det_positions_list.append([x, y, z])
        det_positions_local = np.array(det_positions_list)

        # Extract TPC bounds
        sample_hdf5_file = "/global/cfs/cdirs/dune/www/data/2x2/sandbox/v11/flow/packet-0050017-2024_07_09_01_14_40_CDT.FLOW.hdf5"
        with h5flow.data.H5FlowDataManager(sample_hdf5_file, "r") as file_obj:
            tpc_bounds = extract_tpc_bounds(file_obj)

        # Worker
        worker_process_file(
            csv_file=csv_file,
            hdf5_path=hdf5_path,
            hdf5_name=hdf5_name,
            geom_data=geom_data,
            tpc_bounds=tpc_bounds,
            det_positions_local=det_positions_local,
        )

    except Exception as e:
        print(f"[rank {rank}] ERROR processing {csv_basename}: {e}")
        # traceback.print_exc()
        return


if __name__ == "__main__":
    main()
