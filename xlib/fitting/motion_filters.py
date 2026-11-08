import torch
import numpy as np

def interpolate_gated_frames(track_data: torch.Tensor, accept_mask: torch.Tensor, is_rotation: bool = False) -> torch.Tensor:
    """
    Fills in rejected frames (accept_mask == False) using linear interpolation
    (or SLERP for quaternions) from the nearest accepted neighbors.
    track_data: [T, ...] tensor.
    accept_mask: [T] boolean tensor.
    """
    T = track_data.shape[0]
    out = track_data.clone()
    
    # Find accepted indices
    accepted_indices = torch.where(accept_mask)[0]
    if len(accepted_indices) == 0:
        return out
    
    # For every rejected index, find previous and next accepted index
    rejected_indices = torch.where(~accept_mask)[0]
    
    for idx in rejected_indices:
        idx_val = idx.item()
        
        # Find prev
        prev_idx = None
        next_idx = None
        
        # Using binary search or simple search
        # Since T is small (~8000), simple search is fine
        prev_mask = accepted_indices < idx_val
        next_mask = accepted_indices > idx_val
        
        if prev_mask.any():
            prev_idx = accepted_indices[prev_mask][-1].item()
        if next_mask.any():
            next_idx = accepted_indices[next_mask][0].item()
            
        if prev_idx is None and next_idx is None:
            continue
        elif prev_idx is None:
            out[idx] = out[next_idx]
        elif next_idx is None:
            out[idx] = out[prev_idx]
        else:
            # Interpolate
            alpha = (idx_val - prev_idx) / (next_idx - prev_idx)
            if is_rotation:
                # SLERP between quaternions
                q1 = out[prev_idx]
                q2 = out[next_idx]
                
                # compute dot product
                dot = torch.sum(q1 * q2)
                if dot < 0.0:
                    q1 = -q1
                    dot = -dot
                
                DOT_THRESHOLD = 0.9995
                if dot > DOT_THRESHOLD:
                    # linear interpolation
                    q = q1 + alpha * (q2 - q1)
                    q = q / torch.norm(q)
                else:
                    theta_0 = torch.acos(dot)
                    theta = theta_0 * alpha
                    sin_theta = torch.sin(theta)
                    sin_theta_0 = torch.sin(theta_0)
                    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
                    s1 = sin_theta / sin_theta_0
                    q = (s0 * q1) + (s1 * q2)
                out[idx] = q
            else:
                out[idx] = out[prev_idx] + alpha * (out[next_idx] - out[prev_idx])
                
    return out


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        
        # Avoid division by zero
        if t_e <= 0:
            return x
            
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Per-dimension adaptive cutoff (plan S2: "filter per-dimension directly").
        # Using a global norm here would drive one shared cutoff for all 260 pose dims.
        cutoff = self.min_cutoff + self.beta * dx_hat.abs()
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

def one_euro_filter_track(track_data: torch.Tensor, min_cutoff: float = 1.0, beta: float = 0.0) -> torch.Tensor:
    """
    Applies One-Euro filter to a [T, D] tensor.
    """
    T = track_data.shape[0]
    out = torch.zeros_like(track_data)
    if T == 0:
        return out
        
    f = OneEuroFilter(t0=0.0, x0=track_data[0], min_cutoff=min_cutoff, beta=beta)
    out[0] = track_data[0]
    for t in range(1, T):
        out[t] = f(float(t), track_data[t])
        
    return out

def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3, 3] rotation matrices to [..., 4] quaternions (w, x, y, z)."""
    batch_shape = R.shape[:-2]
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    
    tr = m00 + m11 + m22
    
    q = torch.zeros(*batch_shape, 4, device=R.device, dtype=R.dtype)
    
    # tr > 0
    cond1 = tr > 0
    S1 = torch.sqrt(tr[cond1] + 1.0) * 2
    q[cond1, 0] = 0.25 * S1
    q[cond1, 1] = (m21[cond1] - m12[cond1]) / S1
    q[cond1, 2] = (m02[cond1] - m20[cond1]) / S1
    q[cond1, 3] = (m10[cond1] - m01[cond1]) / S1
    
    # m00 is max
    cond2 = (~cond1) & (m00 > m11) & (m00 > m22)
    S2 = torch.sqrt(1.0 + m00[cond2] - m11[cond2] - m22[cond2]) * 2
    q[cond2, 0] = (m21[cond2] - m12[cond2]) / S2
    q[cond2, 1] = 0.25 * S2
    q[cond2, 2] = (m01[cond2] + m10[cond2]) / S2
    q[cond2, 3] = (m02[cond2] + m20[cond2]) / S2
    
    # m11 is max
    cond3 = (~cond1) & (~cond2) & (m11 > m22)
    S3 = torch.sqrt(1.0 + m11[cond3] - m00[cond3] - m22[cond3]) * 2
    q[cond3, 0] = (m02[cond3] - m20[cond3]) / S3
    q[cond3, 1] = (m01[cond3] + m10[cond3]) / S3
    q[cond3, 2] = 0.25 * S3
    q[cond3, 3] = (m12[cond3] + m21[cond3]) / S3
    
    # m22 is max
    cond4 = (~cond1) & (~cond2) & (~cond3)
    S4 = torch.sqrt(1.0 + m22[cond4] - m00[cond4] - m11[cond4]) * 2
    q[cond4, 0] = (m10[cond4] - m01[cond4]) / S4
    q[cond4, 1] = (m02[cond4] + m20[cond4]) / S4
    q[cond4, 2] = (m12[cond4] + m21[cond4]) / S4
    q[cond4, 3] = 0.25 * S4
    
    return q

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] quaternions (w, x, y, z) to [..., 3, 3] rotation matrices."""
    q = q / torch.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.zeros(*q.shape[:-1], 3, 3, device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2*y*y - 2*z*z
    R[..., 0, 1] = 2*x*y - 2*z*w
    R[..., 0, 2] = 2*x*z + 2*y*w
    
    R[..., 1, 0] = 2*x*y + 2*z*w
    R[..., 1, 1] = 1 - 2*x*x - 2*z*z
    R[..., 1, 2] = 2*y*z - 2*x*w
    
    R[..., 2, 0] = 2*x*z - 2*y*w
    R[..., 2, 1] = 2*y*z + 2*x*w
    R[..., 2, 2] = 1 - 2*x*x - 2*y*y
    
    return R

def slerp_smooth_rotations(R_track: torch.Tensor, min_cutoff: float = 1.0, beta: float = 0.0) -> torch.Tensor:
    """
    R_track: [T, 3, 3] or [T, 9] tensor.
    Returns: smoothed [T, 3, 3] tensor.
    """
    original_shape = R_track.shape
    if R_track.ndim == 2 and R_track.shape[1] == 9:
        R_track = R_track.view(-1, 3, 3)
        
    q_track = rotmat_to_quat(R_track)

    # Resolve the double cover: q and -q are the same rotation, but a sign flip between
    # consecutive frames looks like a huge jump to the filter.
    for t in range(1, q_track.shape[0]):
        if torch.dot(q_track[t], q_track[t - 1]) < 0:
            q_track[t] = -q_track[t]

    # Filter on quaternions
    q_smoothed = one_euro_filter_track(q_track, min_cutoff, beta)
    
    # Normalize quaternions and convert back
    R_smoothed = quat_to_rotmat(q_smoothed)
    
    if len(original_shape) == 2 and original_shape[1] == 9:
        R_smoothed = R_smoothed.view(-1, 9)
        
    return R_smoothed
