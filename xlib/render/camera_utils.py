"""Camera utilities."""
import numpy as np
import torch
import math
from typing import Tuple, Union


def dynamic_orbit_angles(n_orbits: int, n_steps_per_orbit: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample parameters for dynamic orbits.
    
    Args:
        n_orbits: Number of different orbits to sample
        n_steps_per_orbit: Number of camera positions per orbit
        
    Returns:
        elevs: Array of elevation angles for each orbit [n_orbits, n_steps_per_orbit]
        azims: Array of azimuth angles for each orbit [n_orbits, n_steps_per_orbit]
    """
    # Initialize elevation angles
    elevs = np.zeros((n_orbits, n_steps_per_orbit))
    azims = np.zeros((n_orbits, n_steps_per_orbit))
    
    # Sample number of sinusoids for each orbit (1-3)
    n_sinusoids = np.array([4]*n_orbits) #np.random.randint(1, 4, size=n_orbits)
    t = np.repeat(np.linspace(0, 1, n_steps_per_orbit)[None], n_orbits, axis=0)
    
    # For each orbit, create the elevation pattern
    for i in range(n_orbits):
        # Create vectorized sinusoids for this orbit
        periods = np.arange(1, n_sinusoids[i] + 1)
        amplitudes = np.random.uniform(10, 30, size=n_sinusoids[i]) * np.pi / 180
        amplitudes *= 2.0 ** np.arange(0, -4, -1)

        # we let both the phases and azimuth to be a bit random from the center
        phases = np.random.uniform(-np.pi / 12, np.pi / 12, size=n_sinusoids[i])
        azim_0 = np.random.uniform(-np.pi / 12, np.pi / 12, (1,))
        # we let the phases to stochastically add a pi
        random_signs = np.random.choice([-1, 1], size=n_sinusoids[i])
        periods = random_signs * periods

        # Compute all sinusoids at once and sum them
        period_matrix = np.repeat(2 * np.pi * periods[:, None], n_steps_per_orbit, axis=1)
        phase_matrix = np.repeat(phases[:, np.newaxis], n_steps_per_orbit, axis=1)
        amplitude_matrix = np.repeat(amplitudes[:, np.newaxis], n_steps_per_orbit, axis=1)
        
        # Calculate sinusoids and sum them
        sinusoids = amplitude_matrix * np.sin(t * period_matrix + phase_matrix)
        elevs[i] = np.sum(sinusoids, axis=0)

        # Clamp elevation angles to be between -89 and 89 degrees (in radians)
        max_angle = np.pi / 2 - 0.05
        elevs[i] = np.clip(elevs[i], -max_angle, max_angle)

        delta_elevs = np.roll(elevs[i], -1) - elevs[i]
        max_de = np.max(np.abs(delta_elevs)) + 0.01# + np.pi * 2 / n_steps_per_orbit
        delta_azims = max_de - np.abs(delta_elevs)
        delta_azims = delta_azims / delta_azims.sum() * np.pi * 2

        # we randomly decide the rotation direction
        if np.random.rand() > 0.5:
            delta_azims = delta_azims[::-1]
        azims[i] = np.cumsum(np.concatenate([azim_0, delta_azims]))[:-1]
    #print(azims)
    return azims, elevs


def horizontal_turntable_angles(
        n_steps: int = 120,
        n_rot: int = 1,
        elev_low: float = -math.pi/4,
        elev_high: float = math.pi/4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the elevation and azimus angle of 360 rotation.
    
    Args:
        n_steps: Number of steps for the turntable rotation
        n_rot: Number of rotations to complete
        elev_low: Lower bound for elevation angle in radians
        elev_high: Upper bound for elevation angle in radians
        
    Returns:
        tuple: (azims, elevs) where azims and elevs are numpy arrays of azimuth and elevation angles
    """
    half_steps = math.ceil(n_steps / 2)
    rot_steps = n_steps // n_rot
    elevs = np.linspace(elev_low, elev_high, half_steps)
    elevs = np.concatenate([elevs, elevs[::-1]]) if n_steps % 2 == 0 \
        else np.concatenate([elevs[:-1], elevs[::-1]])
    elevs = np.concatenate([elevs[half_steps//2:], elevs[:half_steps//2]])
    azims = np.linspace(0, 2 * np.pi, rot_steps)
    return torch.from_numpy(azims).float(), torch.from_numpy(elevs).float()


def frontal_nearview_angles(
        n_steps: int = 120,
        n_rot: int = 3,
        azim_low: float = -math.pi/4,
        azim_high: float = math.pi/4,
        elev_low: float = -math.pi/4,
        elev_high: float = math.pi/4) -> Tuple[np.ndarray, np.ndarray]:
    """Return the elevation and azimus angle of 360 rotation.
    """
    rot_steps = n_steps // n_rot // 2
    extra_steps = n_steps - rot_steps * n_rot * 2
    elevs = np.linspace(elev_low, elev_high, rot_steps)
    elevs = np.concatenate([elevs, elevs[::-1]]) # low -> high -> low
    # shift to 0 -> high -> low -> 0
    elevs = np.concatenate([elevs[rot_steps//2:], elevs[:rot_steps//2], elevs[:extra_steps]])

    azims = np.linspace(azim_low, azim_high, rot_steps)
    azims = np.concatenate([azims, azims[::-1], azims[:extra_steps]])

    return azims, elevs


def fibonacci_sphere_angles(n_frame: int = 120) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the elevation and azimus angle of 360 rotation.
    """

    indices = torch.arange(n_frame * 2)
    elev_fibonacci = torch.pi / 2 - torch.arccos(1 - indices / n_frame) 
    azim_fibonacci = torch.pi * (1 + 5**0.5) * indices

    # remove a small room around the vertical line
    valid_mask = (elev_fibonacci < torch.pi / 2 - torch.pi / 6) & (elev_fibonacci > - torch.pi / 2 + torch.pi / 6)

    return azim_fibonacci[valid_mask], elev_fibonacci[valid_mask]


def fibonacci_camera_with_noise(
    look_at_base: torch.Tensor,
    radius_min: float = 1.1, 
    radius_max: float = 2.1, 
    fov_max: float = 35.0,
    n_views: int = 60,
    noise_std: float = 0.04) -> tuple[torch.Tensor, float, float]:
    """Sample camera with uniform distribution of radius and fov and uniformly on the sphere using fibonacci sequence.
    """

    radius = np.random.uniform(radius_min, radius_max)
    fov = 2 * math.atan2(radius_max * math.tan(math.radians(fov_max / 2)), radius) * 180 / math.pi

    # first sample 360 views
    azims, elevs = fibonacci_sphere_angles(n_views)
    azims += torch.randn_like(azims) * noise_std 
    elevs += torch.randn_like(elevs) * noise_std

    cam_poses = campos_from_angle(azims, elevs, radius)
    cam_poses = look_at_base[None] + cam_poses.to(look_at_base)
    return cam_poses, radius, fov


def campos_from_angle(
        azim: torch.Tensor,
        elev: torch.Tensor,
        radius: Union[torch.Tensor, float]) -> torch.Tensor:
    """Create point from angles and radius.
        Kaolin coordinate system. (X -> Right, Y -> Up, Z -> Back).
        azim=0 elev=0 -> Z.
    Args:
        azim: (N,) or scalar, azimuthal angle in radians
        elev: (N,) or scalar, polar angle in radians
        radius: (N,) or scalar, radius in meters
    Returns:
        (N, 3), camera position in meters
    """

    cos_elev = torch.cos(elev)
    x = cos_elev * torch.sin(azim)
    z = cos_elev * torch.cos(azim)
    y = torch.sin(elev)

    if isinstance(radius, torch.Tensor):
        return radius.unsqueeze(-1) * torch.stack([x, y, z], -1)
    else:
        return radius * torch.stack([x, y, z], -1)
