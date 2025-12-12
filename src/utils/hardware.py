"""
Hardware detection and automatic batch size computation.
Implements Issue #41: Automatic Batch Size Determination
"""

import gc
import logging

import gymnasium as gym
import psutil
import torch

logger = logging.getLogger(__name__)


def get_available_memory(device: torch.device) -> int:
    """
    Returns available memory in bytes for the given device.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        cached = torch.cuda.memory_reserved(device)
        available = total - allocated - cached
        return available
    elif device.type == "mps":
        # MPS doesn't expose memory info directly, estimate from system RAM
        # Apple Silicon shares memory between CPU and GPU
        mem = psutil.virtual_memory()
        # Use 50% of available RAM for MPS (conservative)
        return int(mem.available * 0.5)
    else:
        # CPU - use system RAM
        mem = psutil.virtual_memory()
        return mem.available


def measure_env_memory(env_name: str, render_mode: str = "rgb_array") -> int:
    """
    Measure memory footprint of a single environment instance.
    Returns memory usage in bytes.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    mem_before = psutil.Process().memory_info().rss

    # Create single env and step it a few times to capture realistic usage
    env = gym.make(env_name, render_mode=render_mode)
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    gc.collect()
    mem_after = psutil.Process().memory_info().rss

    return max(mem_after - mem_before, 1024 * 1024)  # At least 1MB


def measure_env_scaling(env_name: str, render_mode: str = "rgb_array", max_test: int = 4) -> float:
    """
    Measure how memory scales with number of environments.
    Returns memory per environment in bytes.
    """
    import numpy as np

    gc.collect()
    measurements = []

    for n in range(1, max_test + 1):
        mem_before = int(psutil.Process().memory_info().rss)

        envs = gym.make_vec(env_name, num_envs=n, render_mode=render_mode)
        obs, _ = envs.reset()
        for _ in range(5):
            # Use numpy array for vectorized env actions
            actions = np.array([envs.single_action_space.sample() for _ in range(n)])
            obs, rewards, terminated, truncated, info = envs.step(actions)
        envs.close()

        gc.collect()
        mem_after = int(psutil.Process().memory_info().rss)
        mem_used = max(mem_after - mem_before, 0)

        if mem_used > 0:
            measurements.append(mem_used / n)
            logger.debug(
                f"  {n} envs: {mem_used / 1024 / 1024:.1f} MB ({mem_used / n / 1024 / 1024:.1f} MB/env)"
            )

    # Return average memory per env, or fallback
    if measurements:
        return sum(measurements) / len(measurements)
    return 50 * 1024 * 1024  # 50MB fallback


def compute_optimal_num_envs(
    env_name: str,
    device: torch.device,
    render_mode: str = "rgb_array",
    model_memory_estimate_mb: int = 500,
    safety_margin: float = 0.7,
    min_envs: int = 1,
    max_envs: int = 64,
    is_image_based: bool = False,
) -> int:
    """
    Automatically compute the optimal number of parallel environments.

    Args:
        env_name: Gymnasium environment name
        device: torch device (cuda, mps, cpu)
        render_mode: Environment render mode
        model_memory_estimate_mb: Estimated memory for model + training overhead (MB)
        safety_margin: Fraction of available memory to use (0.0-1.0)
        min_envs: Minimum number of environments
        max_envs: Maximum number of environments to consider

    Returns:
        Optimal number of parallel environments
    """
    logger.info("Computing optimal number of parallel environments...")

    # Step 1: Get available memory
    available_memory = get_available_memory(device)
    logger.info(f"  Available memory: {available_memory / 1024 / 1024:.0f} MB")

    # Step 2: Reserve memory for model and training
    model_reserve = model_memory_estimate_mb * 1024 * 1024
    memory_for_envs = (available_memory * safety_margin) - model_reserve
    logger.info(f"  Memory budget for envs: {memory_for_envs / 1024 / 1024:.0f} MB")

    if memory_for_envs <= 0:
        logger.warning("  Low memory! Using minimum environments.")
        return min_envs

    # Step 3: Measure environment memory scaling
    logger.info("  Profiling environment memory usage...")
    try:
        mem_per_env = measure_env_scaling(env_name, render_mode, max_test=min(4, max_envs))
    except Exception as e:
        logger.warning(f"  Memory profiling failed: {e}. Using conservative estimate.")
        mem_per_env = 100 * 1024 * 1024  # 100MB fallback

    logger.info(f"  Memory per environment: {mem_per_env / 1024 / 1024:.1f} MB")

    # Step 4: Calculate optimal count
    optimal = int(memory_for_envs / mem_per_env)
    optimal = max(min_envs, min(optimal, max_envs))

    # Step 4.5: Limit for image-based envs on CPU (compute-bound, not memory-bound)
    if is_image_based and device.type == "cpu":
        cpu_limit = 8  # Image processing on CPU is slow
        if optimal > cpu_limit:
            logger.info(f"  Limiting to {cpu_limit} envs for image-based training on CPU")
            optimal = cpu_limit

    # Step 5: Prefer power of 2 for better GPU utilization
    powers_of_2 = [2**i for i in range(7) if min_envs <= 2**i <= max_envs]
    if powers_of_2:
        # Find largest power of 2 <= optimal
        optimal = max([p for p in powers_of_2 if p <= optimal], default=optimal)

    logger.info(f"  Optimal num_envs: {optimal}")
    return optimal
