from runners.parallel_runner import ParallelRunner


class DelayedParallelRunner(ParallelRunner):
    """Parallel delayed rollouts using the shared delay-aware worker and batch loop."""
