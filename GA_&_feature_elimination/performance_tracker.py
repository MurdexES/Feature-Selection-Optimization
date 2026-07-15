import time
import psutil
import os

class PerformanceTracker:
    """Tracks time and RAM usage across GA generations."""

    def __init__(self):
        self.process        = psutil.Process(os.getpid())
        self.start_time     = None
        self.gen_start_time = None
        self.peak_ram_mb    = 0
        self.gen_times      = []
        self.ram_history    = []

    def start(self):
        self.start_time = time.perf_counter()
        print(f"Starting RAM: {self._current_ram():.1f} MB")

    def start_generation(self):
        self.gen_start_time = time.perf_counter()

    def end_generation(self):
        gen_time   = time.perf_counter() - self.gen_start_time
        current_ram = self._current_ram()

        self.gen_times.append(gen_time)
        self.ram_history.append(current_ram)

        if current_ram > self.peak_ram_mb:
            self.peak_ram_mb = current_ram

        return gen_time, current_ram

    def total_elapsed(self):
        return time.perf_counter() - self.start_time

    def _current_ram(self):
        """RAM used by this process in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def print_summary(self, n_generations):
        total_time = self.total_elapsed()
        avg_gen    = sum(self.gen_times) / len(self.gen_times)
        min_gen    = min(self.gen_times)
        max_gen    = max(self.gen_times)

        print("\n" + "=" * 55)
        print("PERFORMANCE SUMMARY")
        print("=" * 55)
        print(f"  Total time:          {self._format_time(total_time)}")
        print(f"  Avg time/generation: {avg_gen:.2f}s")
        print(f"  Fastest generation:  {min_gen:.2f}s")
        print(f"  Slowest generation:  {max_gen:.2f}s")
        print(f"  Peak RAM usage:      {self.peak_ram_mb:.1f} MB")
        print(f"  Final RAM usage:     {self._current_ram():.1f} MB")
        print("=" * 55)

    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs    = seconds % 60
        return f"{minutes}m {secs:.1f}s"