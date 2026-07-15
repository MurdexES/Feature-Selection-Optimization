import pickle
import os
from datetime import datetime

class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints', save_interval=10, max_checkpoints=3):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints  # keep only the newest N files; None = keep all
        os.makedirs(checkpoint_dir, exist_ok=True)

    def should_save(self, generation):
        return (generation + 1) % self.save_interval == 0

    def save(self, generation, state: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{generation + 1:04d}_{timestamp}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Checkpoint saved: {filename}")

        self._prune_old_checkpoints()
        return filepath

    def _prune_old_checkpoints(self):
        """Keep only the newest `max_checkpoints` files so disk usage stays bounded."""
        if not self.max_checkpoints:
            return
        files = sorted(f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl'))
        for stale in files[:-self.max_checkpoints]:
            os.remove(os.path.join(self.checkpoint_dir, stale))
    
    def load_latest(self):
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')]

        if not files:
            print("No checkpoints found.")
            return None
        
        files.sort()
        latest = files[-1]
        filepath = os.path.join(self.checkpoint_dir, latest)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        print(f"Loaded checkpoint: {latest}")
        print(f"  Resuming from generation {state['generation'] + 1}")
        print(f"  Best R2 so far: {state['best_overall_fit']:.4f}")
        return state

    def load_specific(self, filename):
        """Load a specific checkpoint by filename."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print(f"Loaded checkpoint: {filename}")
        return state

    def list_checkpoints(self):
        """Show all available checkpoints."""
        files = sorted([f for f in os.listdir(self.checkpoint_dir)
                        if f.endswith('.pkl')])
        if not files:
            print("No checkpoints found.")
            return []

        print(f"\nAvailable checkpoints in '{self.checkpoint_dir}':")
        print(f"{'#':>4} {'Filename':<45} {'Size':>8}")
        print("-" * 60)

        for i, f in enumerate(files):
            size = os.path.getsize(
                os.path.join(self.checkpoint_dir, f)
            ) / 1024
            print(f"{i+1:>4} {f:<45} {size:>6.1f}KB")

        return files