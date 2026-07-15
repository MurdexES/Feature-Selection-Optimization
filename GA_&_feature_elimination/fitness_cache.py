import hashlib
import json

class FitnessCache:
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def make_key(self, individual):
        return hashlib.md5(json.dumps(individual).encode()).hexdigest()
    
    def get(self, individual):
        key = self.make_key(individual)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, individual, score):
        key = self.make_key(individual)
        self.cache[key] = score

    def __len__(self):
        return len(self.cache)
    
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self):
        print(f"Cache size: {len(self)} unique individuals evaluated")
        print(f"Cache hits: {self.hits} ({self.hit_rate():.1%} of evalutions avoided)")
        print(f"Cache misses: {self.misses}")