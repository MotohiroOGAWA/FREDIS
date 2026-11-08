"""Insert training batches when a reaction pattern has been absent too long."""

from collections import Counter
import random


class RarePatternReplayGroups:
    def __init__(self, groups, interval, pattern_ids=(), pattern_ids_for_group=None,
                 state=None, rng=None):
        if type(interval) is not int or interval < 0:
            raise ValueError('rare_pattern_max_gap must be a non-negative integer')
        self.groups = list(groups)
        self.interval = interval
        self.rng = rng or random
        self.lookup = pattern_ids_for_group
        self.state = state if state is not None else {}
        self.state.setdefault('step', 0)
        self.state.setdefault('last_seen', {int(p): 0 for p in pattern_ids})
        self.state.setdefault('unavailable', set())
        self.available = {}
        self.replayed_patterns = Counter()
        self.replayed_batches = 0
        self._current = None

    def __len__(self):
        return len(self.groups) + self.replayed_batches

    def observe(self, data):
        if not self.interval:
            return
        patterns = set(data.reactions.reaction_type[:, 0].tolist())
        self.available[tuple(self._current)] = patterns
        self.state['step'] += 1
        for pattern in patterns:
            self.state['last_seen'][pattern] = self.state['step']

    def candidate(self, pattern):
        known = [group for group, patterns in self.available.items() if pattern in patterns]
        if known:
            return self.rng.choice(known)
        # Read stored CPU registries only when a missing pattern needs a batch.
        # This neither runs chemistry nor preloads the complete dataset.
        for group in self.groups:
            key = tuple(group)
            if key not in self.available and self.lookup is not None:
                self.available[key] = set(self.lookup(group))
            if pattern in self.available.get(key, ()):
                return group
        self.state['unavailable'].add(pattern)
        return None

    def __iter__(self):
        for group in self.groups:
            if self.interval:
                overdue = [p for p, step in self.state['last_seen'].items()
                           if self.state['step'] - step >= self.interval
                           and p not in self.state['unavailable']]
                overdue.sort(key=lambda p: (self.state['last_seen'][p], p))
                for pattern in overdue:
                    forced = self.candidate(pattern)
                    if forced is None:
                        continue
                    self._current = forced
                    self.replayed_batches += 1
                    self.replayed_patterns[pattern] += 1
                    yield forced
                    break  # Keep the ordinary full pass moving even with many patterns.
            self._current = group
            yield group
