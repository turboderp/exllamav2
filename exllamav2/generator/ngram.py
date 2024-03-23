from __future__ import annotations

class NgramCacheNode:

    def __init__(self,
                 preload: NgramCacheNode | None = None):

        if preload is None:
            self.transitions = {}
        else:
            self.transitions = dict(preload.transitions)

    def add_transition(self, next_value):
        if next_value in self.transitions:
            self.transitions[next_value] += 1
        else:
            self.transitions[next_value] = 1

    def predict(self, threshold):
        if not self.transitions: return None
        k = max(self.transitions, key = self.transitions.get)
        if self.transitions[k] < threshold: return None
        return k


class NgramCache:

    ngrams: dict[str: NgramCacheNode]
    min_len: int
    max_len: int
    preload: NgramCache | None

    def __init__(self,
                 min_len: int,
                 max_len: int,
                 preload: NgramCache | None = None):

        self.ngrams = {}
        self.min_len = min_len
        self.max_len = max_len
        self.preload = preload

    # Ingest context to cache

    def update(self, sequence):
        for i in range(1 + self.min_len, len(sequence)):
            a = max(i - self.max_len, 0)
            b = i
            substr = sequence[a:b]
            self.update_single(substr)

    # Ingest one set of ngrams, right-aligned

    def update_single(self, sequence):
        new = sequence[-1]
        old = sequence[:-1]
        for i in range(len(old) - self.min_len + 1):
            substr = tuple(old[i:])
            v = self.ngrams.get(substr)
            if v is None and self.preload is not None:
                vp = self.preload.ngrams.get(substr)
            else:
                vp = None
            if v is None:
                v = NgramCacheNode(vp)
                self.ngrams[substr] = v
            v.add_transition(new)

    # Predict next transition from substring

    def predict_next(self, context, threshold, preload):
        for i in range(len(context) - self.min_len + 1):
            substr = tuple(context[i:])
            v = self.ngrams.get(substr)
            if v is None and preload is not None:
                v = preload.ngrams.get(substr)
            if v is not None:
                t = v.predict(threshold)
                if t is not None:
                    return t
        return None

