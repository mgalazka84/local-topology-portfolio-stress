"""O(n K + L) cut-reroot comparator; K <= n-1 active deleted edges.

Preparation groups candidates by cut in O(n+L). Each query reroots the two
components once for each active cut, then evaluates all its crossing edges.
The implementation uses no cached dense component response matrices.
"""
import numpy as np


class CutReroot:
    def __init__(self, tree):
        self.tree = tree
        grouped = [[] for _ in tree.edges]
        for path in tree.paths:
            for k in np.flatnonzero(path.valid):
                grouped[int(path.forward[k]) // 2].append(path.endpoints)
        self.cuts = [(k, np.array(pairs, dtype=int))
                     for k, pairs in enumerate(grouped) if pairs]

    def query(self, weights, return_all=False):
        tr = self.tree
        a = tr.sigma * np.asarray(weights)
        msg, total = tr.messages(a)
        nominal = float(a @ total)
        best, swap = nominal, None
        records = [(None, nominal)] if return_all else None
        for cut, pairs in self.cuts:
            # Two forest passes. Skipping the cut produces two rooted trees.
            forest = np.zeros_like(msg)
            for i in reversed(tr.order[1:]):
                p = tr.parent[i]
                out = tr.directed[i, p]
                if out // 2 == cut:
                    continue
                forest[out] = a[i] + sum(r * forest[inc]
                    for j, r, out2, inc in tr.adj[i]
                    if j != p and out2 // 2 != cut)
            response = np.zeros(tr.n)
            for i in tr.order:
                response[i] = a[i] + sum(r * forest[inc]
                    for _, r, out, inc in tr.adj[i] if out // 2 != cut)
                for j, r, out, inc in tr.adj[i]:
                    if out // 2 != cut and tr.parent[j] == i and j != 0:
                        forest[out] = response[i] - r * forest[inc]
            u, v = pairs.T
            p, q = tr.edges[cut]
            old = tr.rho[p, q] * msg[2*cut] * msg[2*cut+1]
            risks = nominal + 2 * (tr.rho[u, v] * response[u] * response[v] - old)
            k = int(np.argmax(risks))
            if risks[k] > best:
                best = float(risks[k])
                swap = (tr.edges[cut], tuple(map(int, pairs[k])))
            if return_all:
                records.extend(((tr.edges[cut], tuple(map(int, pair))), float(risk))
                               for pair, risk in zip(pairs, risks))
        return (best, swap, records) if return_all else (best, swap)
