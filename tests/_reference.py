"""Independent reference implementation used ONLY by the test suite.

It deliberately does not import :mod:`schur_elim`. It builds its own
Laplacian from raw node/edge/port tuples and solves the full network with
its own Fraction Gaussian elimination, so expected values in the tests never
come from the code under verification.
"""

from __future__ import annotations

from fractions import Fraction


def build_laplacian(nodes, edges):
    """Independent Laplacian construction from raw tuples."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    lap = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for u, v, g in edges:
        i, j = idx[u], idx[v]
        g = Fraction(g)
        lap[i][i] += g
        lap[j][j] += g
        lap[i][j] -= g
        lap[j][i] -= g
    return lap


def components(nodes, edges):
    idx = {n: i for i, n in enumerate(nodes)}
    adj = [set() for _ in nodes]
    for u, v, _ in edges:
        i, j = idx[u], idx[v]
        adj[i].add(j)
        adj[j].add(i)
    seen = [False] * len(nodes)
    out = []
    for start in range(len(nodes)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        comp.sort()
        out.append(tuple(nodes[i] for i in comp))
    return tuple(out)


def gaussian_solve(a, b):
    """Solve A x = b exactly; A must be non-singular. Independent routine."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for k in range(n):
        r = k
        while r < n and m[r][k] == 0:
            r += 1
        if r == n:
            raise ValueError("singular system in reference solver")
        if r != k:
            m[k], m[r] = m[r], m[k]
        piv = m[k][k]
        for i in range(k + 1, n):
            if m[i][k] == 0:
                continue
            f = m[i][k] / piv
            m[i][k] = Fraction(0)
            for j in range(k + 1, n + 1):
                m[i][j] -= f * m[k][j]
    x = [Fraction(0) for _ in range(n)]
    for k in range(n - 1, -1, -1):
        s = m[k][n]
        for j in range(k + 1, n):
            s -= m[k][j] * x[j]
        x[k] = s / m[k][k]
    return x


def full_solution(nodes, edges, ports, vport):
    """Solve the complete network for prescribed port potentials.

    Floating components are gauged by pinning their smallest-index node to
    zero (with no source their whole component is then equipotential zero).
    Returns (voltages, port_currents, power_dissipated) computed entirely
    from this reference's own Laplacian and solver.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    lap = build_laplacian(nodes, edges)
    port_set = set(ports)
    comps = components(nodes, edges)

    volt = [Fraction(0) for _ in nodes]
    for p in ports:
        volt[idx[p]] = Fraction(vport[p])

    free = []
    for comp in comps:
        ci = [idx[n] for n in comp]
        comp_ports = [i for i in ci if nodes[i] in port_set]
        if comp_ports:
            free.extend(i for i in ci if i not in set(comp_ports))
        else:
            # Gauge: pin smallest node, solve the rest from their KCL rows.
            pinned = ci[0]
            volt[pinned] = Fraction(0)
            free.extend(i for i in ci[1:])

    if free:
        a = [[lap[i][j] for j in free] for i in free]
        b = [
            -sum(
                (lap[i][idx[p]] * Fraction(vport[p]) for p in ports),
                Fraction(0),
            )
            for i in free
        ]
        xf = gaussian_solve(a, b)
        for i, val in zip(free, xf):
            volt[i] = val

    currents = {}
    for p in ports:
        i = idx[p]
        currents[p] = sum(
            (lap[i][j] * volt[j] for j in range(len(nodes))), Fraction(0)
        )

    dissipated = Fraction(0)
    for u, v, g in edges:
        dv = volt[idx[u]] - volt[idx[v]]
        dissipated += Fraction(g) * dv * dv
    return volt, currents, dissipated


def reference_dtm(nodes, edges, ports):
    """Global port DtN matrix, one unit-voltage solve per port."""
    p = len(ports)
    mat = [[Fraction(0) for _ in range(p)] for _ in range(p)]
    for c, pc in enumerate(ports):
        vmap = {pk: Fraction(1 if pk == pc else 0) for pk in ports}
        _, currents, _ = full_solution(nodes, edges, ports, vmap)
        for r, pr in enumerate(ports):
            mat[r][c] = currents[pr]
    return mat


def equivalent_two_terminal(nodes, edges, ports):
    """Effective conductance between the two ports via a unit solve."""
    assert len(ports) == 2
    m = reference_dtm(nodes, edges, ports)
    return -m[0][1]
