#!/usr/bin/env python3
"""Fixed-input demonstration for the schur_elim library.

Run with ``python demo.py`` (Python 3.14, standard library only). It performs
real exact computations on a deterministic network:

  1. builds a resistor network with an internal bridge, a port-less
     component and an isolated node;
  2. reduces it to the port Dirichlet-to-Neumann matrix and compares the
     reduced answer against an independent full-graph Gaussian solve;
  3. checks symmetry, zero row sums and exact power conservation;
  4. actually triggers (and prints) the zero-resistance rejection boundary.

Nothing is pre-recorded, no network is used and the run takes well under
8 seconds.
"""

from __future__ import annotations

import time
from fractions import Fraction

from schur_elim import (
    InfiniteConductanceError,
    ResistorNetwork,
    ZeroConductanceError,
    reduce_network,
)


def _full_gaussian_solve(net, prescribed):
    """Independent full-graph reference used only inside this demo.

    Solves every KCL row of port-connected components with its own Fraction
    Gaussian eliminator (port potentials prescribed); nodes in port-less
    components are gauged to zero. It does not call the reduction core.
    """
    n = len(net.nodes)
    lap = net.laplacian()
    port_set = set(prescribed)

    # Restrict the linear system to components that touch a port; the
    # port-less rows are singular and are gauged separately.
    solvable = set()
    for comp in net.connected_components():
        if any(name in port_set for name in comp):
            solvable.update(comp)

    free = [
        i
        for i, name in enumerate(net.nodes)
        if name in solvable and name not in prescribed
    ]
    fixed = {
        i: Fraction(prescribed[name])
        for i, name in enumerate(net.nodes)
        if name in prescribed
    }

    a = [[lap[i][j] for j in free] for i in free]
    b = [
        -sum((lap[i][j] * vj for j, vj in fixed.items()), Fraction(0))
        for i in free
    ]
    m = len(a)
    mat = [a[k][:] + [b[k]] for k in range(m)]
    for k in range(m):
        r = k
        while r < m and mat[r][k] == 0:
            r += 1
        if r == m:
            raise ValueError("singular system in demo reference solve")
        if r != k:
            mat[k], mat[r] = mat[r], mat[k]
        piv = mat[k][k]
        for i in range(k + 1, m):
            f = mat[i][k] / piv
            for j in range(k, m + 1):
                mat[i][j] -= f * mat[k][j]
    x = [Fraction(0)] * m
    for k in range(m - 1, -1, -1):
        s = mat[k][m] - sum(
            (mat[k][j] * x[j] for j in range(k + 1, m)), Fraction(0)
        )
        x[k] = s / mat[k][k]

    volt = [Fraction(0)] * n  # floating components stay at gauge 0
    for i, v in fixed.items():
        volt[i] = v
    for slot, node_i in enumerate(free):
        volt[node_i] = x[slot]
    return volt


def _fmt(fr: Fraction) -> str:
    return str(fr) if fr.denominator != 1 else f"{fr.numerator}"


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Fixed deterministic network.
    # ------------------------------------------------------------------
    # Ported component: a bridge/ladder between ports p and q.
    #   p-a 2 S, p-b 1 S, a-b 1 S, a-q 1 S, b-c 3 S, c-q 2 S, b-q 1 S
    # Port-less floating component: f1-f2 4 S, f2-f3 5 S.
    # Isolated node: z (explicitly reported, never hidden).
    nodes = ["p", "q", "a", "b", "c", "f1", "f2", "f3", "z"]
    edges = [
        ("p", "a", 2),
        ("p", "b", 1),
        ("a", "b", 1),
        ("a", "q", 1),
        ("b", "c", 3),
        ("c", "q", 2),
        ("b", "q", 1),
        ("f1", "f2", 4),
        ("f2", "f3", 5),
    ]
    ports = ["p", "q"]
    net = ResistorNetwork.build(nodes, edges, ports=ports)

    print("=" * 68)
    print("BEFORE reduction: full resistor network")
    print("=" * 68)
    print(f"nodes ({len(nodes)}): {nodes}")
    print(f"edges ({len(edges)}): conductance in siemens")
    for u, v, g in net.edges:
        print(f"    {u} -- {v}: {_fmt(g)} S")
    lap = net.laplacian()
    print(f"full Laplacian: {len(lap)}x{len(lap)} (one unknown potential")
    print("                per internal node)")

    reduced = reduce_network(net)

    print()
    print("=" * 68)
    print("AFTER Schur elimination: port Dirichlet-to-Neumann map")
    print("=" * 68)
    print(f"ports: {list(reduced.ports)}")
    for r, name in enumerate(reduced.ports):
        print(
            f"  S[{name}] = "
            + "  ".join(f"{_fmt(v):>4}" for v in reduced.dtm[r])
        )

    sym = all(
        reduced.dtm[r][c] == reduced.dtm[c][r]
        for r in range(2)
        for c in range(2)
    )
    rowsums = [sum(row) for row in reduced.dtm]
    print(f"symmetric: {sym}")
    print(f"row sums : {rowsums} (must be exactly 0)")

    print()
    print("components:")
    for comp in reduced.components:
        if comp.kind == "ported":
            print(
                f"  [ported]   nodes={list(comp.nodes)} "
                f"internal={list(comp.internal_nodes)} "
                f"eliminated with {len(comp.steps)} recorded steps"
            )
        else:
            print(
                f"  [FLOATING] nodes={list(comp.nodes)} "
                f"-> flagged, not folded into the port model"
            )

    # ------------------------------------------------------------------
    # Prescribed boundary: Vp = 1 V, Vq = 0 V.
    # ------------------------------------------------------------------
    prescribed = {"p": Fraction(1), "q": Fraction(0)}
    sol = reduced.solve_boundary(prescribed)

    print()
    print("=" * 68)
    print("boundary solve  V(p)=1 V, V(q)=0 V  (before vs after)")
    print("=" * 68)
    full_v = _full_gaussian_solve(net, prescribed)
    idx = net.node_index
    print(f"{'node':>4} | {'reduced (Fraction)':>20} | {'full 9x9 solve':>20}")
    for name in ["a", "b", "c", "p", "q", "f1", "f2", "f3", "z"]:
        rv = sol.voltages.get(name, Fraction(0))
        fv = full_v[idx[name]] if name not in ("f1", "f2", "f3", "z") else None
        fvs = "gauge 0 (floating)" if fv is None else _fmt(fv)
        print(f"{name:>4} | {_fmt(rv):>20} | {fvs:>20}")

    agree = all(
        sol.voltages[name] == full_v[idx[name]] for name in ("a", "b", "c")
    )
    print(f"internal potentials agree exactly with full solve: {agree}")

    print()
    print("port currents I = S V:")
    for p in ports:
        print(f"  I({p}) = {_fmt(sol.port_currents[p])} A")
    print("branch currents (edge orientation as declared):")
    for (u, v), cur in sol.branch_currents.items():
        print(f"  {u} -> {v}: {_fmt(cur)} A")

    print()
    print(
        f"power injected at ports : {_fmt(sol.power_injected)} W\n"
        f"power dissipated (g V^2): {_fmt(sol.power_dissipated)} W\n"
        f"residual                : {_fmt(sol.power_balance_residual)} W"
    )

    # ------------------------------------------------------------------
    # Actually triggered rejection boundary: zero resistance.
    # ------------------------------------------------------------------
    print()
    print("=" * 68)
    print("rejection boundary (actually triggered, not pre-recorded)")
    print("=" * 68)
    try:
        ResistorNetwork.build(
            nodes=["p", "x", "q"],
            edges=[("p", "x", 1), ("x", "q", float("inf"))],
            ports=["p", "q"],
        )
    except InfiniteConductanceError as exc:
        print("zero-resistance link (conductance +inf) -> InfiniteConductanceError:")
        print(f"  {exc}")

    try:
        ResistorNetwork.build(
            nodes=["p", "q"], edges=[("p", "q", 0)], ports=["p", "q"]
        )
    except ZeroConductanceError as exc:
        print("zero conductance (open circuit passed as edge) -> "
              "ZeroConductanceError:")
        print(f"  {exc}")

    elapsed = time.perf_counter() - t0
    print()
    print(f"demo finished in {elapsed:.3f} s (budget 8 s)")


if __name__ == "__main__":
    main()
