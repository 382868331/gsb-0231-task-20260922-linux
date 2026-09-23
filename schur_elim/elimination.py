"""Schur (Gaussian) elimination of internal nodes for passive resistor networks.

Public objects are re-exported from :mod:`schur_elim`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Sequence

# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


class CircuitError(ValueError):
    """Base class for all structurally invalid circuit inputs."""


class DuplicateNodeError(CircuitError):
    """A node label appears more than once in the node list."""


class DuplicateEdgeError(CircuitError):
    """Two edges connect the same unordered pair of nodes."""


class DuplicatePortError(CircuitError):
    """A node is declared as a port terminal more than once."""


class UnknownNodeError(CircuitError):
    """An edge or port references a node not present in the node list."""


class SelfLoopError(CircuitError):
    """An edge connects a node to itself."""


class ZeroConductanceError(CircuitError):
    """Conductance 0 (an absent / infinite-resistance connection)."""


class NegativeConductanceError(CircuitError):
    """Negative conductance (network must be passive)."""


class InfiniteConductanceError(CircuitError):
    """Non-finite conductance; conductance +inf means a zero-resistance link."""


class NodeLimitError(CircuitError):
    """More than the contracted maximum (30) of nodes."""


class NoPortsError(CircuitError):
    """No port terminal was declared."""


class BoundaryConditionError(CircuitError):
    """Port voltage map is missing terminals, names unknown nodes, etc."""


class EliminationSingularError(CircuitError):
    """An internal principal block is singular.

    Exact arithmetic is used, so for a connected component with at least one
    port this cannot occur; raising (rather than returning a guess) keeps
    'proved unsolvable' distinct from any budget condition. The module
    performs no budgeted search, hence no 'budget exhausted' error exists.
    """


# ---------------------------------------------------------------------------
# Scalar validation
# ---------------------------------------------------------------------------

MAX_NODES = 30


def fraction_scalar(value: object, where: str) -> Fraction:
    """Coerce an exact scalar input to :class:`Fraction`.

    ``bool`` is rejected even though it is an ``int`` subclass. Floats must be
    finite; they are converted *exactly* (the rational value of the binary
    float), so no rounding is introduced here.
    """
    if isinstance(value, bool):
        raise TypeError(f"{where}: bool is not accepted as a numeric value")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InfiniteConductanceError(
                f"{where}: value must be finite; NaN/+inf/-inf rejected "
                f"(infinite conductance is a zero-resistance link), got {value!r}"
            )
        return Fraction(value)
    raise TypeError(
        f"{where}: expected int, float or Fraction, got {type(value).__name__}"
    )


def positive_conductance(value: object, where: str) -> Fraction:
    g = fraction_scalar(value, where)
    if g == 0:
        raise ZeroConductanceError(
            f"{where}: conductance must be strictly positive, got 0 "
            f"(an open / infinite-resistance connection is not an edge)"
        )
    if g < 0:
        raise NegativeConductanceError(
            f"{where}: conductance must be strictly positive, got {g} "
            f"(passive network only)"
        )
    return g


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EliminationStep:
    """One recorded pivot row of the forward elimination (recovery trace).

    After previously eliminated pivots have been removed, the pivot row is the
    exact equation::

        internal_coeffs[0] * V(pivot)
        + sum_k internal_coeffs[k+1] * V(remaining_internal[k])
        + sum_k port_coeffs[k]     * V(port_nodes[k]) = 0

    Steps are ordered by elimination and replayed in reverse order to recover
    internal potentials from prescribed port potentials.
    """

    pivot: str
    remaining_internal: tuple[str, ...]
    port_nodes: tuple[str, ...]
    internal_coeffs: tuple[Fraction, ...]
    port_coeffs: tuple[Fraction, ...]


@dataclass(frozen=True)
class ComponentInfo:
    """One connected component and its elimination record."""

    kind: str  # "ported" or "floating"
    nodes: tuple[str, ...]
    port_nodes: tuple[str, ...]
    internal_nodes: tuple[str, ...]
    dtm_block: tuple[tuple[Fraction, ...], ...] | None
    steps: tuple[EliminationStep, ...]


@dataclass(frozen=True)
class BoundarySolution:
    """Result of prescribing every port potential."""

    voltages: dict[str, Fraction]
    internal_voltages: dict[str, Fraction]
    port_currents: dict[str, Fraction]
    branch_currents: dict[tuple[str, str], Fraction]
    power_injected: Fraction
    power_dissipated: Fraction

    @property
    def power_balance_residual(self) -> Fraction:
        """``power_injected - power_dissipated``; identically 0 in exact math."""
        return self.power_injected - self.power_dissipated


@dataclass(frozen=True)
class ReducedNetwork:
    """Port Dirichlet-to-Neumann map plus per-component elimination traces."""

    ports: tuple[str, ...]
    dtm: tuple[tuple[Fraction, ...], ...]
    components: tuple[ComponentInfo, ...]
    edges: tuple[tuple[str, str, Fraction], ...]

    # -- queries -----------------------------------------------------------

    def ported_components(self) -> tuple[ComponentInfo, ...]:
        return tuple(c for c in self.components if c.kind == "ported")

    def floating_components(self) -> tuple[ComponentInfo, ...]:
        return tuple(c for c in self.components if c.kind == "floating")

    def solve_boundary(self, voltages: Mapping[str, object]) -> BoundarySolution:
        """Recover internal potentials, currents and powers from port potentials.

        Every global port must be assigned exactly once; unknown keys, missing
        ports, non-finite numbers and booleans are rejected. Floating
        components carry no boundary information: their potentials are
        defined up to one additive constant per component, so the gauge
        ``V = 0`` is chosen (all their branch currents are then zero).
        """
        if not isinstance(voltages, Mapping):
            raise BoundaryConditionError(
                "voltages must be a mapping {port_node: potential}"
            )
        port_set = set(self.ports)
        unknown = sorted(k for k in voltages if k not in port_set)
        if unknown:
            raise BoundaryConditionError(
                f"voltage assigned to non-port node(s) {unknown!r}; "
                f"valid ports are {list(self.ports)!r}"
            )
        missing = [p for p in self.ports if p not in voltages]
        if missing:
            raise BoundaryConditionError(
                f"missing prescribed potential for port(s) {missing!r}"
            )

        v_port: dict[str, Fraction] = {
            name: fraction_scalar(voltages[name], f"port voltage for {name!r}")
            for name in self.ports
        }

        node_v: dict[str, Fraction] = dict(v_port)
        internal_v: dict[str, Fraction] = {}

        # Floating components: gauge each one to zero.
        for comp in self.floating_components():
            for n in comp.nodes:
                node_v[n] = Fraction(0)

        # Ported components: replay the elimination trace backwards.
        for comp in self.ported_components():
            solved: dict[str, Fraction] = {}
            for step in reversed(comp.steps):
                total = Fraction(0)
                for label, coeff in zip(
                    step.remaining_internal, step.internal_coeffs[1:]
                ):
                    total += coeff * solved[label]
                for label, coeff in zip(step.port_nodes, step.port_coeffs):
                    total += coeff * v_port[label]
                if step.internal_coeffs[0] == 0:
                    raise EliminationSingularError(
                        "zero pivot while replaying elimination trace"
                    )
                solved[step.pivot] = -total / step.internal_coeffs[0]
            for label, value in solved.items():
                node_v[label] = value
                internal_v[label] = value

        # Port currents from the per-component DtN blocks: I = S v.
        port_current = {p: Fraction(0) for p in self.ports}
        for comp in self.ported_components():
            assert comp.dtm_block is not None
            for i, pi in enumerate(comp.port_nodes):
                row = comp.dtm_block[i]
                port_current[pi] += sum(
                    (row[j] * v_port[pj] for j, pj in enumerate(comp.port_nodes)),
                    Fraction(0),
                )

        branch_current = {
            (u, v): g * (node_v[u] - node_v[v]) for u, v, g in self.edges
        }
        injected = sum(
            (v_port[p] * port_current[p] for p in self.ports), Fraction(0)
        )
        dissipated = sum(
            (g * (node_v[u] - node_v[v]) ** 2 for u, v, g in self.edges),
            Fraction(0),
        )
        return BoundarySolution(
            voltages=node_v,
            internal_voltages=internal_v,
            port_currents=port_current,
            branch_currents=branch_current,
            power_injected=injected,
            power_dissipated=dissipated,
        )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResistorNetwork:
    """An undirected passive resistor graph with designated port nodes.

    Instances are immutable; build them with :meth:`build`. Conductances are
    stored exactly as :class:`~fractions.Fraction`. The endpoint order of
    edges as supplied is preserved for branch-current reporting; the graph
    itself is undirected.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, Fraction], ...]
    ports: tuple[str, ...]
    node_index: dict[str, int] = field(compare=False)
    adjacency: tuple[frozenset[int], ...] = field(compare=False)

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        nodes: Sequence[str],
        edges: Sequence[tuple[str, str, object]],
        ports: Sequence[str],
    ) -> "ResistorNetwork":
        nodes = tuple(nodes)
        for n in nodes:
            if not isinstance(n, str):
                raise TypeError(
                    f"node labels must be str, got {type(n).__name__}: {n!r}"
                )
        if len(nodes) == 0:
            raise CircuitError("network must contain at least one node")
        if len(nodes) > MAX_NODES:
            raise NodeLimitError(
                f"node count {len(nodes)} exceeds the interface limit "
                f"of {MAX_NODES}"
            )
        if len(set(nodes)) != len(nodes):
            dupes = sorted({n for n in nodes if nodes.count(n) > 1})
            raise DuplicateNodeError(f"duplicate node label(s): {dupes!r}")

        index = {n: i for i, n in enumerate(nodes)}

        ports = tuple(ports)
        if len(ports) == 0:
            raise NoPortsError("at least one port terminal must be designated")
        if len(set(ports)) != len(ports):
            dupes = sorted({p for p in ports if ports.count(p) > 1})
            raise DuplicatePortError(f"duplicate port terminal(s): {dupes!r}")
        for p in ports:
            if not isinstance(p, str) or p not in index:
                raise UnknownNodeError(
                    f"port {p!r} is not one of the declared nodes"
                )

        norm_edges: list[tuple[str, str, Fraction]] = []
        seen_pairs: set[frozenset[str]] = set()
        for k, edge in enumerate(edges):
            if not isinstance(edge, (tuple, list)) or len(edge) != 3:
                raise CircuitError(
                    f"edge #{k}: expected (node_a, node_b, conductance), "
                    f"got {edge!r}"
                )
            u, v, raw_g = edge
            where = f"edge #{k} ({u!r}, {v!r})"
            if not isinstance(u, str) or u not in index:
                raise UnknownNodeError(f"{where}: endpoint {u!r} is not a node")
            if not isinstance(v, str) or v not in index:
                raise UnknownNodeError(f"{where}: endpoint {v!r} is not a node")
            if u == v:
                raise SelfLoopError(f"{where}: self-loops are not allowed")
            pair = frozenset((u, v))
            if pair in seen_pairs:
                raise DuplicateEdgeError(
                    f"{where}: an edge between this pair already exists; "
                    f"merge parallel edges into one summed conductance"
                )
            seen_pairs.add(pair)
            g = positive_conductance(raw_g, where)
            norm_edges.append((u, v, g))

        adj: list[set[int]] = [set() for _ in nodes]
        for u, v, _ in norm_edges:
            iu, iv = index[u], index[v]
            adj[iu].add(iv)
            adj[iv].add(iu)

        return cls(
            nodes=nodes,
            edges=tuple(norm_edges),
            ports=ports,
            node_index=index,
            adjacency=tuple(frozenset(s) for s in adj),
        )

    # -- graph / linear algebra -------------------------------------------

    def laplacian(self) -> list[list[Fraction]]:
        """Return a fresh copy of the n x n conductance Laplacian.

        ``L[i, i]`` is the sum of conductances incident to node ``i`` and
        ``L[i, j] = -g`` for an edge of conductance ``g`` between them.
        """
        n = len(self.nodes)
        mat = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        for u, v, g in self.edges:
            i, j = self.node_index[u], self.node_index[v]
            mat[i][i] += g
            mat[j][j] += g
            mat[i][j] -= g
            mat[j][i] -= g
        return mat

    def connected_components(self) -> tuple[tuple[str, ...], ...]:
        """Components in deterministic node-index order (isolated nodes included)."""
        n = len(self.nodes)
        seen = [False] * n
        result: list[tuple[str, ...]] = []
        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            comp: list[int] = []
            seen[start] = True
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in self.adjacency[cur]:
                    if not seen[nxt]:
                        seen[nxt] = True
                        stack.append(nxt)
            comp.sort()
            result.append(tuple(self.nodes[i] for i in comp))
        return tuple(result)


# ---------------------------------------------------------------------------
# Schur reduction
# ---------------------------------------------------------------------------


def _eliminate_component(
    network: ResistorNetwork,
    lap: list[list[Fraction]],
    comp: tuple[str, ...],
) -> ComponentInfo:
    """Build the DtN block (or floating marker) for one component."""
    idx = network.node_index
    comp_indices = [idx[n] for n in comp]
    comp_set = set(comp_indices)
    boundary = [idx[p] for p in network.ports if idx[p] in comp_set]

    if not boundary:
        return ComponentInfo(
            kind="floating",
            nodes=comp,
            port_nodes=(),
            internal_nodes=comp,
            dtm_block=None,
            steps=(),
        )

    bset = set(boundary)
    internal = [i for i in comp_indices if i not in bset]
    m, p = len(internal), len(boundary)

    # Working rows: [internal columns | carried boundary columns].
    work = [[Fraction(0) for _ in range(m + p)] for _ in range(m)]
    for a, gi in enumerate(internal):
        for b, gj in enumerate(internal):
            work[a][b] = lap[gi][gj]
        for c, gp in enumerate(boundary):
            work[a][m + c] = lap[gi][gp]

    abb = [[lap[gi][gj] for gj in boundary] for gi in boundary]

    # Rows are whole equations and may be swapped for pivoting; the column
    # variable order (internal[0..m-1]) never changes.
    port_labels = tuple(network.nodes[i] for i in boundary)
    steps: list[EliminationStep] = []
    for k in range(m):
        r = k
        while r < m and work[r][k] == 0:
            r += 1
        if r == m:
            raise EliminationSingularError(
                f"singular internal block in component containing ports "
                f"{[network.nodes[i] for i in boundary]!r}; cannot eliminate"
            )
        if r != k:
            work[k], work[r] = work[r], work[k]
        row = work[k]
        steps.append(
            EliminationStep(
                pivot=network.nodes[internal[k]],
                remaining_internal=tuple(
                    network.nodes[internal[t]] for t in range(k + 1, m)
                ),
                port_nodes=port_labels,
                internal_coeffs=tuple(row[k:m]),
                port_coeffs=tuple(row[m : m + p]),
            )
        )
        pivot = row[k]
        for i in range(k + 1, m):
            if work[i][k] == 0:
                continue
            factor = work[i][k] / pivot
            work[i][k] = Fraction(0)
            for j in range(k + 1, m + p):
                work[i][j] -= factor * row[j]

    # Back-substitute: X = A_II^{-1} A_IB from the carried columns, U X = C.
    xmat = [[Fraction(0) for _ in range(p)] for _ in range(m)]
    for k in range(m - 1, -1, -1):
        piv = work[k][k]
        for c in range(p):
            s = work[k][m + c]
            for j in range(k + 1, m):
                s -= work[k][j] * xmat[j][c]
            xmat[k][c] = s / piv

    # Schur complement S = A_BB - A_IB^T X.
    aib_t = [[lap[gp][gi] for gi in internal] for gp in boundary]
    block = [
        [
            abb[r][c]
            - sum(
                (aib_t[r][a] * xmat[a][c] for a in range(m)),
                Fraction(0),
            )
            for c in range(p)
        ]
        for r in range(p)
    ]

    return ComponentInfo(
        kind="ported",
        nodes=comp,
        port_nodes=port_labels,
        internal_nodes=tuple(network.nodes[i] for i in internal),
        dtm_block=tuple(tuple(row) for row in block),
        steps=tuple(steps),
    )


def reduce_network(network: ResistorNetwork) -> ReducedNetwork:
    """Eliminate every internal node of every ported component.

    Pure function: ``network`` is never mutated; either the whole reduction
    succeeds and a :class:`ReducedNetwork` is returned, or an exception is
    raised and no partial result is observable.
    """
    if not isinstance(network, ResistorNetwork):
        raise TypeError("reduce_network expects a ResistorNetwork")
    lap = network.laplacian()
    components = tuple(
        _eliminate_component(network, lap, comp)
        for comp in network.connected_components()
    )

    n_ports = len(network.ports)
    pidx = {p: i for i, p in enumerate(network.ports)}
    dtm = [[Fraction(0) for _ in range(n_ports)] for _ in range(n_ports)]
    for comp in components:
        if comp.kind != "ported":
            continue
        assert comp.dtm_block is not None
        for r, pr in enumerate(comp.port_nodes):
            gr = pidx[pr]
            for c, pc in enumerate(comp.port_nodes):
                dtm[gr][pidx[pc]] += comp.dtm_block[r][c]

    return ReducedNetwork(
        ports=network.ports,
        dtm=tuple(tuple(row) for row in dtm),
        components=components,
        edges=network.edges,
    )
