"""schur_elim — exact Schur elimination for passive resistor networks.

Build a conductance Laplacian from an undirected graph with strictly positive
:class:`fractions.Fraction` conductances, eliminate every node internal to a
port-connected component, and obtain the port Dirichlet-to-Neumann matrix
plus a replayable elimination trace that restores internal potentials,
currents and power. Port-less components are reported as floating, never
silently discarded. Zero resistance (infinite conductance) is rejected.

Typical use::

    from schur_elim import ResistorNetwork, reduce_network

    net = ResistorNetwork.build(
        nodes=["p", "i", "q"],
        edges=[("p", "i", 1), ("i", "q", 1)],
        ports=["p", "q"],
    )
    reduced = reduce_network(net)
    sol = reduced.solve_boundary({"p": 1, "q": 0})
"""

from .elimination import (
    MAX_NODES,
    BoundaryConditionError,
    BoundarySolution,
    CircuitError,
    ComponentInfo,
    DuplicateEdgeError,
    DuplicateNodeError,
    DuplicatePortError,
    EliminationSingularError,
    EliminationStep,
    InfiniteConductanceError,
    NegativeConductanceError,
    NoPortsError,
    NodeLimitError,
    ReducedNetwork,
    ResistorNetwork,
    SelfLoopError,
    UnknownNodeError,
    ZeroConductanceError,
    fraction_scalar,
    positive_conductance,
    reduce_network,
)

__all__ = [
    "MAX_NODES",
    "BoundaryConditionError",
    "BoundarySolution",
    "CircuitError",
    "ComponentInfo",
    "DuplicateEdgeError",
    "DuplicateNodeError",
    "DuplicatePortError",
    "EliminationSingularError",
    "EliminationStep",
    "InfiniteConductanceError",
    "NegativeConductanceError",
    "NoPortsError",
    "NodeLimitError",
    "ReducedNetwork",
    "ResistorNetwork",
    "SelfLoopError",
    "UnknownNodeError",
    "ZeroConductanceError",
    "fraction_scalar",
    "positive_conductance",
    "reduce_network",
]

__version__ = "1.0.0"
