"""Required tests for the schur_elim package.

Expected values come either from closed-form series/parallel analysis or from
``_reference.py`` (its own Laplacian + Gaussian solver, which never imports
the package under test). Round-trip properties are used only as supplemental
checks.
"""

from __future__ import annotations

import math
import random
import unittest
from fractions import Fraction

from schur_elim import (
    BoundaryConditionError,
    CircuitError,
    DuplicateEdgeError,
    DuplicateNodeError,
    DuplicatePortError,
    InfiniteConductanceError,
    MAX_NODES,
    NegativeConductanceError,
    NoPortsError,
    NodeLimitError,
    ReducedNetwork,
    ResistorNetwork,
    SelfLoopError,
    UnknownNodeError,
    ZeroConductanceError,
    reduce_network,
)

from tests import _reference as ref


def F(*x):
    return Fraction(*x)


class SeriesParallelTests(unittest.TestCase):
    def test_series_three_equal_resistors(self):
        # Three conductances g in series: effective conductance g/3.
        g = F(2)
        net = ResistorNetwork.build(
            nodes=["p", "a", "b", "q"],
            edges=[("p", "a", g), ("a", "b", g), ("b", "q", g)],
            ports=["p", "q"],
        )
        reduced = reduce_network(net)
        geq = F(2) / 3
        self.assertEqual(
            reduced.dtm, ((geq, -geq), (-geq, geq))
        )
        sol = reduced.solve_boundary({"p": 1, "q": 0})
        # I = 2/3 A, each segment resistance 1/2 ohm -> drop 1/3 V per edge.
        self.assertEqual(sol.voltages["a"], F(2, 3))
        self.assertEqual(sol.voltages["b"], F(1, 3))
        self.assertEqual(sol.port_currents["p"], geq)
        self.assertEqual(sol.port_currents["q"], -geq)
        self.assertEqual(sol.power_balance_residual, 0)
        self.assertEqual(sol.power_dissipated, geq)  # V=1 across

    def test_series_closed_form_mixed_fractions(self):
        # Conductances 1, 1/2, 1/3 S in series -> equivalent conductance
        # 1 / (1 + 2 + 3) = 1/6 S.
        net = ResistorNetwork.build(
            nodes=["p", "x", "y", "q"],
            edges=[
                ("p", "x", F(1)),
                ("x", "y", F(1, 2)),
                ("y", "q", F(1, 3)),
            ],
            ports=["p", "q"],
        )
        reduced = reduce_network(net)
        self.assertEqual(reduced.dtm[0][1], -F(1, 6))
        sol = reduced.solve_boundary({"p": 6, "q": 0})
        # Common current = 1 A; drops across R = 1, 2, 3 ohm segments.
        self.assertEqual(sol.port_currents["p"], 1)
        self.assertEqual(sol.voltages["x"], 5)
        self.assertEqual(sol.voltages["y"], 3)
        self.assertEqual(sol.power_dissipated, 6)
        self.assertEqual(sol.power_injected, 6)

    def test_parallel_branches(self):
        # Direct edge g0 plus two series branches (g1,g1) and (g2,g2):
        # geq = g0 + g1/2 + g2/2.
        g0, g1, g2 = F(1), F(2), F(4)
        net = ResistorNetwork.build(
            nodes=["p", "q", "a", "b"],
            edges=[
                ("p", "q", g0),
                ("p", "a", g1),
                ("a", "q", g1),
                ("p", "b", g2),
                ("b", "q", g2),
            ],
            ports=["p", "q"],
        )
        reduced = reduce_network(net)
        geq = g0 + g1 / 2 + g2 / 2
        self.assertEqual(reduced.dtm, ((geq, -geq), (-geq, geq)))
        sol = reduced.solve_boundary({"p": 1, "q": 0})
        self.assertEqual(sol.voltages["a"], F(1, 2))
        self.assertEqual(sol.voltages["b"], F(1, 2))
        self.assertEqual(sol.power_balance_residual, 0)

    def test_wheatstone_bridge_matches_analysis_and_reference(self):
        # Classic bridge: p-a 1, p-b 2, a-q 2, b-q 1, a-b 3 (all S).
        nodes = ["p", "q", "a", "b"]
        edges = [
            ("p", "a", 1), ("p", "b", 2),
            ("a", "q", 2), ("b", "q", 1),
            ("a", "b", 3),
        ]
        net = ResistorNetwork.build(nodes, edges, ports=["p", "q"])
        reduced = reduce_network(net)
        expected = ref.reference_dtm(nodes, edges, ["p", "q"])
        self.assertEqual([[F(x) for x in row] for row in reduced.dtm], expected)
        # Symmetric and zero row sum.
        self.assertEqual(reduced.dtm[0][1], reduced.dtm[1][0])
        self.assertEqual(sum(reduced.dtm[0]), 0)
        sol = reduced.solve_boundary({"p": 1, "q": 0})
        _, ref_currents, ref_power = ref.full_solution(
            nodes, edges, ["p", "q"], {"p": 1, "q": 0}
        )
        self.assertEqual(sol.port_currents["p"], ref_currents["p"])
        self.assertEqual(sol.power_dissipated, ref_power)


class StructuralTests(unittest.TestCase):
    def test_floating_component_is_flagged_not_hidden(self):
        # Ported component p-x-q plus an isolated node and a separate
        # port-less component a-b. Both must be reported explicitly.
        net = ResistorNetwork.build(
            nodes=["p", "x", "q", "a", "b", "z"],
            edges=[
                ("p", "x", 1), ("x", "q", 1),
                ("a", "b", 1),
            ],
            ports=["p", "q"],
        )
        reduced = reduce_network(net)
        floating = reduced.floating_components()
        groups = {c.nodes for c in floating}
        self.assertIn(("a", "b"), groups)
        self.assertIn(("z",), groups)
        ported = reduced.ported_components()
        self.assertEqual(len(ported), 1)
        self.assertEqual(ported[0].nodes, ("p", "x", "q"))
        # Floating component contributes nothing to the DtN.
        self.assertEqual(reduced.dtm[0][0], F(1, 2))

    def test_two_ported_components_block_diagonal(self):
        net = ResistorNetwork.build(
            nodes=["p", "q", "x", "r", "s", "y"],
            edges=[
                ("p", "x", 1), ("x", "q", 1),
                ("r", "y", 2), ("y", "s", 2),
            ],
            ports=["p", "q", "r", "s"],
        )
        reduced = reduce_network(net)
        g1, g2 = F(1, 2), F(1)
        expected = (
            ( g1, -g1,  0,  0),
            (-g1,  g1,  0,  0),
            (  0,   0,  g2, -g2),
            (  0,   0, -g2,  g2),
        )
        self.assertEqual(reduced.dtm, expected)
        sol = reduced.solve_boundary({"p": 2, "q": 0, "r": 3, "s": 0})
        self.assertEqual(sol.voltages["x"], 1)
        self.assertEqual(sol.voltages["y"], F(3, 2))

    def test_isolated_port_has_zero_dtm_row(self):
        net = ResistorNetwork.build(
            nodes=["p", "q"], edges=[], ports=["p", "q"]
        )
        reduced = reduce_network(net)
        self.assertEqual(reduced.dtm, ((0, 0), (0, 0)))
        sol = reduced.solve_boundary({"p": 5, "q": 7})
        self.assertEqual(sol.port_currents, {"p": 0, "q": 0})
        self.assertEqual(sol.power_dissipated, 0)

    def test_all_nodes_are_ports(self):
        # Nothing internal: DtN is the Laplacian itself.
        net = ResistorNetwork.build(
            nodes=["a", "b", "c"],
            edges=[("a", "b", 1), ("b", "c", 2)],
            ports=["a", "b", "c"],
        )
        reduced = reduce_network(net)
        lap = net.laplacian()
        self.assertEqual([list(r) for r in reduced.dtm], lap)
        self.assertEqual(ported_internal_nodes(reduced), ())

    def test_elimination_trace_is_recorded(self):
        net = ResistorNetwork.build(
            nodes=["p", "i1", "i2", "q"],
            edges=[("p", "i1", 1), ("i1", "i2", 1), ("i2", "q", 1)],
            ports=["p", "q"],
        )
        reduced = reduce_network(net)
        (comp,) = reduced.ported_components()
        self.assertEqual(len(comp.steps), 2)
        self.assertEqual(comp.steps[0].pivot, "i1")
        self.assertEqual(comp.steps[1].pivot, "i2")
        self.assertEqual(comp.steps[0].port_nodes, ("p", "q"))


def ported_internal_nodes(reduced: ReducedNetwork):
    out = []
    for c in reduced.ported_components():
        out.extend(c.internal_nodes)
    return tuple(out)


class RandomNetworkComparisonTests(unittest.TestCase):
    """Cross-check against the independent reference on deterministic graphs."""

    def setUp(self):
        self.rng = random.Random(20260923)

    def _make_network(self, n, edge_prob, port_count):
        nodes = [f"n{i}" for i in range(n)]
        edges = []
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                if self.rng.random() < edge_prob:
                    g = Fraction(self.rng.randint(1, 5))
                    edges.append((nodes[i], nodes[j], g))
                    seen.add((i, j))
        ports = self.rng.sample(nodes, port_count)
        return ResistorNetwork.build(nodes, edges, ports), nodes, edges, ports

    def test_dtm_and_recovery_against_reference(self):
        for trial in range(25):
            with self.subTest(trial=trial):
                n = self.rng.randint(3, 12)
                net, nodes, edges, ports = self._make_network(
                    n, edge_prob=0.45, port_count=self.rng.randint(1, 3)
                )
                reduced = reduce_network(net)

                # DtN vs independent reference.
                ref_dtm = ref.reference_dtm(nodes, edges, ports)
                self.assertEqual(
                    [[F(v) for v in row] for row in reduced.dtm], ref_dtm
                )

                # Structural properties (exact): symmetric, zero row sums,
                # and the sign pattern of a graph Laplacian (off-diagonals
                # are sums of negative Schur complements).
                size = len(ports)
                for r in range(size):
                    self.assertEqual(sum(reduced.dtm[r]), 0)
                    self.assertGreaterEqual(reduced.dtm[r][r], 0)
                    for c in range(size):
                        self.assertEqual(
                            reduced.dtm[r][c], reduced.dtm[c][r]
                        )
                        if c != r:
                            self.assertLessEqual(reduced.dtm[r][c], 0)

                # Random prescribed voltages.
                vmap = {p: self.rng.randint(-4, 4) for p in ports}
                sol = reduced.solve_boundary(vmap)
                ref_v, ref_i, ref_p = ref.full_solution(
                    nodes, edges, ports, vmap
                )
                idx = {name: k for k, name in enumerate(nodes)}
                floating = {
                    n
                    for comp in reduced.floating_components()
                    for n in comp.nodes
                }
                for k, name in enumerate(nodes):
                    if name not in floating:
                        self.assertEqual(sol.voltages[name], ref_v[k])
                for p in ports:
                    self.assertEqual(sol.port_currents[p], ref_i[p])
                self.assertEqual(sol.power_dissipated, ref_p)
                self.assertEqual(sol.power_balance_residual, 0)

                # Supplemental round trip: branch currents satisfy KCL at
                # every internal node.
                inflow = {name: F(0) for name in nodes}
                for (u, v), cur in sol.branch_currents.items():
                    inflow[u] -= cur
                    inflow[v] += cur
                for name in nodes:
                    if name not in ports and name not in floating:
                        self.assertEqual(inflow[name], 0)

    def test_connected_30_node_network_at_limit(self):
        # Interface boundary: exactly MAX_NODES nodes is accepted.
        n = MAX_NODES
        nodes = [f"v{i}" for i in range(n)]
        edges = [(nodes[i], nodes[i + 1], i + 1) for i in range(n - 1)]
        ports = [nodes[0], nodes[-1]]
        net = ResistorNetwork.build(nodes, edges, ports)
        reduced = reduce_network(net)
        ref_dtm = ref.reference_dtm(nodes, edges, ports)
        self.assertEqual([list(r) for r in reduced.dtm], ref_dtm)


class InputValidationTests(unittest.TestCase):
    def _base(self):
        return ["a", "b"], [("a", "b", 1)], ["a", "b"]

    def test_zero_conductance_rejected(self):
        nodes, _, ports = self._base()
        with self.assertRaises(ZeroConductanceError) as cm:
            ResistorNetwork.build(nodes, [("a", "b", 0)], ports)
        self.assertIn("edge #0", str(cm.exception))

    def test_negative_conductance_rejected(self):
        with self.assertRaises(NegativeConductanceError):
            ResistorNetwork.build(["a", "b"], [("a", "b", -2)], ["a", "b"])

    def test_nan_conductance_rejected(self):
        with self.assertRaises(InfiniteConductanceError):
            ResistorNetwork.build(
                ["a", "b"], [("a", "b", float("nan"))], ["a", "b"]
            )

    def test_infinite_conductance_is_zero_resistance_rejected(self):
        for bad in (float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(InfiniteConductanceError) as cm:
                    ResistorNetwork.build(
                        ["a", "b"], [("a", "b", bad)], ["a", "b"]
                    )
                self.assertIn("zero-resistance", str(cm.exception))

    def test_bool_conductance_rejected(self):
        with self.assertRaises(TypeError):
            ResistorNetwork.build(["a", "b"], [("a", "b", True)], ["a", "b"])

    def test_bool_port_voltage_rejected(self):
        net = ResistorNetwork.build(["a", "b"], [("a", "b", 1)], ["a", "b"])
        reduced = reduce_network(net)
        with self.assertRaises(TypeError):
            reduced.solve_boundary({"a": True, "b": 0})

    def test_nonfinite_port_voltage_rejected(self):
        net = ResistorNetwork.build(["a", "b"], [("a", "b", 1)], ["a", "b"])
        reduced = reduce_network(net)
        with self.assertRaises(InfiniteConductanceError):
            reduced.solve_boundary({"a": float("inf"), "b": 0})
        with self.assertRaises(InfiniteConductanceError):
            reduced.solve_boundary({"a": float("nan"), "b": 0})

    def test_unknown_node_in_edge_and_port(self):
        with self.assertRaises(UnknownNodeError):
            ResistorNetwork.build(["a", "b"], [("a", "c", 1)], ["a", "b"])
        with self.assertRaises(UnknownNodeError):
            ResistorNetwork.build(["a", "b"], [("a", "b", 1)], ["a", "z"])

    def test_self_loop_rejected(self):
        with self.assertRaises(SelfLoopError):
            ResistorNetwork.build(["a", "b"], [("a", "a", 1)], ["a", "b"])

    def test_duplicate_edge_node_port_rejected(self):
        with self.assertRaises(DuplicateEdgeError):
            ResistorNetwork.build(
                ["a", "b"], [("a", "b", 1), ("b", "a", 2)], ["a", "b"]
            )
        with self.assertRaises(DuplicateNodeError):
            ResistorNetwork.build(["a", "a"], [], ["a"])
        with self.assertRaises(DuplicatePortError):
            ResistorNetwork.build(["a", "b"], [], ["a", "a"])

    def test_too_many_nodes_rejected(self):
        nodes = [f"n{i}" for i in range(MAX_NODES + 1)]
        with self.assertRaises(NodeLimitError):
            ResistorNetwork.build(nodes, [], ["n0"])

    def test_empty_nodes_and_no_ports_rejected(self):
        with self.assertRaises(CircuitError):
            ResistorNetwork.build([], [], [])
        with self.assertRaises(NoPortsError):
            ResistorNetwork.build(["a", "b"], [("a", "b", 1)], [])

    def test_malformed_edge_rejected(self):
        with self.assertRaises(CircuitError):
            ResistorNetwork.build(["a", "b"], [("a", "b")], ["a", "b"])

    def test_boundary_map_missing_or_unknown(self):
        net = ResistorNetwork.build(
            ["p", "q", "x"], [("p", "x", 1), ("x", "q", 1)], ["p", "q"]
        )
        reduced = reduce_network(net)
        with self.assertRaises(BoundaryConditionError) as cm:
            reduced.solve_boundary({"p": 1})
        self.assertIn("missing", str(cm.exception))
        with self.assertRaises(BoundaryConditionError) as cm:
            reduced.solve_boundary({"p": 1, "q": 0, "x": 0})
        self.assertIn("non-port", str(cm.exception))
        with self.assertRaises(BoundaryConditionError):
            reduced.solve_boundary([1, 0])

    def test_reduce_rejects_wrong_type_and_network_unchanged(self):
        net = ResistorNetwork.build(
            ["p", "x", "q"], [("p", "x", 1), ("x", "q", 1)], ["p", "q"]
        )
        before = (net.nodes, net.edges, net.ports)
        with self.assertRaises(TypeError):
            reduce_network([1, 2, 3])
        # No partial/observable mutation.
        self.assertEqual((net.nodes, net.edges, net.ports), before)

    def test_float_inputs_are_exact(self):
        # 0.1 is accepted but interpreted as the exact rational of the binary
        # float; two identical 0.1 edges therefore combine exactly.
        net = ResistorNetwork.build(
            nodes=["p", "q"],
            edges=[("p", "q", 0.1)],
            ports=["p", "q"],
        )
        # The binary float 0.1 is stored exactly as its rational value,
        # not silently rounded to the decimal 1/10.
        self.assertEqual(
            net.edges[0][2], Fraction(3602879701896397, 36028797018963968)
        )
        self.assertNotEqual(net.edges[0][2], Fraction(1, 10))


class ExactArithmeticTests(unittest.TestCase):
    def test_results_are_fractions_and_power_conservation_exact(self):
        # Conductances 1/3 and 1/7 in series: geq = 1/10 S.
        net = ResistorNetwork.build(
            nodes=["p", "x", "q"],
            edges=[("p", "x", F(1, 3)), ("x", "q", F(1, 7))],
            ports=["p", "q"],
        )
        reduced = reduce_network(net)
        self.assertEqual(reduced.dtm[0][1], -F(1, 10))
        sol = reduced.solve_boundary({"p": F(13), "q": F(-2)})
        self.assertIsInstance(sol.power_dissipated, Fraction)
        self.assertEqual(
            sol.power_injected, sol.power_dissipated
        )
        # P = geq * (15 V)^2 = 22.5 W.
        self.assertEqual(sol.power_dissipated, F(45, 2))

    def test_no_float_epsilon_anywhere(self):
        # The whole pipeline must stay in Fraction land.
        net = ResistorNetwork.build(
            nodes=["p", "a", "b", "q"],
            edges=[("p", "a", F(3)), ("a", "b", F(7)), ("b", "q", F(5))],
            ports=["p", "q"],
        )
        reduced = reduce_network(net)
        sol = reduced.solve_boundary({"p": 1, "q": 0})
        for row in reduced.dtm:
            for v in row:
                self.assertIsInstance(v, Fraction)
        for v in sol.voltages.values():
            self.assertIsInstance(v, Fraction)
        self.assertFalse(any(math.isnan(float(v)) for v in sol.voltages.values()))


if __name__ == "__main__":
    unittest.main()
