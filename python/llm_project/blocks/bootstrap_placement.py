"""Cachemir Section 6: bootstrap placement via DAG shortest-path optimization.

Implements the algorithm from "Cachemir: Fully Homomorphic Encrypted Inference
of Generative Large Language Model with KV Cache" (Section 6, Figure 7,
Equation 1).

The pipeline is decomposed into a sequence of *layers*. Each layer i has:
    - multiplicative depth ell(i) (levels consumed)
    - runtime function t_i(x) (ms; depends on input level x)
    - can_bootstrap_at_input flag (False for sub-layers grouped via in-module
      bootstrapping rule, e.g. polynomial evaluation with ct count > 1)

The level graph has vertex v[i, x] = "layer i starts at input level x". Edge
e[i, x, y] -> v[i+1, y] has weight (Equation 1):

    w[i, x, y] = t_i(x) + 1{x - ell(i) < y} * t_boot(y)

For a single bootstrap target (constant t_boot ~= 275 ms), and applying the
paper's pruning rule (only edges with x == 0 OR x - y - ell(i) == 0 survive),
the search becomes O(L * D) edges.

Public API
----------
    LayerSpec(name, depth, runtime_ms, can_bootstrap_at_input)
        - runtime_ms can be a scalar (level-independent) or callable(level) -> ms
    find_optimal_placement(layers, max_level, t_boot, t_boot_fn=None)
        Returns the optimal placement plan, including per-layer (input_level,
        output_level) and which transitions require bootstrap before the layer.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union


@dataclass
class LayerSpec:
    """One layer in the decoder forward pass.

    depth: multiplicative depth (levels consumed; 0 for additive ops).
    runtime_ms: wall time (ms); scalar or callable(input_level) -> ms.
    can_bootstrap_at_input: False for mid-group sub-layers (ct count > 1).
    output_level: if set, layer ends at this fixed level (e.g. attention
        layout-shift resets to fresh).
    requires_fresh_input: if True, forces bootstrap when input < max_level
        (e.g. rms layers needing freshest-chain galois keys).
    """
    name: str
    depth: int
    runtime_ms: Union[float, Callable[[int], float]]
    can_bootstrap_at_input: bool = True
    output_level: Optional[int] = None
    requires_fresh_input: bool = False

    def cost(self, input_level: int) -> float:
        if callable(self.runtime_ms):
            return float(self.runtime_ms(input_level))
        return float(self.runtime_ms)


@dataclass
class PlacementStep:
    """Per-layer entry in the optimal plan."""
    layer_idx: int
    name: str
    input_level: int          # level x at which the layer starts
    output_level: int         # level after the layer (= input_level - depth, or max_level if bootstrapped here)
    bootstrap_before: bool    # True if a bootstrap was inserted *before* this layer

    @property
    def display(self) -> str:
        boot = "[boot] " if self.bootstrap_before else "        "
        return f"{boot}{self.layer_idx:2d} {self.name:32s} L={self.input_level:2d} -> {self.output_level:2d}"


@dataclass
class PlacementPlan:
    layers: List[LayerSpec]
    steps: List[PlacementStep]
    total_runtime_ms: float
    total_bootstraps: int
    max_level: int
    t_boot: float

    def summary(self) -> str:
        lines = [
            f"Placement plan: {self.total_bootstraps} bootstraps, "
            f"{self.total_runtime_ms:.1f} ms total (max_level={self.max_level}, t_boot={self.t_boot:.1f} ms)",
        ]
        for s in self.steps:
            lines.append("  " + s.display)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def find_optimal_placement(
    layers: Sequence[LayerSpec],
    max_level: int,
    t_boot: float,
    *,
    t_boot_fn: Optional[Callable[[int], float]] = None,
    initial_level: Optional[int] = None,
    final_level_min: int = 0,
) -> PlacementPlan:
    """Run Cachemir's pruned-DAG shortest path over the layer sequence.

    initial_level defaults to max_level (fresh input).  final_level_min is
    the minimum acceptable level after the last layer (default 0).
    """
    if initial_level is None:
        initial_level = max_level

    if not layers:
        return PlacementPlan(list(layers), [], 0.0, 0, max_level, t_boot)

    def boot_cost(target_level: int) -> float:
        return float(t_boot_fn(target_level) if t_boot_fn is not None else t_boot)

    # Topological-order relaxation with pruning (Section 6).
    # dist[x] = (cost, parent_level, bootstrap_before) to enter layer i at level x.
    INF = math.inf
    Parent = Tuple[float, Optional[int], bool]

    # Layer 0 entry levels: starting from initial_level. Optionally we could
    # also allow entering at 0 (bootstrap before layer 0) -- include both.
    dist: dict[int, Parent] = {}
    dist[initial_level] = (0.0, None, False)
    # The "bootstrap before layer 0" alternative: pay t_boot, restart at max_level.
    if (initial_level < layers[0].depth) and layers[0].can_bootstrap_at_input:
        cost_b = boot_cost(max_level)
        if max_level not in dist or cost_b < dist[max_level][0]:
            dist[max_level] = (cost_b, None, True)

    history: List[dict[int, Parent]] = [dict(dist)]

    for i, layer in enumerate(layers):
        next_dist: dict[int, Parent] = {}

        for x, (cost_x, _parent, _boot) in dist.items():
            # --- Option A: do NOT bootstrap before layer i. ---
            # Allowed only when:
            #   - x >= layer.depth (level budget OK)
            #   - layer does NOT require fresh input, OR x already == max_level
            no_boot_allowed = (x >= layer.depth) and (
                (not layer.requires_fresh_input) or (x == max_level)
            )
            if no_boot_allowed:
                # Output level: layer.output_level if set, else x - layer.depth.
                y_raw = x - layer.depth
                y = layer.output_level if layer.output_level is not None else y_raw
                w = layer.cost(x)
                cand = cost_x + w
                if y not in next_dist or cand < next_dist[y][0]:
                    next_dist[y] = (cand, x, False)

            # --- Option B: bootstrap before layer i (if allowed). ---
            # Pay t_boot(max_level), enter at max_level (single target),
            # then run layer normally.
            if layer.can_bootstrap_at_input and max_level >= layer.depth:
                y_raw = max_level - layer.depth
                y = layer.output_level if layer.output_level is not None else y_raw
                w_boot = boot_cost(max_level)
                w_layer = layer.cost(max_level)
                cand = cost_x + w_boot + w_layer
                if y not in next_dist or cand < next_dist[y][0]:
                    next_dist[y] = (cand, x, True)

        if not next_dist:
            raise RuntimeError(
                f"find_optimal_placement: layer {i} ({layer.name}) is unreachable "
                f"(depth={layer.depth} > max_level={max_level} or all paths invalid)"
            )

        dist = next_dist
        history.append(dict(dist))

    # Pick best terminal vertex meeting final_level_min.
    best_y, best_cost = None, INF
    for y, (c, _p, _b) in dist.items():
        if y >= final_level_min and c < best_cost:
            best_y, best_cost = y, c
    if best_y is None:
        raise RuntimeError(
            f"find_optimal_placement: no terminal vertex with level >= {final_level_min}"
        )

    # Backtrack via parent pointers.
    steps: List[PlacementStep] = []
    cur_level = best_y
    for i in reversed(range(len(layers))):
        entry_dist = history[i + 1]
        cost, parent_level, boot_before = entry_dist[cur_level]
        layer = layers[i]
        in_level = parent_level
        if boot_before:
            in_level_for_layer = max_level
        else:
            in_level_for_layer = parent_level if parent_level is not None else cur_level + layer.depth
        out_level = cur_level
        steps.append(PlacementStep(
            layer_idx=i,
            name=layer.name,
            input_level=in_level_for_layer,
            output_level=out_level,
            bootstrap_before=boot_before,
        ))
        cur_level = parent_level if parent_level is not None else in_level_for_layer

    steps.reverse()
    n_boot = sum(1 for s in steps if s.bootstrap_before)
    return PlacementPlan(
        layers=list(layers),
        steps=steps,
        total_runtime_ms=best_cost,
        total_bootstraps=n_boot,
        max_level=max_level,
        t_boot=t_boot,
    )


# ---------------------------------------------------------------------------
# Convenience: tabular layer construction
# ---------------------------------------------------------------------------

def build_layers_from_table(table) -> List[LayerSpec]:
    """Build LayerSpec list from tuples or dicts.

    Tuple rows: (name, depth, runtime_ms[, can_bootstrap_at_input
    [, output_level[, requires_fresh_input]]]).
    """
    _FIELDS = ("name", "depth", "runtime_ms", "can_bootstrap_at_input",
                "output_level", "requires_fresh_input")
    out = []
    for row in table:
        if isinstance(row, dict):
            out.append(LayerSpec(**row))
        elif 3 <= len(row) <= 6:
            out.append(LayerSpec(**dict(zip(_FIELDS, row))))
        else:
            raise ValueError(f"build_layers_from_table: bad row arity {len(row)}: {row}")
    return out


def render_plan_table(plan: PlacementPlan) -> str:
    """Pretty-print a markdown-ish table for the report."""
    lines = []
    lines.append(f"{'idx':>3} | {'layer':32s} | {'in':>3} | {'depth':>5} | {'out':>3} | "
                 f"{'t_layer(ms)':>10} | {'boot?':>5}")
    lines.append("-" * 86)
    total = 0.0
    for s in plan.steps:
        layer = plan.layers[s.layer_idx]
        t = layer.cost(s.input_level)
        total += t
        if s.bootstrap_before:
            total += plan.t_boot
        lines.append(
            f"{s.layer_idx:3d} | {s.name:32s} | {s.input_level:3d} | {layer.depth:5d} | "
            f"{s.output_level:3d} | {t:10.2f} | {'Y' if s.bootstrap_before else ' ':5s}"
        )
    lines.append(
        f"\ntotal: {plan.total_bootstraps} bootstraps × {plan.t_boot:.1f} ms + "
        f"layer runtime = {plan.total_runtime_ms:.1f} ms"
    )
    return "\n".join(lines)
