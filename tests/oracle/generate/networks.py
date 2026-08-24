"""Canonical fixture networks for the iitx oracle harness.

Each entry defines a substrate factory, the analyzed state, and provenance.
Substrates come from ``pyphi.examples`` where PyPhi ``main`` ships them; the
rest are reconstructed here from their historical definitions:

- ``noised``: the stochastic 3-node network from PyPhi 1.2.1's test suite
  (``test/example_networks.py::noised``, tag 1.2.1 @ 852b006a).
- ``fig8c``: the second deterministic system of Fig. 8C of the IIT 4.0 paper
  (Albantakis et al. 2023), transcribed from the ``feature/iit-4.0`` branch's
  ``docs/examples/IIT_4.0_demo.ipynb`` (cells 5-11).
- ``nonbinary_ab``: a hand-constructed 2-unit system with alphabets (3, 2).
  No published numbers exist for it; PyPhi main is its only oracle. It exists
  so iitx has at least one minimal non-binary regression fixture alongside the
  literature-derived ``p53_mdm2``.

States are little-endian tuples; ``state[i]`` is the state of unit ``i``.
"""

import dataclasses
from collections.abc import Callable

import numpy as np


@dataclasses.dataclass(frozen=True)
class NetworkSpec:
    """One fixture network: how to build it and how to analyze it."""

    name: str
    factory: Callable  # () -> pyphi.Substrate
    state: tuple[int, ...]
    description: str
    source: str
    slow: bool = False
    # Formalisms to generate for this network ("iit3", "iit4").
    formalisms: tuple[str, ...] = ("iit3", "iit4")
    # Emit the full mechanism-level micro-fixture table (all mechanisms x
    # purviews: repertoires + MIPs). Only sensible for small networks.
    mechanism_table: bool = False
    # Named mechanism-level spot checks: (kind, mechanism, purview) triples,
    # kind in {"cause_info", "cause_mip", "effect_mip"}.
    mechanism_checks: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = ()


def _noised_substrate():
    import pyphi

    # PyPhi 1.2.1 test/example_networks.py::noised — state-by-node TPM,
    # little-endian rows, full connectivity.
    tpm = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8],
            [0.7, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.3],
            [0.1, 1.0, 0.0],
        ]
    )
    cm = np.ones((3, 3), dtype=int)
    return pyphi.Substrate(tpm, cm=cm, node_labels=("A", "B", "C"))


def _fig8c_substrate():
    import pyphi

    # IIT 4.0 paper Fig. 8C (second deterministic system), from the
    # feature/iit-4.0 demo notebook. State-by-node TPM, little-endian rows.
    tpm = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
        ]
    )
    cm = np.array(
        [
            [1, 1, 0],  # A->A, A->B
            [0, 1, 1],  # B->B, B->C
            [1, 1, 1],  # C->A, C->B, C->C
        ]
    )
    return pyphi.Substrate(tpm, cm=cm, node_labels=("A", "B", "C"))


def _nonbinary_ab_substrate():
    import pyphi

    # Hand-constructed deterministic 2-unit system, alphabets (3, 2):
    #   A' = (A + B) mod 3
    #   B' = 1 if A >= 1 else 0
    # Strongly connected (A <-> B, A -> A). (0, 0) is a fixed point.
    alphabet = (3, 2)
    update = (
        lambda a, b: (a + b) % 3,
        lambda a, b: 1 if a >= 1 else 0,
    )
    marginals = []
    for i, k in enumerate(alphabet):
        factor = np.zeros((*alphabet, k))
        for state in np.ndindex(*alphabet):
            factor[(*state, update[i](*state))] = 1.0
        marginals.append(factor)
    return pyphi.Substrate(
        marginals=marginals,
        state_space=((0, 1, 2), (0, 1)),
        node_labels=("A", "B"),
    )


def _example(name):
    def factory():
        import pyphi

        return getattr(pyphi.examples, name)()

    return factory


NETWORKS: tuple[NetworkSpec, ...] = (
    NetworkSpec(
        name="basic",
        factory=_example("basic_substrate"),
        state=(1, 0, 0),
        description="The standard 3-node example (A=OR, B=COPY, C=XOR) of "
        "Oizumi et al. 2014 / the PyPhi 1.x docs.",
        source="pyphi.examples.basic_substrate; published: Phi=2.3125, "
        "4 concepts, MIP cut (1,2)->(0,) [1.2.1 test_big_phi.py]; "
        "phi_s=0.41503749927884376, MIP {[0],[1,2]} [feature/iit-4.0 goldens]",
        mechanism_table=True,
    ),
    NetworkSpec(
        name="xor",
        factory=_example("xor_substrate"),
        state=(0, 0, 0),
        description="3-node all-XOR network.",
        source="pyphi.examples.xor_substrate; published: Phi=1.874999 "
        "[1.2.1 docs/examples/xor.rst]; phi_s=1.5, sum_phi_d=2.5, "
        "15 relations [feature/iit-4.0 goldens]",
    ),
    NetworkSpec(
        name="noised",
        factory=_noised_substrate,
        state=(1, 0, 0),
        description="Stochastic 3-node network from PyPhi 1.2.1's test suite.",
        source="1.2.1 test/example_networks.py::noised; published: "
        "Phi=1.928592 [1.2.1 test_big_phi.py]; phi_s=1.5232604640011718 "
        "[feature/iit-4.0 goldens, s_noised]",
    ),
    NetworkSpec(
        name="basic_noisy_selfloop",
        factory=_example("basic_noisy_selfloop_substrate"),
        state=(1, 0, 0),
        description="Basic network with noisy (eps=0.1) self-loops on every "
        "node.",
        source="pyphi.examples.basic_noisy_selfloop_substrate; published: "
        "phi_s=-0.38198987262266504 (negative) [feature/iit-4.0 goldens]",
    ),
    NetworkSpec(
        name="fig4",
        factory=_example("fig4_substrate"),
        state=(1, 0, 1),
        description="Oizumi et al. 2014 Fig. 4 network (= figs 6/8/9/10).",
        source="pyphi.examples.fig4_substrate; published: phi_s=0.0, "
        "sum_phi_d=1.7174433312179418, 15 relations [feature/iit-4.0 goldens]",
    ),
    NetworkSpec(
        name="grid3",
        factory=_example("grid3_substrate"),
        state=(0, 0, 0),
        description="3-node sigmoidal grid (IIT 4.0-era example).",
        source="pyphi.examples.grid3_substrate; published: "
        "phi_s=0.024665907374197056, 39 relations [feature/iit-4.0 goldens]",
    ),
    NetworkSpec(
        name="rule110",
        factory=_example("rule110_substrate"),
        state=(0, 0, 0),
        description="3-cell rule-110 cellular automaton ring ('magic cut' "
        "example).",
        source="pyphi.examples.rule110_substrate; published: Phi=1.35708 "
        "[1.2.1 docs/examples/magic_cut.rst doctest]",
    ),
    NetworkSpec(
        name="residue",
        factory=_example("residue_substrate"),
        state=(0, 0, 0, 0, 0),
        description="5-node residue example (A,B = ANDs of overlapping "
        "input pairs from C,D,E).",
        source="pyphi.examples.residue_substrate; published small-phi: "
        "cause_info(AB,CDE)=0.5, cause-MIP phi(AB,CDE)=0.1, "
        "cause-MIP phi(A,CD)=0.166667 [1.2.1 docs/examples/residue.rst]",
        mechanism_checks=(
            ("cause_info", (0, 1), (2, 3, 4)),
            ("cause_mip", (0, 1), (2, 3, 4)),
            ("cause_mip", (0,), (2, 3)),
        ),
    ),
    NetworkSpec(
        name="fig8c",
        factory=_fig8c_substrate,
        state=(1, 0, 0),
        description="IIT 4.0 paper Fig. 8C second deterministic 3-node "
        "system (demo-notebook tutorial system).",
        source="feature/iit-4.0 docs/examples/IIT_4.0_demo.ipynb; published: "
        "phi_s=2.0, Phi=3.642982290404643 = sum_phi_d 1.792481250360578 "
        "(4 distinctions) + sum_phi_r 1.850501040044065 (9 relations)",
    ),
    NetworkSpec(
        name="p53_mdm2",
        factory=_example("gomez_p53_mdm2_substrate"),
        state=(0, 0, 1),
        description="Multi-valued p53-Mdm2 regulatory network; P ternary, "
        "Mc/Mn binary; fixed point (P,Mc,Mn)=(0,0,1).",
        source="pyphi.examples.gomez_p53_mdm2_substrate (Gomez et al. 2021, "
        "Entropy 23(1):6, Fig. 3A). No independent IIT 4.0 golden; PyPhi "
        "main is the only oracle for this fixture.",
    ),
    NetworkSpec(
        name="nonbinary_ab",
        factory=_nonbinary_ab_substrate,
        state=(0, 0),
        description="Hand-constructed 2-unit deterministic system with "
        "alphabets (3,2): A'=(A+B) mod 3, B'=[A>=1].",
        source="Defined in networks.py; PyPhi main is the only oracle.",
    ),
    # ----- slow tier -----
    NetworkSpec(
        name="big",
        factory=None,  # built inline below
        state=(1, 1, 1, 1, 1),
        description="5-node 'big' network from PyPhi 1.2.1's test suite "
        "(explicit deterministic TPM, full connectivity).",
        source="1.2.1 test/example_networks.py::big; published: "
        "Phi=10.729491, 31 concepts, cut (2,4)->(0,1,3) "
        "[1.2.1 test_big_phi.py]",
        slow=True,
    ),
    NetworkSpec(
        name="rule152",
        factory=None,  # built inline below
        state=(0, 0, 0, 0, 0),
        description="5-cell rule-152 cellular automaton ring from PyPhi "
        "1.2.1's test suite.",
        source="1.2.1 test/example_networks.py::rule152; published: "
        "Phi=6.952286, 31 concepts [1.2.1 test_big_phi.py]; "
        "phi_s=0.8300749985576875 [feature/iit-4.0 goldens]",
        slow=True,
    ),
)


def _big_substrate():
    import pyphi

    # Transcribed literally from 1.2.1 test/example_networks.py::big
    # (state-by-node TPM, little-endian rows; cm=None => full connectivity).
    tpm = np.array(
        [
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        ]
    )
    return pyphi.Substrate(tpm, node_labels=tuple("ABCDE"))


def _rule152_substrate():
    import pyphi

    # Transcribed literally from 1.2.1 test/example_networks.py::rule152
    # (state-by-node TPM, little-endian rows, ring connectivity).
    tpm = np.array(
        [
            [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 0, 1, 0], [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 1], [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
        ]
    )
    cm = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
        ]
    )
    return pyphi.Substrate(tpm, cm=cm, node_labels=tuple("ABCDE"))


def get_networks() -> list[NetworkSpec]:
    """Return the network specs with the inline factories filled in."""
    out = []
    for spec in NETWORKS:
        if spec.factory is None:
            factory = {"big": _big_substrate, "rule152": _rule152_substrate}[spec.name]
            spec = dataclasses.replace(spec, factory=factory)
        out.append(spec)
    return out
