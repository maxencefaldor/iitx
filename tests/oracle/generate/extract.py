"""Serialization of PyPhi ``main`` results into iitx oracle fixture JSON.

Conventions (documented in every fixture under ``"conventions"``):

- States are little-endian tuples: ``state[i]`` is the state of unit ``i``,
  and a decimal state index encodes unit 0 as the least-significant
  (fastest-varying) digit, mixed-radix for non-binary alphabets.
- ``tpm_state_by_state[i][j]`` = P(next state j | previous state i), with both
  row and column indices little-endian as above.
- Repertoires are flattened over the purview units in ascending unit order,
  little-endian (first purview unit fastest): ``np.squeeze`` of the
  purview-shaped array followed by ``ravel(order="F")``.
- ``cm[i][j] == 1`` means a directed edge from unit i to unit j.
"""

import itertools

import numpy as np

CONVENTIONS = {
    "state_indexing": "little-endian; unit 0 is the least-significant "
    "(fastest-varying) digit; mixed-radix for non-binary alphabets",
    "tpm": "state-by-state; tpm[i][j] = P(next=j | prev=i); rows and columns "
    "little-endian",
    "repertoires": "flattened over purview units in ascending unit order, "
    "little-endian (first purview unit fastest); squeeze then ravel(order='F')",
    "cm": "cm[i][j] == 1 means directed edge i -> j",
}


def le_states(alphabet):
    """All states of the given per-unit alphabet in little-endian order."""
    rev = [range(k) for k in reversed(alphabet)]
    return [tuple(reversed(s)) for s in itertools.product(*rev)]


def sbs_tpm(substrate):
    """State-by-state TPM (little-endian rows/cols) from a Substrate.

    Built as the product of the per-unit factors of the substrate's
    ``FactoredTPM`` — exact for the conditionally-independent TPMs PyPhi
    represents, for binary and non-binary alphabets alike.
    """
    factors = [np.asarray(f, dtype=float) for f in substrate.tpm.factors]
    alphabet = tuple(int(a) for a in substrate.tpm.alphabet_sizes)
    states = le_states(alphabet)
    n = len(alphabet)
    sbs = np.zeros((len(states), len(states)))
    for i, prev in enumerate(states):
        for j, nxt in enumerate(states):
            p = 1.0
            for u in range(n):
                p *= factors[u][prev + (nxt[u],)]
            sbs[i, j] = p
    if not np.allclose(sbs.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("state-by-state TPM rows do not sum to 1")
    return sbs


def flatten_repertoire(rep):
    """Flatten a purview-shaped repertoire to a little-endian list."""
    arr = np.asarray(rep, dtype=float)
    return [float(x) for x in np.squeeze(arr).ravel(order="F")]


def network_section(spec, substrate):
    """The self-contained network description of a fixture."""
    alphabet = tuple(int(a) for a in substrate.tpm.alphabet_sizes)
    return {
        "n_units": int(substrate.size),
        "alphabet": list(alphabet),
        "node_labels": list(substrate.node_labels),
        "tpm_state_by_state": [
            [float(x) for x in row] for row in sbs_tpm(substrate)
        ],
        "cm": [[int(x) for x in row] for row in np.asarray(substrate.cm)],
        "state": list(spec.state),
    }


def _partition_parts(partition):
    """Serialize a mechanism partition as [(mechanism part, purview part)]."""
    if partition is None:
        return None
    try:
        return [
            {
                "mechanism": [int(i) for i in part.mechanism],
                "purview": [int(i) for i in part.purview],
            }
            for part in partition
        ]
    except TypeError:
        return {"repr": str(partition)}


def _system_cut(partition):
    """Serialize a system partition (IIT 3.0 directed bipartition cut)."""
    if partition is None or getattr(partition, "is_null", False):
        return None
    out = {"type": type(partition).__name__}
    if hasattr(partition, "from_nodes"):
        out["from_nodes"] = [int(i) for i in partition.from_nodes]
        out["to_nodes"] = [int(i) for i in partition.to_nodes]
    if hasattr(partition, "parts"):
        try:
            out["parts"] = [[int(i) for i in part] for part in partition.parts]
        except TypeError:
            pass
    direction = getattr(partition, "direction", None)
    if direction is not None:
        out["direction"] = str(direction)
    out["num_connections_cut"] = int(partition.num_connections_cut())
    return out


def _mice_section(mice, include_repertoire=True):
    """Serialize a MaximallyIrreducibleCause/Effect (works for 3.0 and 4.0)."""
    out = {
        "phi": float(mice.phi),
        "purview": [int(i) for i in mice.purview],
        "mip_partition": _partition_parts(getattr(mice, "partition", None)),
    }
    purview_state = getattr(mice, "purview_state", None)
    if purview_state is not None:
        out["purview_state"] = [int(s) for s in purview_state]
    spec = getattr(mice, "specified_state", None)
    if spec is not None and hasattr(spec, "intrinsic_information"):
        out["specified_state"] = {
            "state": [int(s) for s in spec.state],
            "intrinsic_information": float(spec.intrinsic_information),
        }
    if include_repertoire:
        rep = getattr(mice, "repertoire", None)
        if rep is not None:
            out["repertoire"] = flatten_repertoire(rep)
        prep = getattr(mice, "partitioned_repertoire", None)
        if prep is not None:
            out["partitioned_repertoire"] = flatten_repertoire(prep)
    return out


def iit3_results(analysis, conceptual_info=None):
    """Serialize an IIT 3.0 Analysis (SIA + CES)."""
    sia = analysis.sia
    concepts = sorted(
        analysis.ces, key=lambda c: (len(c.mechanism), tuple(c.mechanism))
    )
    return {
        "phi": float(sia.phi),
        "mip_cut": _system_cut(getattr(sia, "partition", None)),
        "num_concepts": len(concepts),
        "sum_small_phi": float(sum(float(c.phi) for c in concepts)),
        "conceptual_info": (
            float(conceptual_info) if conceptual_info is not None else None
        ),
        "concepts": [
            {
                "mechanism": [int(i) for i in c.mechanism],
                "phi": float(c.phi),
                "cause": _mice_section(c.cause)
                | {"repertoire": flatten_repertoire(c.cause_repertoire)},
                "effect": _mice_section(c.effect)
                | {"repertoire": flatten_repertoire(c.effect_repertoire)},
            }
            for c in concepts
        ],
    }


def _state_spec_section(spec):
    """Serialize one side of a system-state specification."""
    return {
        "purview": [int(i) for i in spec.purview],
        "state": [int(s) for s in spec.state],
        "intrinsic_information": float(spec.intrinsic_information),
    }


def iit4_results(analysis):
    """Serialize an IIT 4.0 Analysis (SIA + Phi-structure)."""
    sia = analysis.sia
    ces = analysis.ces
    distinctions = sorted(
        ces.distinctions, key=lambda d: (len(d.mechanism), tuple(d.mechanism))
    )
    relations = ces.relations
    partition = sia.partition
    mip = None
    if partition is not None and not getattr(partition, "is_null", False):
        mip = {
            "type": type(partition).__name__,
            "parts": sorted(
                [sorted(int(i) for i in part) for part in partition.parts]
            ),
            "num_connections_cut": int(partition.num_connections_cut()),
            "repr_compact": str(partition).splitlines()[1].strip("│ ")
            if "\n" in str(partition)
            else str(partition),
        }
    system_state = sia.system_state
    return {
        "phi_s": float(sia.phi),
        "normalized_phi": float(sia.normalized_phi),
        "reasons": [str(r) for r in sia.reasons] if sia.reasons else None,
        "mip": mip,
        "system_state": {
            "cause": _state_spec_section(system_state.cause),
            "effect": _state_spec_section(system_state.effect),
        }
        if system_state is not None
        else None,
        "num_distinctions": len(distinctions),
        "sum_phi_d": float(ces.sum_phi_distinctions),
        "num_relations": int(relations.num_relations()),
        "sum_phi_r": float(ces.sum_phi_relations),
        "big_phi": float(ces.big_phi),
        "distinctions": [
            {
                "mechanism": [int(i) for i in d.mechanism],
                "phi": float(d.phi),
                "cause": _mice_section(d.cause, include_repertoire=False),
                "effect": _mice_section(d.effect, include_repertoire=False),
            }
            for d in distinctions
        ],
    }


def _powerset_nonempty(indices):
    out = []
    for r in range(1, len(indices) + 1):
        out.extend(itertools.combinations(indices, r))
    return out


def mechanism_table(system):
    """Full mechanism-level micro-fixture table for a (small) system.

    For every nonempty mechanism M and nonempty purview Z over the system's
    units: cause and effect repertoires; for every nonempty mechanism: the
    cause MIP and effect MIP (phi, purview via MICE, partition). Also the
    unconstrained repertoires over every purview (mechanism = ()).

    Run under the IIT 3.0 preset: these are the classical repertoires and
    EMD small-phi MIPs — the most diagnostic quantities for a
    reimplementation.
    """
    import pyphi

    indices = tuple(int(i) for i in system.node_indices)
    subsets = _powerset_nonempty(indices)
    repertoires = []
    for purview in subsets:
        entry = {
            "purview": [int(i) for i in purview],
            "unconstrained_cause": flatten_repertoire(
                system.cause_repertoire((), purview)
            ),
            "unconstrained_effect": flatten_repertoire(
                system.effect_repertoire((), purview)
            ),
            "by_mechanism": [],
        }
        for mechanism in subsets:
            entry["by_mechanism"].append(
                {
                    "mechanism": [int(i) for i in mechanism],
                    "cause_repertoire": flatten_repertoire(
                        system.cause_repertoire(mechanism, purview)
                    ),
                    "effect_repertoire": flatten_repertoire(
                        system.effect_repertoire(mechanism, purview)
                    ),
                }
            )
        repertoires.append(entry)

    mips = []
    for mechanism in subsets:
        mic = system.mic(mechanism)
        mie = system.mie(mechanism)
        mips.append(
            {
                "mechanism": [int(i) for i in mechanism],
                "cause": _mice_section(mic),
                "effect": _mice_section(mie),
                "phi_max": float(min(mic.phi, mie.phi)),
            }
        )
    return {"repertoires": repertoires, "mice": mips}


def mechanism_checks(system, checks):
    """Targeted mechanism-level spot checks (published small-phi values)."""
    import pyphi

    out = []
    for kind, mechanism, purview in checks:
        entry = {
            "kind": kind,
            "mechanism": [int(i) for i in mechanism],
            "purview": [int(i) for i in purview],
        }
        if kind == "cause_info":
            entry["value"] = float(system.cause_info(mechanism, purview))
        elif kind == "cause_mip":
            ria = system.cause_mip(mechanism, purview)
            entry["value"] = float(ria.phi)
            entry["mip_partition"] = _partition_parts(ria.partition)
        elif kind == "effect_mip":
            ria = system.effect_mip(mechanism, purview)
            entry["value"] = float(ria.phi)
            entry["mip_partition"] = _partition_parts(ria.partition)
        else:
            raise ValueError(f"unknown check kind: {kind}")
        out.append(entry)
    return out
