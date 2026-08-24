"""Generate iitx oracle fixtures from PyPhi ``main`` (pinned).

Usage (from this directory)::

    uv run python generate.py            # fast tier
    uv run python generate.py --slow     # slow tier too (big, rule152)
    uv run python generate.py --only basic --only xor

Writes ``../fixtures/iit3/<name>.json`` and ``../fixtures/iit4/<name>.json``,
plus ``../fixtures/iit3/basic_mechanisms.json`` (the mechanism-level
micro-fixture table for the basic network).

Fails loudly: any error during a fixture's generation aborts the run unless
``--keep-going`` is passed, in which case the error is recorded in the
skipped-fixtures report and re-raised at the end.
"""

import argparse
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"

ORACLE_REF = "ce2b28321686e472650c8bbe118a19cef55ac555"
PRESETS = {"iit3": "IIT_3_0", "iit4": "IIT_4_0_2023", "iit4_2026": "IIT_4_0_2026"}
SCHEMA_VERSION = 1


def _guard_cwd():
    """PyPhi loads ``pyphi_config.yml`` from the CWD at import time.

    Force the CWD to this directory (which must not contain such a file) so
    fixtures can never be silently generated under a stray local config.
    """
    os.chdir(HERE)
    if (HERE / "pyphi_config.yml").exists():
        raise RuntimeError(
            "pyphi_config.yml found next to generate.py; refusing to "
            "generate fixtures under a local config override"
        )


_guard_cwd()
os.environ.setdefault("PYPHI_WELCOME_OFF", "1")
os.environ.setdefault("PYPHI_AGENT_NOTE_OFF", "1")

import pyphi  # noqa: E402  (import after CWD guard, by design)

import extract  # noqa: E402
from networks import get_networks  # noqa: E402


def oracle_metadata(preset_key):
    import importlib.metadata

    version = importlib.metadata.version("pyphi")
    # hatch-vcs encodes the commit as +g<short-hash>; make sure the installed
    # pyphi really is the pinned oracle commit.
    if f"g{ORACLE_REF[:9]}" not in version:
        raise RuntimeError(
            f"installed pyphi version {version!r} does not carry the pinned "
            f"oracle commit {ORACLE_REF[:9]}; refusing to generate"
        )
    return {
        "package": "pyphi",
        "repository": "https://github.com/wmayner/pyphi",
        "ref": ORACLE_REF,
        "version": version,
        "preset": PRESETS[preset_key],
        "python": platform.python_version(),
    }


def preset_config(preset_key):
    return pyphi.conf.presets.by_name[PRESETS[preset_key]]


def base_fixture(spec, substrate, preset_key):
    return {
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "description": spec.description,
        "source": spec.source,
        "slow": spec.slow,
        "oracle": oracle_metadata(preset_key),
        "conventions": extract.CONVENTIONS,
        "network": extract.network_section(spec, substrate),
    }


def generate_iit3(spec, substrate):
    with pyphi.config.override(progress_bars=False, **preset_config("iit3")):
        analysis = pyphi.analyze(substrate, spec.state)
        conceptual_info = None
        if substrate.size <= 3:
            from pyphi.formalism import iit3 as formalism_iit3

            system = pyphi.System.from_substrate(
                substrate, spec.state, substrate.node_indices
            )
            conceptual_info = formalism_iit3.conceptual_info(system)
        fixture = base_fixture(spec, substrate, "iit3")
        fixture["results"] = extract.iit3_results(
            analysis, conceptual_info=conceptual_info
        )
        if spec.mechanism_checks:
            system = pyphi.System.from_substrate(
                substrate, spec.state, substrate.node_indices
            )
            fixture["mechanism_checks"] = extract.mechanism_checks(
                system, spec.mechanism_checks
            )
    return fixture


def generate_iit4(spec, substrate, preset_key="iit4"):
    with pyphi.config.override(progress_bars=False, **preset_config(preset_key)):
        analysis = pyphi.analyze(substrate, spec.state)
        fixture = base_fixture(spec, substrate, preset_key)
        fixture["results"] = extract.iit4_results(analysis)
    return fixture


def generate_basic_mechanism_table():
    """The mechanism-level micro-fixture for the basic network (IIT 3.0)."""
    spec = next(s for s in get_networks() if s.name == "basic")
    substrate = spec.factory()
    with pyphi.config.override(progress_bars=False, **preset_config("iit3")):
        system = pyphi.System.from_substrate(
            substrate, spec.state, substrate.node_indices
        )
        fixture = base_fixture(spec, substrate, "iit3")
        fixture["name"] = "basic_mechanisms"
        fixture["description"] = (
            "Mechanism-level micro-fixtures for the basic network under the "
            "IIT 3.0 preset: cause/effect repertoires for every nonempty "
            "mechanism x purview pair (plus unconstrained repertoires), and "
            "the MIC/MIE (small-phi MIPs) for every mechanism. These are the "
            "most diagnostic quantities for a reimplementation."
        )
        fixture["results"] = extract.mechanism_table(system)
    return fixture


def write_fixture(fixture, preset_key, name=None):
    out_dir = FIXTURES / preset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name or fixture['name']}.json"
    with open(path, "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow", action="store_true", help="include slow tier")
    parser.add_argument(
        "--only", action="append", default=None, help="generate only these names"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="record per-fixture errors and continue; re-raise at the end",
    )
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pyphi.conf._global"
    )

    specs = get_networks()
    if args.only:
        unknown = set(args.only) - {s.name for s in specs}
        if unknown:
            raise SystemExit(f"unknown fixture names: {sorted(unknown)}")
        specs = [s for s in specs if s.name in args.only]
    elif not args.slow:
        specs = [s for s in specs if not s.slow]

    failures = []
    for spec in specs:
        for preset_key in spec.formalisms:
            label = f"{spec.name} [{preset_key}]"
            start = time.time()
            try:
                substrate = spec.factory()
                if preset_key == "iit3":
                    fixture = generate_iit3(spec, substrate)
                else:
                    fixture = generate_iit4(spec, substrate, preset_key)
                path = write_fixture(fixture, preset_key)
                elapsed = time.time() - start
                print(f"  ok  {label:32s} -> {path.name} ({elapsed:.1f}s)")
            except Exception as exc:  # noqa: BLE001 — reported and re-raised
                if not args.keep_going:
                    raise
                failures.append((label, exc))
                traceback.print_exc()
                print(f"FAIL  {label}: {exc}")

    if args.only is None or "basic" in args.only:
        try:
            fixture = generate_basic_mechanism_table()
            path = write_fixture(fixture, "iit3", name="basic_mechanisms")
            print(f"  ok  basic_mechanisms [iit3]        -> {path.name}")
        except Exception as exc:  # noqa: BLE001
            if not args.keep_going:
                raise
            failures.append(("basic_mechanisms [iit3]", exc))
            traceback.print_exc()

    if failures:
        print(f"\n{len(failures)} fixture(s) FAILED:", file=sys.stderr)
        for label, exc in failures:
            print(f"  {label}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
