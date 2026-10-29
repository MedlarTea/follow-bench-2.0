"""Load a user-provided follower policy for ``--planner custom``.

A *policy factory* is any callable ``factory(ctx: PolicyBuildContext)`` that
returns a :class:`~followbench.sim.follow_policy_adapter.FollowerPolicyAdapter`
(typically a :class:`~followbench.interfaces.PerceptionPlanningPolicy`).

It is referenced by a spec string, either

* ``"package.module:callable"`` (importable module), or
* ``"path/to/file.py:callable"`` (loaded from a file).

The spec comes from ``--policy-factory`` or, if that is empty, from the
``FB_POLICY_FACTORY`` environment variable (convenient for batch runs).
``--policy-arg KEY=VALUE`` (repeatable) or ``FB_POLICY_ARGS="K1=V1 K2=V2"``
fill :attr:`PolicyBuildContext.options`.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shlex
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter

ENV_FACTORY = "FB_POLICY_FACTORY"
ENV_ARGS = "FB_POLICY_ARGS"


@dataclass
class PolicyBuildContext:
    """Everything a factory may need to build a policy for one episode run."""

    scene: str
    dt: float
    desired_distance: float
    follow_position: str
    # Static occupancy grid of the scene (.npz), if the scene provides one.
    grid_npz: Optional[str] = None
    # Free-form ``--policy-arg KEY=VALUE`` pairs (values are strings).
    options: Dict[str, str] = field(default_factory=dict)
    # Full runner argparse namespace (read-only use recommended).
    args: Optional[argparse.Namespace] = None


def add_policy_factory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy-factory",
        type=str,
        default="",
        help=("Factory for --planner custom: 'pkg.module:callable' or 'file.py:callable'. "
              f"Falls back to ${ENV_FACTORY}."),
    )
    parser.add_argument(
        "--policy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=f"Option passed to the custom policy factory (repeatable). Also read from ${ENV_ARGS}.",
    )


def parse_policy_options(pairs: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs:
        item = str(item).strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--policy-arg expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_policy_factory(spec: str) -> Callable[[PolicyBuildContext], FollowerPolicyAdapter]:
    spec = str(spec or "").strip()
    if ":" not in spec:
        raise SystemExit(f"policy factory spec must be 'module:callable' or 'file.py:callable', got {spec!r}")
    target, attr = spec.rsplit(":", 1)
    if target.endswith(".py") or os.sep in target:
        path = os.path.abspath(target)
        if not os.path.isfile(path) and not os.path.isabs(target):
            from followbench.paths import REPO_ROOT  # noqa: PLC0415

            path = os.path.join(str(REPO_ROOT), target)
        if not os.path.isfile(path):
            raise SystemExit(f"policy factory file not found: {target}")
        mod_name = "_fb_custom_policy_" + os.path.splitext(os.path.basename(path))[0]
        mspec = importlib.util.spec_from_file_location(mod_name, path)
        if mspec is None or mspec.loader is None:
            raise SystemExit(f"cannot load policy factory file: {path}")
        module = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise SystemExit(f"{attr!r} in {target!r} is not callable")
    return factory


def build_custom_policy(args: argparse.Namespace, scene: str) -> FollowerPolicyAdapter:
    """Build the ``--planner custom`` policy from runner args."""
    spec = str(getattr(args, "policy_factory", "") or os.environ.get(ENV_FACTORY, "")).strip()
    if not spec:
        raise SystemExit(f"--planner custom requires --policy-factory (or ${ENV_FACTORY}).")
    options = parse_policy_options(shlex.split(os.environ.get(ENV_ARGS, "")))
    options.update(parse_policy_options(getattr(args, "policy_arg", None) or []))
    ctx = PolicyBuildContext(
        scene=str(scene),
        dt=float(args.dt),
        desired_distance=float(args.desired_distance),
        follow_position=str(args.follow_position),
        grid_npz=getattr(args, "grid_npz", None),
        options=options,
        args=args,
    )
    policy = load_policy_factory(spec)(ctx)
    if not isinstance(policy, FollowerPolicyAdapter):
        missing = [m for m in ("reset", "act") if not callable(getattr(policy, m, None))]
        if missing:
            raise SystemExit(f"custom policy from {spec!r} lacks {missing}")
    print(f"[POLICY] custom policy {type(policy).__name__} from {spec} options={options}", flush=True)
    return policy
