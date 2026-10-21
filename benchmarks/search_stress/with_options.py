"""Run a benchmark script with other defaults for the estimators' options.

The suites in ``benchmarks/`` build their estimators with the default options.
To measure an option on them unchanged, this sets new defaults from the JSON
object in the environment variable ``BEAMFEAT_OPTIONS`` and then runs the
script as ``__main__``:

    BEAMFEAT_OPTIONS='{"magnitude_slots": 10}' \\
        python benchmarks/search_stress/with_options.py benchmarks/run_benchmarks.py --suite all

The defaults of ``BeamFeatRegressor``, ``BeamFeatClassifier``,
``BeamFeatTransformer`` and ``BeamSearch`` change for the whole process; an
option a script passes explicitly still wins. A list in the JSON becomes a
tuple. With ``BEAMFEAT_OPTIONS`` unset or empty, nothing changes.
"""

from __future__ import annotations

import inspect
import json
import os
import runpy
import sys


def set_defaults(options: dict) -> None:
    """Replace the default of each named parameter of the estimators and the search."""
    from beamfeat import BeamFeatClassifier, BeamFeatRegressor, BeamFeatTransformer, BeamSearch
    from beamfeat.estimators import _BeamFeatBase

    unknown = set(options)
    for cls in (BeamFeatClassifier, BeamFeatRegressor, BeamFeatTransformer, BeamSearch, _BeamFeatBase):
        init = cls.__init__
        parameters = [p for p in inspect.signature(init).parameters.values() if p.name != "self"]
        with_default = [
            p for p in parameters if p.default is not inspect.Parameter.empty and p.kind == p.POSITIONAL_OR_KEYWORD
        ]
        defaults = list(init.__defaults__ or ())
        for index, parameter in enumerate(with_default):
            if parameter.name in options:
                defaults[index] = options[parameter.name]
                unknown.discard(parameter.name)
        init.__defaults__ = tuple(defaults)
    if unknown:
        raise SystemExit(f"BEAMFEAT_OPTIONS names no parameter of the estimators: {sorted(unknown)}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    options = json.loads(os.environ.get("BEAMFEAT_OPTIONS") or "{}")
    options = {key: tuple(value) if isinstance(value, list) else value for key, value in options.items()}
    if options:
        set_defaults(options)
        import beamfeat

        print(f"beamfeat defaults changed: {options} ({beamfeat.__file__})", file=sys.stderr, flush=True)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
