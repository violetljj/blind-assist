"""Schema dispatch preserving historical bank definitions and access policies."""
import json
from pathlib import Path

import discriminating_bank
import scenario_bank


def load_scenarios(path, split="regression", *, allow_held_out=False):
    # Routing reads the envelope only. Each owning loader checks its own complete
    # source-bound definition and access policy before returning any scenes.
    schema = json.loads(Path(path).read_text(encoding="utf-8")).get("schema")
    if schema == discriminating_bank.SCHEMA:
        return discriminating_bank.load_scenarios(path, split, allow_held_out=allow_held_out)
    if schema in scenario_bank.SUPPORTED_SCHEMAS:
        return scenario_bank.load_scenarios(path, split, allow_held_out=allow_held_out)
    raise ValueError(f"Unsupported street bank schema: {schema}")
