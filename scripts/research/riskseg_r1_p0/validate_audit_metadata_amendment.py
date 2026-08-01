"""Validation-only metadata amendment for the locked RISKSEG-R1 P0 scorer.

The locked independent validator reproduced the numerical candidate selection
but omitted the non-computational ``lateral_profile`` name from serialized
``selected_config`` dictionaries. This wrapper restores that field from the
locked contract before invoking the otherwise unchanged validator.
"""

from __future__ import annotations

from . import validate_audit as locked_validator


_locked_configs = locked_validator.configs


def configs_with_profile(contract):
    values = _locked_configs(contract)
    profiles = contract["adapter_grid"]["lateral_profiles"]
    for value in values:
        matches = [
            name
            for name, weights in profiles.items()
            if [float(item) for item in weights] == value["lateral_weights"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cannot uniquely recover lateral profile for {value['config_id']}"
            )
        value["lateral_profile"] = matches[0]
    return values


locked_validator.configs = configs_with_profile


if __name__ == "__main__":
    locked_validator.main()

