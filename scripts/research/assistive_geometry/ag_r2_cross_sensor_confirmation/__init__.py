"""AG R2 cross-sensor factor-confirmation executor implementation.

Importing this package never opens an archive, loads a checkpoint, runs a model,
or creates an evidence root.  Formal execution remains guarded by a separately
frozen one-shot execution lock.
"""

PROTOCOL_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_R0"
IMPLEMENTATION_LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK"
)
EXECUTION_LOCK_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_"
    "CONFIRMATION_ONE_SHOT_EXECUTION_LOCK"
)

