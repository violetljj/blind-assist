"""Small synthetic integrity check; no dataset labels or trained models."""
import json
import numpy as np

from build_features import adapt_ocr, field_evidence, whole_ocr, FIELD_KEYS, union_rectangle_area, token_f1


gray = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (60, 1))
observation = {"width": 100, "height": 60, "status": "ok", "lines": [
    {"text": "Alpha Coffee", "confidence": 0.9, "box": [[10, 10], [60, 10], [60, 25], [10, 25]]},
    {"text": "Decaf", "confidence": 0.6, "box": [[10, 28], [40, 28], [40, 40], [10, 40]]},
]}
lines, status, warnings = adapt_ocr(observation, 100, 60)
assert not warnings
features, evidence = field_evidence("ALPHA coffee!", lines, gray, status)
assert set(features) == set(FIELD_KEYS) and len(features) == 10
assert features["field_exact_support"] == 1.0
assert features["field_conf_support"] == 0.9
assert features["field_text_height"] == 0.25
assert features["field_box_area"] == 0.125
assert features["field_spatial_compactness"] == 1.0
assert evidence["matched_boxes"][0]["box"] == observation["lines"][0]["box"]

blank, blank_evidence = field_evidence("", lines, gray, status)
assert all(value is None for value in blank.values())
assert blank_evidence["state"] == "empty_prediction"
failed, _ = field_evidence("coffee", [], gray, "error")
assert all(value is None for value in failed.values())
assert whole_ocr([], "error")["ocr_box_count"] is None
none, _ = field_evidence("coffee", [], gray, "no_text")
assert none["field_exact_support"] == 0 and none["field_local_sharpness"] is None

mismatched, _, mismatch_warning = adapt_ocr(observation, 200, 60)
mismatch_features, _ = field_evidence("coffee", mismatched, np.zeros((60, 200)), "ok")
assert mismatch_warning and mismatch_features["field_exact_support"] == 1
assert mismatch_features["field_text_height"] is None
assert union_rectangle_area([(0, 0, 10, 10), (5, 5, 15, 15)]) == 175
assert token_f1("", "") is None and token_f1("alpha coffee", "coffee alpha") == 1
print(json.dumps({"status": "PASS", "checks": ["10 features", "native coordinates", "prediction-only lexical support", "empty preserved", "failure distinct from no text", "dimensions mismatch disables geometry", "rectangle union", "empty agreement unknown"]}))
