# Evidence receipt validators

These dependency-free validators check receipt assertions and internal consistency.
They do not run models, authenticate measurements or grant product authority.
All committed fixtures are synthetic; actual provenance and device evidence remain
tracked in issues [#20](https://github.com/violetljj/blind-assist/issues/20) and
[#21](https://github.com/violetljj/blind-assist/issues/21).

- [YOLO export receipt](../../docs/YOLO_EXPORT_PROVENANCE_RECEIPT.md): independently
  pinned distribution identity, tensor contract and numerical assertions.
- [Android admission receipt](../../docs/ANDROID_RESEARCH_PREVIEW_ADMISSION.md):
  frozen-contract identities, per-session quality/performance and derived verdicts.

From the repository root:

```powershell
python -m unittest discover -s tools/contracts -p "test_*.py"
```

Schemas are under `schemas/`, examples under `fixtures/`. Valid FAIL and UNKNOWN
receipts are accepted as truthful records; a zero validator exit code alone does
not mean the recorded candidate passed.
