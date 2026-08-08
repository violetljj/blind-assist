import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


HFTF_ROOT = Path(__file__).resolve().parents[2]
OP_PACKAGE = HFTF_ROOT / "depthart_selective_scan_op_package.xml"
CONVERTER_SOURCE = HFTF_ROOT / "depthart_selective_scan_converter_op.cpp"
HTP_REFERENCE_SOURCE = Path(__file__).with_name("depthart_selective_scan_htp_reference.cpp")
HTP_BUILD_SCRIPT = Path(__file__).with_name("build_depthart_selective_scan_htp_op_package.ps1")
MIGRATION_MANIFEST = HFTF_ROOT / "DEPTHART_P0_MIGRATION_MANIFEST.json"


class DepthArtSelectiveScanOpPackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ET.parse(OP_PACKAGE).getroot()

    def test_qairt_mapping_contract(self) -> None:
        self.assertEqual(self.root.attrib["PackageName"], "DepthArtSelectiveScanPackage")
        self.assertEqual(self.root.attrib["Domain"], "com.depthart")
        op = self.root.find("./OpDefList/OpDef")
        self.assertIsNotNone(op)
        self.assertEqual(op.findtext("Name"), "SelectiveScan")
        self.assertEqual([item.findtext("Name") for item in op.findall("Input")], [
            "u", "delta", "A", "B", "C", "D", "delta_bias"
        ])
        self.assertEqual([item.findtext("Name") for item in op.findall("Output")], ["y"])
        parameters = {
            item.findtext("Name"): (item.findtext("Datatype"), item.findtext("Default"))
            for item in op.findall("Parameter")
        }
        self.assertEqual(parameters, {
            "delta_softplus": ("QNN_DATATYPE_BOOL_8", "1"),
            "out_float": ("QNN_DATATYPE_BOOL_8", "0"),
        })
        self.assertEqual(op.findtext("SupportedBackend"), "HTP")

    def test_htp_supplemental_contract_is_float32(self) -> None:
        supplemental = self.root.find("./SupplementalOpDefList[@Backend='HTP']/SupplementalOpDef")
        self.assertIsNotNone(supplemental)
        tensors = supplemental.findall("Input") + supplemental.findall("Output")
        self.assertEqual(len(tensors), 8)
        self.assertTrue(all(item.findtext("Datatype") == "QNN_DATATYPE_FLOAT_32" for item in tensors))

    def test_converter_library_contract_is_shape_and_dtype_only(self) -> None:
        source = CONVERTER_SOURCE.read_text(encoding="utf-8")
        for required in (
            "SelectiveScanShapeInference",
            "SelectiveScanDataTypeInference",
            "numOfInputs != 7",
            "numOfOutputs != 1",
            "numOfParams != 2",
            "input->v1.rank != 3",
            "output->v1.dimensions[index] = input->v1.dimensions[index]",
            "outputTensors[0].v1.dataType = op->v1.inputTensors[0].v1.dataType",
        ):
            self.assertIn(required, source)
        self.assertNotIn("QnnHtp", source)

    def test_p0c_paths_remain_frozen(self) -> None:
        manifest = json.loads(MIGRATION_MANIFEST.read_text(encoding="utf-8"))
        batch = next(item for item in manifest["batches"] if item["id"] == "P0-C")
        self.assertEqual(batch["move_status"], "blocked_preserve_parallel_task_paths")
        self.assertEqual(batch["files"], [
            "depthart_selective_scan_converter_op.cpp",
            "depthart_selective_scan_op_package.xml",
        ])

    def test_htp_reference_kernel_is_bounded_and_formula_complete(self) -> None:
        source = HTP_REFERENCE_SOURCE.read_text(encoding="utf-8")
        for required in (
            "selectivescanReferenceImpl<Tensor>",
            "kFrozenGroups = 4",
            "kFrozenStateDim = 8",
            "stableSoftplus",
            "std::exp(dt * float(A(channel, state_index)))",
            "transition * state[state_index]",
            "float(B(batch_index, group, state_index, step)) * input",
            "state[state_index] * float(C(batch_index, group, state_index, step))",
            "value + float(D(channel)) * input",
        ):
            self.assertIn(required, source)
        for forbidden in ("malloc(", "calloc(", "operator new", "std::vector", "push_back("):
            self.assertNotIn(forbidden, source)

    def test_htp_build_script_keeps_binaries_in_local_evidence(self) -> None:
        source = HTP_BUILD_SCRIPT.read_text(encoding="utf-8")
        for required in (
            "OutputRoot must be under",
            "qnn-op-package-generator",
            "ValidateSet('v73', 'v75')",
            '"build\\hexagon-$TargetArch-manual"',
            "aarch64-android-manual",
            "DepthArtSelectiveScanPackageInterfaceProvider",
            "COMPILED_NOT_RUNTIME_EVALUATED",
            "build-receipt.json",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
