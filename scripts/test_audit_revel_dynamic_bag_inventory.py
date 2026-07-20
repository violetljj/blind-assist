import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("audit_revel_dynamic_bag_inventory.py")
SPEC = importlib.util.spec_from_file_location("revel_inventory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FakeConnection:
    msgtype = "geometry_msgs/msg/PoseStamped"
    msgcount = 7


class RevelDynamicBagInventoryTest(unittest.TestCase):
    def test_topic_record_preserves_source_fields(self):
        self.assertEqual(
            {"topic": "/vicon/person", "msgtype": "geometry_msgs/msg/PoseStamped", "message_count": 7},
            MODULE._topic_record("/vicon/person", FakeConnection()),
        )


if __name__ == "__main__":
    unittest.main()
