"""Synthetic provisioning contract tests; no serial port is opened."""
import unittest

from provision_wifi import provisioning_command


class ProvisioningCommandTests(unittest.TestCase):
    def test_legacy_single_profile_is_unchanged(self):
        self.assertEqual(b"WIFI\tlocal-lan\tfake-password\n",
                         provisioning_command(dict(ssid="local-lan", password="fake-password")))

    def test_hotspot_slot_keeps_existing_lan_on_device(self):
        self.assertEqual(b"WIFI_SLOT\t1\tphone-hotspot\tfake-password\n",
                         provisioning_command(dict(ssid="phone-hotspot", password="fake-password"), 1))

    def test_both_profiles_are_one_atomic_firmware_command(self):
        config = dict(profiles=[dict(ssid="local-lan", password="fake-password"),
                                dict(ssid="phone-hotspot", password="fake-hotspot-pass")])
        self.assertEqual(b"WIFI2\tlocal-lan\tfake-password\tphone-hotspot\tfake-hotspot-pass\n",
                         provisioning_command(config))
        with self.assertRaises(ValueError):
            provisioning_command(config, 1)

    def test_rejects_controls_type_length_duplicate_and_bad_slot_without_exposing_values(self):
        sentinel = "must-not-be-printed"
        cases = [dict(ssid="bad\tname", password=sentinel),
                 dict(ssid="lan", password="short"), dict(ssid="lan", password=None),
                 dict(ssid="长" * 11, password=sentinel),
                 dict(profiles=[dict(ssid="lan", password=sentinel)] * 2),
                 dict(profiles=[]), [dict(ssid="lan", password=sentinel)]]
        for config in cases:
            with self.subTest(config_type=type(config).__name__):
                with self.assertRaises(ValueError) as caught:
                    provisioning_command(config)
                self.assertNotIn(sentinel, str(caught.exception))
        with self.assertRaises(ValueError):
            provisioning_command(dict(ssid="lan", password=sentinel), 2)


if __name__ == "__main__":
    unittest.main()
