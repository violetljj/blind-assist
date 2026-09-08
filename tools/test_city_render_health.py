import unittest
from city_render_health import inspect_log


class RenderHealthTests(unittest.TestCase):
    def test_missing_resources_make_entire_capture_ineligible(self):
        result = inspect_log("LogNaniteStreaming: Warning: Giving up and marking resource invalid.\n"
                             "LogVTDiskCache: Failed to fetch data from DDC (key: example)")
        self.assertFalse(result['ready_data_eligible'])
        self.assertEqual(result['status'], 'REVIEW')
        self.assertEqual(sum(result['counts'].values()), 2)

    def test_optional_profiler_warning_is_not_missing_render_data(self):
        result = inspect_log("LogWindows: Failed to load 'aqProf.dll'\nLogShaderCompilers: 0 jobs")
        self.assertTrue(result['ready_data_eligible'])
        self.assertEqual(result['status'], 'NO_MATCHING_RESOURCE_ERRORS')

    def test_example_limit_does_not_hide_failure_count(self):
        result = inspect_log(('LogVTDiskCache: Failed to fetch data from DDC\n') * 70)
        self.assertEqual(result['counts']['virtual_texture_ddc_fetch_failed'], 70)
        self.assertEqual(len(result['examples']), 64)


if __name__ == '__main__':
    unittest.main()
