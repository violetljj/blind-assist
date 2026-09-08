from pathlib import Path
import socket
import unittest
from run_city_pcg_capture import cache_service_port


class CachePortTests(unittest.TestCase):
    def test_uses_separate_service_port(self):
        port = cache_service_port(Path('test-cache-port'))
        self.assertGreaterEqual(port, 20000)
        self.assertLess(port, 30000)

    def test_existing_listener_is_preserved_and_skipped(self):
        cache = Path('test-cache-occupied')
        port = cache_service_port(cache)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', port))
            listener.listen()
            self.assertNotEqual(cache_service_port(cache), port)
            self.assertEqual(listener.getsockname()[1], port)


if __name__ == '__main__':
    unittest.main()
