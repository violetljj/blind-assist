import unittest

from cnh_route_source_capture import prevent_reentry


class SlateReentryTests(unittest.TestCase):
    def test_nested_slate_pump_does_not_repeat_transition(self):
        transitions=[]
        @prevent_reentry
        def tick(delta):
            transitions.append('begin')
            tick(delta)
            transitions.append('end')
        tick(0)
        tick(0)
        self.assertEqual(transitions,['begin','end','begin','end'])

    def test_exception_releases_callback_guard(self):
        calls=[]
        @prevent_reentry
        def tick():
            calls.append(1)
            if len(calls)==1:
                raise ValueError('compile failed')
            return 'recovered'
        with self.assertRaises(ValueError):
            tick()
        self.assertEqual(tick(),'recovered')


if __name__=='__main__':
    unittest.main()
