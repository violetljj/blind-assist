"""Prevent unit-test primitives or a renamed role from starting benchmark capture."""
import unittest
from unittest.mock import patch
from cnh_route_pilot import require_benchmark_source
from cnh_route_capture import validate_spec
from cnh_route_spec import RIG, frames, layouts


class SourceRoleTests(unittest.TestCase):
    def test_legacy_and_primitive_pilot_rejected(self):
        for source in ({}, {'scene_layer':'PRIMITIVE_UNIT_TEST'}, {'scene_layer':'CITY_SAMPLE_GENERALIZATION'}):
            with self.assertRaises(ValueError):
                require_benchmark_source(source)

    def test_role_rename_does_not_enable_unimplemented_renderer(self):
        with self.assertRaises(NotImplementedError):
            require_benchmark_source({'scene_layer':'REALISTIC_MODULAR'})

    def test_only_explicit_unit_fixture_enters_primitive_capture(self):
        spec={'rig':RIG,'frames':frames(layouts()[:1])[:1]}
        with self.assertRaises(ValueError):
            validate_spec(spec)
        spec.update(scene_layer='PRIMITIVE_UNIT_TEST',benchmark_eligible=False)
        validate_spec(spec)
        spec['benchmark_eligible']=True
        with self.assertRaises(ValueError):
            validate_spec(spec)

    def test_worker_refuses_before_remote_dispatch(self):
        import cnh_route_worker
        with patch.object(cnh_route_worker.Path,'read_text',return_value='{}'), patch.object(cnh_route_worker,'remote') as remote:
            with self.assertRaises(ValueError):
                cnh_route_worker.dispatch('unused.json','pilot-test',600,pilot=True)
            remote.assert_not_called()


if __name__=='__main__':
    unittest.main()
