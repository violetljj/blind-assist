import copy
import unittest

from city_hlod_graph import merge_graph


def source(package, nested=False):
    transform = dict(translation_cm=[0,0,0], scale=[1,1,1], rotation_xyzw=[0,0,0,1])
    return dict(package=package, world_package='/Map', container_id='0', actor_instance_guid=package,
                nested_hlod_source=nested, container_transform=transform,
                editor_only_parent_transform=copy.deepcopy(transform))


def node(package, sources):
    return dict(proxy_actor_package=package, sources=sources, status='EXPORTED')


class GraphTests(unittest.TestCase):
    def test_nested_leaf_resolution_and_identity_transforms(self):
        report = dict(hlod=[node('/parent',[source('/child',True)]),node('/child',[source('/leaf')])])
        result = merge_graph([dict(actor_package=p) for p in ('/parent','/child')],[report])
        self.assertEqual(result['status'],'SOURCE_GRAPH_COMPLETE')
        self.assertEqual(result['leaf_package_count'],1)
        self.assertEqual(result['reachable_leaf_ids']['/parent'],result['reachable_leaf_ids']['/child'])

    def test_cycle_and_missing_nodes_fail_closed(self):
        rows=[dict(actor_package=p) for p in ('/a','/b','/missing')]
        result=merge_graph(rows,[dict(hlod=[node('/a',[source('/b',True)]),node('/b',[source('/a',True)])])])
        self.assertEqual(result['status'],'UNVERIFIED')
        self.assertEqual(result['missing_packages'],['/missing'])
        self.assertTrue(result['cycle_packages'])

    def test_nonidentity_transform_is_not_claimed_composed(self):
        edge=source('/leaf');edge['container_transform']['translation_cm']=[1,0,0]
        result=merge_graph([dict(actor_package='/a')],[dict(hlod=[node('/a',[edge])])])
        self.assertEqual(result['transform_composition'],'UNVERIFIED_NONIDENTITY_COMPOSITION')
        self.assertEqual(result['visible_background_isolation'],'UNVERIFIED')


if __name__=='__main__':unittest.main()
