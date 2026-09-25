"""Future sidecar contract only; no dataset or capture access."""
import copy
import json
from pathlib import Path
import unittest

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None


@unittest.skipIf(Draft202012Validator is None, 'Optional jsonschema validator is not installed')
class QueryObjectSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads(Path(__file__).with_name('cnh_query_object_attribution.schema.json').read_text())
        Draft202012Validator.check_schema(schema)
        cls.validator = Draft202012Validator(schema)
        cls.names = schema['properties']['queries']['required']

    def frame(self, query):
        return dict(schema='cnh-query-object-attribution-v1',
                    authority='EVALUATOR_ONLY_PHYSICAL_TRIANGLE_BOX_CONTACT',
                    source_manifest_sha256='a' * 64, frame_key='b' * 64,
                    layout_id='site-A/layout-1',
                    queries={name: copy.deepcopy(query) for name in self.names})

    def object(self, identity='source/site-A/layout-1/native/component:0'):
        return dict(object_id=identity, source_kind='NATIVE', source_instance_key='component:0',
                    geometry_sha256='c' * 64, category=None, thin_rod=None,
                    annotation_provenance=None)

    def test_known_empty_and_unknown_are_distinct(self):
        negative = dict(label=0, coverage='COMPLETE', attribution_status='COMPLETE', objects=[])
        unknown = dict(label=-1, coverage='INCOMPLETE', attribution_status='UNKNOWN', objects=None)
        for query in (negative, unknown):
            self.validator.validate(self.frame(query))
        unknown['objects'] = []
        negative['objects'] = None
        for query in (negative, unknown):
            self.assertFalse(self.validator.is_valid(self.frame(query)))

    def test_multiple_contacts_and_partial_coverage(self):
        objects = [self.object(), self.object('source/site-A/layout-1/native/component:1')]
        for coverage, status in [('COMPLETE', 'COMPLETE'), ('INCOMPLETE', 'PARTIAL')]:
            query = dict(label=1, coverage=coverage, attribution_status=status, objects=objects)
            self.validator.validate(self.frame(query))
            query['objects'] = []
            self.assertFalse(self.validator.is_valid(self.frame(query)))

    def test_metadata_requires_provenance_and_six_queries(self):
        obj = self.object()
        obj['thin_rod'] = False
        query = dict(label=1, coverage='COMPLETE', attribution_status='COMPLETE', objects=[obj])
        self.assertFalse(self.validator.is_valid(self.frame(query)))
        obj['annotation_provenance'] = 'object-catalog-v1/thin-rod-definition-v1'
        frame = self.frame(query)
        self.validator.validate(frame)
        del frame['queries'][self.names[0]]
        self.assertFalse(self.validator.is_valid(frame))


if __name__ == '__main__':
    unittest.main()
