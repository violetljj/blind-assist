import unittest
from types import SimpleNamespace

from cnh_street_alley_build import closure_roots_for_site


class ClosureRootTests(unittest.TestCase):
    def test_each_map_closure_excludes_other_split_sources(self):
        site=dict(map_asset='/Game/BAResearchAlley/Test',
                  rows=[dict(asset='wall'),dict(asset='paving'),dict(asset='awning')],
                  insert_assets=['plastic_drum','bike_stand'],
                  wall_material='wall_mat',floor_material='floor_mat')
        sources={key:dict(source='/Game/Source/'+key+'.'+key)
                 for key in ('wall','paving','awning','trashcan',
                             'insert__plastic_drum','insert__bike_stand','insert__bollard')}
        materials={key:SimpleNamespace(get_path_name=lambda key=key:'/Game/Materials/'+key+'.'+key)
                   for key in ('wall_mat','floor_mat','other_mat')}
        self.assertEqual(closure_roots_for_site(site,sources,materials),{
            '/Game/BAResearchAlley/Test','/Game/Source/wall','/Game/Source/paving',
            '/Game/Source/awning','/Game/Source/insert__plastic_drum',
            '/Game/Source/insert__bike_stand','/Game/Materials/wall_mat',
            '/Game/Materials/floor_mat'})


if __name__=='__main__':unittest.main()
