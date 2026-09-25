import numpy as np
from cnh_track_a_v13_generate import trajectory, make_object, object_templates, labels_for_poses, purpose_seed
from cnh_track_a_v13_evaluate import calib_threshold


def test_10hz_even_frames_equal_5hz_path():
    p = trajectory(3, 5, 1.6, 0)
    assert np.allclose(np.asarray(p['world_from_Q_10hz'])[::2], p['world_from_Q'])
    assert np.allclose(np.asarray(p['angles_10hz'])[::2], p['angles'])
    assert trajectory(3, 5, 1.6, 1)['seed'] != p['seed']


def test_object_sizes_respect_bands():
    rng = np.random.default_rng(0)
    classes = []
    for split in ('train', 'calib', 'audit'):
        for category, (lo, hi) in (('HEAD', (-.2, .42)), ('BODY', (.42, .9))):
            centre = .08 if category == 'HEAD' else .65
            for _ in range(200):
                o = make_object(split, rng, category, wide=False)
                classes.append(o['size_class'])
                t = np.asarray(o['triangles']).reshape(-1, 3)
                assert t[:, 1].min()+centre >= lo+.05 and t[:, 1].max()+centre <= hi-.05
                assert t[:, 0].max()-t[:, 0].min() <= .14+1e-9
    tiny = classes.count('tiny')/len(classes)
    assert .2 < tiny < .3


def test_labels_for_poses_matches_template_placement():
    rng = np.random.default_rng(1)
    templates = object_templates('audit', (1, 0), rng, np.random.default_rng(2))
    pose = np.eye(4)
    obj = dict(templates[0], triangles_world=np.asarray(templates[0]['triangles'])+[-.45, .08, 1.5])
    m, labels, contributors, boundary = labels_for_poses([obj], [pose])
    assert labels[0].tolist() == [1, 0, 0, 0, 0, 0]
    assert contributors[0][0] == [obj['id']]


def test_calib_threshold_respects_fpr():
    rng = np.random.default_rng(3)
    y = rng.random(2000) < .3
    s = y + rng.normal(0, 1, 2000)
    t = calib_threshold(y, s)
    assert ((s >= t) & ~y).sum() / (~y).sum() <= .05


def test_seed_purposes_distinct():
    assert purpose_seed(0, 0, 'trajectory') != purpose_seed(0, 0, 'size_placement')
