"""New physical fixtures for the preregistered frozen ToF comparison.

Authoring/evaluator module only. No scores or protected data are accessed.
New geometry, not a new photon seed on an old world, defines freshness.
"""
import itertools
import numpy as np
import cnh_counterfactual_data_dev as D

SPLITS = ('cal', 'hold')
K = 2
PHOTON_PREFIX = 2026101073


def backgrounds(split):
    families = (('offset_double_backwall', 'split_side_portals') if split == 'cal'
                else ('staggered_side_piers', 'twin_side_canopies'))
    rows = []
    for fi, family in enumerate(families):
        for instance in range(2):
            rho = (.28, .59)[instance]
            floor = D.box([-8, 1.58, -8], [8, 1.77, 9], (.33, .41)[instance])
            wall = D.box([-8, -3, 4.65+.21*instance], [8, 1.58, 4.84+.21*instance], .39)
            if family == 'offset_double_backwall':
                near = [D.box([.47, -.39, 1.33], [.66, .37, 1.52], rho),
                        D.box([-.96, .51, 1.94], [-.47, 1.28, 2.17], .43),
                        D.box([.47, .98, 2.36], [1.18, 1.11, 2.63], rho)]
                wall = D.box([-8, -3, 4.75+.21*instance], [8, 1.58, 4.91+.21*instance], .39)
                near += [D.box([-.91, -.74, 3.71], [-.48, 1.37, 4.03], .51)]
            elif family == 'split_side_portals':
                near = [D.box([.47, -.61, 1.61], [.63, 1.31, 1.76], rho),
                        D.box([-.73, -.41, 1.97], [-.47, .79, 2.18], .43),
                        D.box([-.73, 1.03, 1.97], [-.47, 1.39, 2.18], .43),
                        D.box([-1.07, -.49, 2.49], [1.07, -.28, 2.71], rho)]
            elif family == 'staggered_side_piers':
                near = [D.box([.47, -.53, 1.42], [.62, .18, 1.72], rho),
                        D.box([.53, .49, 1.96], [.78, 1.29, 2.22], rho),
                        D.box([-.71, -.32, 2.18], [-.47, .38, 2.46], .43),
                        D.box([-.94, .76, 1.53], [-.47, 1.29, 1.79], .43)]
            elif family == 'twin_side_canopies':
                near = [D.box([.47, -.46, 1.37], [1.27, -.27, 2.59], rho),
                        D.box([-.99, -.59, 1.88], [-.47, -.31, 3.13], .43),
                        D.box([.47, -.46, 2.41], [.65, 1.36, 2.59], rho),
                        D.box([-.67, -.59, 2.94], [-.47, 1.34, 3.13], .43)]
            else:
                raise ValueError(family)
            near = D.mirror_boxes(near, 1 if instance == 0 else -1)
            rows.append(dict(background_id=(0 if split == 'cal' else 4)+len(rows),
                background_family=family, instance=instance, boxes=[floor, wall, *near]))
    return rows


def target(split, shape, height, category, side, rho_index, size):
    y = .10 if height == 0 else .65
    widths = (.37, .83) if split == 'cal' else (.43, .79)
    thick = (.013, .075) if split == 'cal' else (.017, .085)
    rho = ((.23, .61) if split == 'cal' else (.19, .57))[rho_index]
    inner = {'contact': (.285, .235), 'pass': (.335, .375), 'clear': (.435, .485)}[category][size]
    if shape == 'horizontal':
        width, yl, yh, depth = widths[size], y-thick[size]/2, y+thick[size]/2, thick[size]
    elif shape == 'vertical':
        width = thick[size]
        yl, yh = ((-.035, .325) if height == 0 else (.465, .825))
        depth = thick[size]
    elif shape == 'protrusion':
        width = (.185, .265)[size]
        yl, yh = y-(.085, .115)[size], y+(.085, .115)[size]
        depth = (.065, .095)[size] + (.01 if split == 'hold' else 0)
    elif shape == 'sign_edge':
        width, depth = ((widths[0], .025) if size == 0 else (.025, widths[0]))
        yl, yh = y-.085, y+.085
    else:
        raise ValueError(shape)
    xl, xh = ((inner, inner+width) if side == 1 else (-inner-width, -inner))
    return D.box([xl, yl, .65], [xh, yh, .65+depth], rho)


def scenes():
    result = {}
    for split in SPLITS:
        rows = []
        for bg, shape, height, category, side, ri, size in itertools.product(
                backgrounds(split), D.SHAPES, range(2), ('contact', 'pass', 'clear'),
                (-1, 1), range(2), range(2)):
            t = target(split, shape, height, category, side, ri, size)
            boxes = [t, *bg['boxes']]
            rows.append(dict(scene_id=len(rows), scene_uid=f'frozen-e2e-20261010/{split}/{len(rows)}',
                split=split, shape_family=shape, group=height, side=side, rho=t['rho'],
                placement=category, presence=True, size_variant=size, target_box=t,
                background_id=bg['background_id'], background_family=bg['background_family'],
                background_boxes=bg['boxes'], boxes=boxes, physical_key=D.physical_key(boxes),
                occurrence_draw_id=0))
        assert len(rows) == 768
        assert len({r['physical_key'] for r in rows}) == len(rows)
        result[split] = rows
    assert not ({r['physical_key'] for r in result['cal']} & {r['physical_key'] for r in result['hold']})
    return result


def sampling_seed(row, replica):
    words = np.frombuffer(bytes.fromhex(row['physical_key']), dtype='<u4').tolist()
    return int(np.random.SeedSequence([PHOTON_PREFIX, *words, int(replica)]).generate_state(1)[0])


def categories(rows, sensor):
    result = []
    for row in rows:
        labels = np.array([D.category_boxes(row['boxes'], sensor[f, :3, 3]) for f in D.FRAMES])
        assert np.all(labels == labels[:1])
        for f in D.FRAMES:
            assert D.category_boxes(row['background_boxes'], sensor[f, :3, 3]) == ['clear', 'clear']
        expected = ['clear', 'clear']
        expected[row['group']] = row['placement']
        assert labels[0].tolist() == expected, (row['scene_uid'], labels[0], expected)
        result.append(labels[0])
    return np.array(result)
