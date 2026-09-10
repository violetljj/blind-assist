"""Frozen MZ6 controlled posed sequences; no real-time cadence claim."""
import argparse
import json
from pathlib import Path

MAP_SHA = 'cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb'


def specification():
    cases, clips = [], []
    def box(name, center, size, material='Bronze'):
        return dict(name=name, kind='cube', center_m=center, size_m=size,
                    material='/Game/StreetLab/Materials/'+material)
    for scenario in ('approach_head_bar', 'lateral_body_enter', 'thin_pole_background', 'lateral_target_exit'):
        for control in (False, True):
            clip_id = scenario + ('_empty_target' if control else '_target')
            clips.append(dict(clip_id=clip_id, scenario=scenario, empty_target_control=control))
            for frame in range(25):
                alpha = frame / 24
                camera = dict(x=26., y=0., z=1.82, pitch=0., yaw=0., roll=0.)
                background = []
                if scenario == 'approach_head_bar':
                    target = box('target', [29.3, 0., 1.82], [.08, 1.2, .08])
                    camera['x'] += 2.5*alpha
                elif scenario == 'lateral_body_enter':
                    target = box('target', [28.2, .95*(1-alpha), 1.12], [.20, .25, .60], 'Charcoal')
                elif scenario == 'thin_pole_background':
                    camera['y'] = -.10+.20*alpha
                    target = box('target', [27.9, .10, 1.37], [.035, .035, 1.25], 'Bronze')
                    background = [box('background_wall', [29.24, 0., 1.62], [.08, 3., 3.], 'Limestone')]
                else:
                    target = box('target', [27.5, 1.2*alpha, 1.12], [.20, .25, .60], 'Charcoal')
                cases.append(dict(name=f'{clip_id}_f{frame:02d}', clip_id=clip_id,
                    frame_in_clip=frame, nominal_time_s=frame/12., camera=camera,
                    floor_z_m=.12, objects=background+([] if control else [target])))
    assert len(cases)==200 and len(clips)==8
    return dict(schema='mz6-short-sequence-source-v1', map_sha256=MAP_SHA,
        sampling='POSED_SETTLED_SAMPLES_NOMINAL_12HZ_NOT_WALLCLOCK_WALKING',
        calibration=dict(width=640,height=360,horizontal_fov_degrees=100.,
            eye_height_m=1.7,pitch_degrees=0.,roll_degrees=0.,depth_max_m=100.),
        frame_budget=200, clips=clips, cases=cases,
        source_scope='CONTROLLED_WILLOW_DEVELOPMENT; exact poses and objects evaluator only; native zero is UNKNOWN/no positive support, never CLEAR')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[4]
    assert a.output.resolve().is_relative_to((root/'artifacts.local').resolve()) and not a.output.exists()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(specification(),indent=2)+'\n',encoding='utf-8')
