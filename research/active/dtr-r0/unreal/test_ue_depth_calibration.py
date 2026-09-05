"""Defect checks for metric depth and independently decoded edge verification."""
import array
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw
import ue_depth_calibration as calibration


def images(bounds, shift=0):
    w,h=calibration.W,calibration.H
    left,top,right,bottom=map(int,bounds)
    image=Image.new('RGB',(w,h),(0,0,0))
    ImageDraw.Draw(image).rectangle((left+shift,top,right-1+shift,bottom-1),fill=(255,255,255))
    depth=array.array('f',[6.])*(w*h)
    for y in range(top,bottom):
        depth[y*w+left:y*w+right]=array.array('f',[3.])*(right-left)
    return image,depth


class CalibrationTests(unittest.TestCase):
    def test_units_invalid_and_range(self):
        self.assertTrue(calibration.assess([300.]*(calibration.W*calibration.H),3)['passed'])
        self.assertFalse(calibration.assess([0.]*(calibration.W*calibration.H),3)['passed'])
        with self.assertRaises(ValueError):calibration.assess([],3)
        import math
        focal=calibration.W/(2*math.tan(math.radians(calibration.FOV/2)))
        ranges=[300*math.sqrt(1+((x-calibration.W/2)/focal)**2)
                for y in range(calibration.H) for x in range(calibration.W)]
        self.assertFalse(calibration.assess(ranges,3)['passed'])

    def test_shift_missing_and_invalid_edges(self):
        bounds=[260,135,380,225]
        for shift,passed in ((0,True),(2,True),(3,False)):
            image,depth=images(bounds,shift)
            rgb=[sum(p)/(3*255.) for p in image.getdata()]
            self.assertEqual(calibration.assess_alignment(rgb,depth,bounds)['passed'],passed)
        with self.assertRaisesRegex(ValueError,'missing_or_weak'):
            calibration.assess_alignment([0.]*(calibration.W*calibration.H),depth,bounds)
        with self.assertRaisesRegex(ValueError,'invalid_depth'):
            calibration.assess_alignment(rgb,[0.]*(calibration.W*calibration.H),bounds)

    def test_moving_schedule_and_previous_frame_rgb_rejected(self):
        schedules=[calibration.alignment_schedule(i+len(calibration.EDGE_CASES)) for i in range(4)]
        self.assertEqual([s['warmup_captures'] for s in schedules],[32,0,0,0])
        self.assertEqual([s['time_s'] for s in schedules],[0.,.1,.2,.3])
        self.assertTrue(all(s['settle_delay_s']==.1 and s['rgb_captures_outside_warmup']==2 for s in schedules))
        previous,_=images(schedules[0]['expected_bounds_px'])
        _,current_depth=images(schedules[1]['expected_bounds_px'])
        rgb=[sum(p)/(3*255.) for p in previous.getdata()]
        with self.assertRaises(ValueError):
            calibration.assess_alignment(rgb,current_depth,schedules[1]['expected_bounds_px'])

    def test_serialized_verification_and_rehashed_corruption(self):
        temporary=Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ue-calibration-test-',dir=temporary) as directory:
            root=Path(directory)
            def bind(data,name,prefix):
                (root/name).write_bytes(data)
                return {prefix+'_path':name,prefix+'_bytes':len(data),
                        prefix+'_sha256':hashlib.sha256(data).hexdigest()}
            def floats(values):
                value=array.array('f',values)
                if sys.byteorder!='little':value.byteswap()
                return value.tobytes()
            report={'status':'CAPTURE_COMPLETE_PENDING_ALIGNMENT_VALIDATION','cleanup_errors':[],
                    'cases':[],'rgb_depth_edge_alignment':{'cases':[]}}
            for index,(pitch,distance) in enumerate(calibration.CASES):
                report['cases'].append(dict(pitch_degrees=pitch,expected_axial_m=distance,
                    **bind(floats([distance]*(calibration.W*calibration.H)),f'metric{index}.f32','depth')))
            for index in range(len(calibration.EDGE_CASES)+len(calibration.MOVING_EDGE_CASES)):
                schedule=calibration.alignment_schedule(index)
                bounds=schedule['expected_bounds_px']
                image,depth=images(bounds);name=f'edge{index}.png';image.save(root/name)
                report['rgb_depth_edge_alignment']['cases'].append(dict(case=index,**schedule,
                    **bind((root/name).read_bytes(),name,'rgb'),
                    **bind(floats(depth),f'edge{index}.f32','depth')))
            (root/'result.json').write_text(json.dumps(report))
            self.assertEqual(calibration.verify(root)['edge_scanlines_verified'],120)
            with self.assertRaises(FileExistsError):calibration.verify(root)
            (root/'alignment-validation.json').unlink()
            row=report['rgb_depth_edge_alignment']['cases'][0]
            shifted,_=images(row['expected_bounds_px'],3);shifted.save(root/row['rgb_path'])
            row.update(bind((root/row['rgb_path']).read_bytes(),row['rgb_path'],'rgb'))
            (root/'result.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'exceeds_two_pixels'):calibration.verify(root)
            failure=json.loads((root/'alignment-validation.json').read_text())
            self.assertEqual(failure['status'],'FAIL')
            self.assertEqual(failure['reason'],'rgb_depth_alignment_exceeds_two_pixels')


if __name__=='__main__':unittest.main()
