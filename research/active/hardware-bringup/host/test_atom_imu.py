"""Focused IMU contracts with synthetic packets; no device or model invocation."""
import hashlib
import json
import math
from pathlib import Path
from collections import deque
import tempfile
import unittest
import atom_imu as imu

ROOT = Path(__file__).resolve().parents[4]
TEMP = ROOT/'artifacts.local/work/atom-imu-host-tests'


def frame(seq=1, boot='test-boot', gravity=(0, 0, 1)):
    return dict(type='imu_frame', schema_version=1, seq=seq, device_us=seq*10000,
                device_id='test-atom', boot_id=boot, accel_raw=[0,0,8192], gyro_raw=[0,0,0],
                accel_g=[0.,0.,1.], gyro_dps=[0.,0.,0.], gravity_unit=list(gravity),
                pitch_deg=0., roll_deg=0., valid=True, sample_gap=False)


def record(ns, sensor=None):
    return dict(sensor=sensor or frame(), host_received_monotonic_ns=ns, raw_offset=0, raw_length=10)


class ImuTests(unittest.TestCase):
    def test_short_probe_drift_unavailable_but_long_nonoverlapping_window_keeps_measurement(self):
        from atom_imu_probe import summarize
        TEMP.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            path=Path(directory)/'receipts.jsonl'
            rows=[]
            for seq,t,pitch in ((1,0,0.),(2,5_000_000,2.)):
                sensor=frame(seq);sensor.update(device_us=t,pitch_deg=pitch)
                rows.append(dict(valid_frame=True,sensor=sensor))
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            short=summarize(directory)
            self.assertIsNone(short['attitude']['pitch_deg']['drift_last5s_minus_first5s_deg'])
            self.assertIn('overlap',short['attitude']['pitch_deg']['drift_unavailable_reason'])
            rows[-1]['sensor']['device_us']=60_000_000
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            long=summarize(directory)
            self.assertEqual(2,long['attitude']['pitch_deg']['drift_last5s_minus_first5s_deg'])
            self.assertIsNone(long['attitude']['pitch_deg']['drift_unavailable_reason'])
    def test_engine_per_frame_pose_preserves_history_until_identity_or_availability_changes(self):
        import numpy as np
        from m3_live_engine import Engine
        engine=Engine.__new__(Engine)
        engine.nominal_pitch=engine.pitch=-10.
        engine.transform=np.eye(4)
        def rotation(deg,axis):
            angle=math.radians(deg);c,s=math.cos(angle),math.sin(angle)
            return np.array([[1,0,0],[0,c,-s],[0,s,c]]) if axis=='x' else np.array([[c,-s,0],[s,c,0],[0,0,1]])
        # Match the inherited projector helper: x/y only, so z would fall back to x.
        engine.rotation=lambda deg,axis:rotation(deg,'x')
        engine.imu_mounting=imu.derive_mounting([0,0,1],[0,.5,math.sqrt(.75)],atom_device_id='test-atom',mounting_id='synthetic',evidence='synthetic')
        engine.imu_mounting['verified']=True
        engine.pose_epoch=engine.pose_last=None
        engine.history=deque([1]);engine.projected_history=deque([2]);engine.logits=deque([3]);engine.last=None;engine.resets=0
        pose,reset=engine.frame_pose(None)
        self.assertEqual('NOMINAL_MOUNT_ASSUMPTION',pose['source'])
        self.assertEqual(-10,pose['used_pitch_deg'])
        self.assertFalse(reset)
        pair=dict(available=True,sensor=frame(gravity=(0,.5,math.sqrt(.75))),residual_ms=1)
        pose,reset=engine.frame_pose(pair)
        self.assertTrue(reset)
        self.assertAlmostEqual(-30,pose['pitch_deg'])
        engine.projected_history.append('preserved')
        pair['sensor']=frame(2,gravity=(0,-.5,math.sqrt(.75)))
        pose,reset=engine.frame_pose(pair)
        self.assertFalse(reset)
        self.assertEqual(['preserved'],list(engine.projected_history))
        self.assertAlmostEqual(30,pose['pitch_deg'])
        self.assertFalse(np.allclose(engine.transform[:3,:3],rotation(-30,'x')))
        pair['sensor']=frame(3,gravity=(.5,0,math.sqrt(.75)))
        pose,reset=engine.frame_pose(pair)
        self.assertFalse(reset)
        self.assertAlmostEqual(30,pose['roll_deg'])
        np.testing.assert_allclose(engine.transform[:3,:3],rotation(30,'z'),atol=1e-15)
        _,reset=engine.frame_pose(None)
        self.assertTrue(reset)
        self.assertFalse(engine.projected_history)

    def test_guided_capture_writes_per_cnh_pair_and_alignment_summary(self):
        import real_labeled_capture as capture
        TEMP.mkdir(parents=True,exist_ok=True)
        cnh=dict(type='cnh_frame',seq=1,ms=100,rows=8,cols=8,bins=16,
                 distance_mm=[1000]*64,target_status=[5]*64,nb_target=[1]*64,
                 hist_raw=[[0]*16]*64,hist_scaler=[[0]*16]*64,ambient_raw=[0]*64,ambient_scaler=[0]*64)
        class Clock:
            value=0.
            def monotonic(self):return self.value
            def monotonic_ns(self):return int(self.value*1e9)
            def utcnow(self):return 'synthetic-utc'
        clock=Clock()
        class Link:
            timeout=.2
            def read(self,_):
                clock.value+=.2
                cnh['seq']+=1;cnh['ms']+=200
                return (json.dumps(cnh)+'\n').encode()
        class Feed:
            def snapshot(self):return {'status':'SYNTHETIC_TEST'}
            def align(self,ns):return dict(available=True,status='ALIGNED',residual_ms=-4,
                                          sensor=frame(),host_received_monotonic_ns=ns-4_000_000)
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            app=capture.GuidedCapture(directory,lambda:Link(),clock=clock,print_fn=lambda *_:None,replay=True,imu_feed=Feed())
            app.device=Link()
            result=app.collect(Path(directory)/'segment',.5,{})
            rows=[json.loads(line) for line in (Path(directory)/'segment/frames.jsonl').read_text().splitlines()]
            self.assertEqual(3,len(rows))
            self.assertTrue(all(row['imu']['residual_ms']==-4 for row in rows))
            summary=result['metadata']['imu_alignment_summary']
            self.assertEqual(3,summary['aligned_frames'])
            self.assertEqual(4,summary['absolute_max_residual_ms'])
            self.assertFalse(summary['exposure_synchronized'])
    def test_chunk_final_receipt_and_exact_offsets(self):
        records=[]
        decoder=imu.ImuDecoder(records.append)
        first=(json.dumps(frame())+'\n').encode()
        second=(json.dumps(frame(2))+'\r\n').encode()
        decoder.feed(first[:9],10)
        self.assertEqual(records,[])
        decoder.feed(first[9:]+second,20,'test-utc')
        self.assertEqual([20,20],[r['host_received_monotonic_ns'] for r in records])
        self.assertEqual([0,len(first)],[r['raw_offset'] for r in records])
        self.assertEqual([len(first),len(second)],[r['raw_length'] for r in records])
        self.assertEqual([frame(),frame(2)],[r['sensor'] for r in records])

    def test_invalid_attitude_nonfinite_and_raw_shape_never_available(self):
        for change in ({'pitch_deg':float('nan')},{'valid':False},{'gyro_raw':[1,2]}, {'gravity_unit':[0,0,0]}):
            broken=dict(frame(),**change)
            with self.assertRaises(ValueError):imu.validate_frame(broken)
        records=[]
        decoder=imu.ImuDecoder(records.append)
        decoder.feed(b'{"type":"imu_frame","pitch_deg":NaN}\nnot-json\n',30)
        self.assertEqual(2,decoder.counts['invalid_lines'])
        self.assertFalse(any(r.get('valid_frame') for r in records))

    def test_nearest_receipt_is_signed_and_age_bounded(self):
        buffer=imu.ImuBuffer()
        buffer.add(record(90_000_000,frame(1)))
        buffer.add(record(106_000_000,frame(2)))
        pair=buffer.nearest(100_000_000,future_wait_ms=0)
        self.assertEqual(6,pair['residual_ms'])
        self.assertEqual(2,pair['sensor']['seq'])
        self.assertTrue(pair['available'])
        old=buffer.nearest(300_000_000,future_wait_ms=0)
        self.assertEqual('IMU_TOO_OLD',old['status'])
        self.assertIsNone(old['sensor'])
        self.assertEqual(-194,old['residual_ms'])

    def test_boot_or_clock_regression_clears_old_receipts(self):
        buffer=imu.ImuBuffer()
        buffer.add(record(100_000_000,frame(30)))
        buffer.add(record(110_000_000,frame(1,'new-boot')))
        self.assertEqual(1,len(buffer.records))
        buffer.add(record(120_000_000,frame(0,'new-boot')))
        self.assertEqual(1,len(buffer.records))

    def test_no_mounting_or_wrong_atom_never_fabricates_optical_pose(self):
        pair=dict(available=True,sensor=frame(),residual_ms=2)
        self.assertFalse(imu.calibrated_attitude(pair,None)['available'])
        mounting=imu.derive_mounting([0,0,1],[0,.5,math.sqrt(.75)],atom_device_id='another-atom',mounting_id='synthetic',evidence='synthetic')
        mounting['verified']=True
        self.assertEqual('MOUNTING_DEVICE_ID_MISMATCH',imu.calibrated_attitude(pair,mounting)['reason'])

    def test_arbitrary_native_axis_basis_down_negative_roll_correct(self):
        level=[0,0,1];down=[0,.5,math.sqrt(.75)]
        mounting=imu.derive_mounting(level,down,atom_device_id='test-atom',mounting_id='synthetic',evidence='synthetic')
        mounting.update(verified=True)
        for gravity,pitch in ((level,0),(down,-30),([0,-.5,math.sqrt(.75)],30)):
            pose=imu.calibrated_attitude(dict(available=True,sensor=frame(gravity=gravity),residual_ms=3),mounting)
            self.assertAlmostEqual(pitch,pose['pitch_deg'])
            self.assertAlmostEqual(0,pose['roll_deg'])
        # Here ToF right maps to IMU -X. Positive optical roll tilts fused up toward +X.
        pose=imu.calibrated_attitude(dict(available=True,sensor=frame(gravity=(.5,0,math.sqrt(.75))),residual_ms=3),mounting)
        self.assertAlmostEqual(30,pose['roll_deg'])
        mounting.update(verification_level='OPERATOR_PILOT_ZERO',
                        absolute_optical_pitch_accuracy_verified=False,
                        tof_atom_extrinsic_independently_measured=False,
                        zero_repeat_residual={'pitch_deg':3.85},
                        installation_reference={'source_sha256':'synthetic-sha'})
        pilot=imu.calibrated_attitude(dict(available=True,sensor=frame(gravity=down),residual_ms=-2),mounting)
        self.assertEqual('MEASURED_IMU_OPERATOR_PILOT_ZERO',pilot['source'])
        self.assertFalse(pilot['absolute_optical_pitch_accuracy_verified'])
        self.assertEqual({'pitch_deg':3.85},pilot['zero_repeat_residual'])
        self.assertEqual('synthetic-sha',pilot['installation_reference']['source_sha256'])
        self.assertAlmostEqual(-30,pilot['pitch_deg'])

    def test_verification_gate_requires_actual_level_wall_and_proper_rotation(self):
        TEMP.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            path=Path(directory)/'mounting.json'
            candidate=imu.derive_mounting([0,0,1],[0,.5,math.sqrt(.75)],atom_device_id='test-atom',mounting_id='synthetic',evidence='synthetic')
            path.write_text(json.dumps(candidate),encoding='utf-8')
            with self.assertRaises(ValueError):imu.load_mounting(path)
            candidate.update(verified=True,level_reference={'verified':True},wall_check={'verified':True},up_check={'verified':True})
            path.write_text(json.dumps(candidate),encoding='utf-8')
            self.assertEqual('synthetic',imu.load_mounting(path)['mounting_id'])
            candidate['tof_from_imu_rotation'][0]=[0,0,0]
            path.write_text(json.dumps(candidate),encoding='utf-8')
            with self.assertRaises(ValueError):imu.load_mounting(path)

    def test_feed_saves_original_bytes_and_receipt_index(self):
        TEMP.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            chunks=[b'{"type":"imu_boot","ready":true}\n'+json.dumps(frame()).encode()[:20],
                    json.dumps(frame()).encode()[20:]+b'\npartial-tail']
            expected=b''.join(chunks)
            class Link:
                def settimeout(self,_):pass
                def close(self):pass
                def recv(self,_):
                    chunk=chunks.pop(0)
                    if not chunks:feed.stop.set()
                    return chunk
            feed=imu.ImuFeed('192.168.1.9',Path(directory)/'feed',socket_factory=lambda *_args,**_kwargs:Link())
            feed.run()
            self.assertEqual(expected,(feed.output/'raw.bin').read_bytes())
            receipt=json.loads((feed.output/'receipt.json').read_text())
            self.assertEqual(imu.RECEIPT_CLOCK_DOMAIN,receipt['host_receipt_clock']['domain'])
            self.assertEqual(hashlib.sha256(expected).hexdigest(),receipt['raw_sha256'])
            rows=[json.loads(line) for line in (feed.output/'receipts.jsonl').read_text().splitlines()]
            valid=next(r for r in rows if r.get('valid_frame'))
            self.assertEqual(imu.RECEIPT_CLOCK_DOMAIN,valid['host_receipt_clock_domain'])
            self.assertGreaterEqual(feed.snapshot()['age_ms'],0)
            self.assertLess(feed.snapshot()['age_ms'],1000)
            original=expected[valid['raw_offset']:valid['raw_offset']+valid['raw_length']]
            self.assertEqual(frame(),json.loads(original))
            self.assertEqual('incomplete_connection_tail',rows[-1]['issue'])


if __name__=='__main__':unittest.main(verbosity=2)
