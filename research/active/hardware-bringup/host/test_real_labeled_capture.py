"""Offline simple dorm tests; no COM, network, or real labels."""
from datetime import datetime, timedelta, timezone
import copy, hashlib, json, tempfile, unittest
from pathlib import Path
from capture import Decoder
import real_labeled_capture as guided
import real_labeled_annotation as annotation
REPO=Path(__file__).resolve().parents[4]
PAYLOAD=REPO/'artifacts.local/work/real-labeled-capture-simple-prep-20261011'
RAW=REPO/'artifacts.local/hardware-bringup/captures/cnh-h3-1mhz-20260927/raw.bin'
PLANE_RAW=REPO/'artifacts.local/hardware-bringup/captures/simple-20260927T164234Z-cc4d0e/01-background-attempt1/tof/raw.bin'
class FakeClock:
    def __init__(self): self.value=0.
    def sleep(self,s): self.value+=s
    def monotonic(self): return self.value
    def monotonic_ns(self): return int(self.value*1e9)
    def utcnow(self): return (datetime(2026,10,11,tzinfo=timezone.utc)+timedelta(seconds=self.value)).isoformat()
class Answers:
    def __init__(self,skip=None,rerecord=False): self.prompts=[]; self.ready=0; self.finished=0; self.skip=skip; self.rerecord=rerecord
    def __call__(self,prompt):
        self.prompts.append(prompt)
        if len(self.prompts)>200: raise RuntimeError('unhandled input '+prompt)
        if 's跳过' in prompt:
            self.ready+=1
            return 's' if self.ready==self.skip else 'optional-photo.jpg'
        if 'r重录' in prompt:
            self.finished+=1
            return 'r' if self.rerecord and self.finished==1 else ''
        return ''
class SimpleReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PAYLOAD.mkdir(parents=True,exist_ok=True)
        cls.workspace=tempfile.TemporaryDirectory(prefix='independent-simple-',dir=PAYLOAD)
        cls.root=Path(cls.workspace.name)
        cls.source_sha=hashlib.sha256(RAW.read_bytes()).hexdigest()
        cls.plane_sha=hashlib.sha256(PLANE_RAW.read_bytes()).hexdigest()
    @classmethod
    def tearDownClass(cls): cls.workspace._finalizer.detach()
    def make(self,name,answers=None,duration=1.,probe=.4,interrupt=None,source=None):
        clock,devices,console=FakeClock(),[],[]
        def factory():
            device=guided.ReplaySerial(source(len(devices)) if source else RAW,clock)
            if interrupt:
                original=device.read
                def read(size):
                    if clock.monotonic()>=interrupt: raise KeyboardInterrupt('injected replay interruption')
                    return original(size)
                device.read=read
            devices.append(device)
            return device
        app=guided.GuidedCapture(self.root/name,factory,clock=clock,input_fn=answers or Answers(),print_fn=console.append,segment_seconds=duration,probe_seconds=probe,replay=True)
        return app,devices,console
    def integrity(self,directory):
        meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        data=(directory/'raw.bin').read_bytes()
        self.assertEqual((len(data),hashlib.sha256(data).hexdigest()),(meta['raw_bytes'],meta['raw_sha256']))
        rows=[json.loads(x) for x in (directory/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        decoded=[]
        decoder=Decoder(lambda kind,row: decoded.append(row) if kind=='frames' else None)
        decoder.feed(data,0)
        self.assertEqual([r['sensor'] for r in decoded],[r['sensor'] for r in rows])
        self.assertEqual(len(rows),meta['summary']['frames'])
        self.assertTrue(all(r['host_received_monotonic_ns'] is not None and r['host_received_utc'] for r in rows))
        return meta,rows
    def test_day_night_all_eleven_book_isolation_and_reboot(self):
        answers=Answers(); app,devices,console=self.make('full',answers)
        day=app.run_session()
        self.assertTrue(day['day_complete']); self.assertFalse(day['night_complete']); self.assertEqual(9,len(day['attempts']))
        self.assertEqual(2,len(devices)); self.assertTrue(all(d.closed for d in devices))
        startup=json.loads((app.root/'startup-02/metadata.json').read_text(encoding='utf-8'))
        self.assertEqual('independent_unplug_restart_before_segment9',startup['reason'])
        self.assertIn('第9段独立重启','\n'.join(answers.prompts)); self.assertEqual('default_estimate',day['sensor_height_source']); self.assertEqual(1.2,day['sensor_height_m'])
        for a in day['attempts']:
            meta,capture=self.integrity(app.root/a['path']); self.assertEqual('optional-photo.jpg',meta['photo_optional'])
            _,probe=self.integrity(app.root/a['path']/'distance-probe-01')
            self.assertFalse({r['sensor']['seq'] for r in capture}&{r['sensor']['seq'] for r in probe})
            if 5<=a['condition_index']<=8:
                self.assertEqual('book_proxy',meta['distance_reference']['source_kind'])
                _,book=self.integrity(app.root/a['path']/'book-probe-02')
                self.assertFalse({r['sensor']['seq'] for r in capture}&{r['sensor']['seq'] for r in book})
        self.assertIn('拿走书','\n'.join(answers.prompts)); self.assertIn('3s',console)
        night,night_devices,_=self.make('full'); result=night.run_session(night=True)
        self.assertEqual('complete',result['status']); self.assertEqual(11,len(result['attempts'])); self.assertTrue(night_devices[0].closed)
        self.assertTrue((app.root/'inventory.tsv').is_file())
    def test_strict_startup_identity_layout_rate_loss(self):
        app,_,_=self.make('gate',Answers(skip=1)); result=app.run_session()
        summary=result['startups'][0]['summary']; events=[json.loads(x) for x in (app.root/'startup-01/events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(guided.check_preflight(events,summary)['passed']); self.assertFalse(guided.check_preflight([],summary)['passed'])
        bad=copy.deepcopy(events); bad[0]['sensor']['firmware']='wrong'; self.assertFalse(guided.check_preflight(bad,summary)['passed'])
        for key,value in (('frames',2),('cnh_layout_counts',{'4x4x24':62}),('observed_sensor_hz',3.81),('missing_sequences',20),('duplicate_sequences',1),('device_errors',1),('issues',{'invalid_frame':1})):
            self.assertFalse(guided.check_preflight(events,dict(summary,**{key:value}))['passed'],key)
        for event in bad:
            if event['sensor'].get('type')=='config': event['sensor']['bins']=24
        self.assertFalse(guided.check_preflight(bad,summary)['passed'])
    def test_two_second_probe_and_thirty_second_capture(self):
        app,_,_=self.make('windows'); app.device=guided.ReplaySerial(PLANE_RAW,app.clock)
        try:
            app.collect(app.root/'probe-2s',2.,{'phase':'distance_probe'}); ref=guided.summarize_distance(app.root/'probe-2s')
            self.assertTrue(ref['stable']); self.assertGreaterEqual(ref['frames_seen'],10); self.integrity(app.root/'probe-2s')
        finally: app.device.close()
        app.device=app.serial_factory()
        try: app.collect(app.root/'capture-30s',30.,{'scope':'recorded-byte-window'})
        finally: app.device.close()
        meta,_=self.integrity(app.root/'capture-30s'); self.assertEqual(154,meta['summary']['frames']); self.assertTrue(30<=meta['host_elapsed_seconds']<30.3)
        self.assertEqual(self.source_sha,hashlib.sha256(RAW.read_bytes()).hexdigest())
        self.assertEqual(self.plane_sha,hashlib.sha256(PLANE_RAW.read_bytes()).hexdigest())
    def test_skip_rerecord_interrupted_resume_preserve_attempts(self):
        app,_,_=self.make('repeat',Answers(skip=1,rerecord=True)); state=app.run_session()
        self.assertEqual('skipped',state['attempts'][0]['status']); self.assertEqual('superseded',state['attempts'][1]['status'])
        old=app.root/state['attempts'][1]['path']/'raw.bin'; original=old.read_bytes()
        redo,_,_=self.make('repeat'); state=redo.run_session(redo=['2']); self.assertEqual(original,old.read_bytes())
        self.assertEqual(3,len([a for a in state['attempts'] if a['condition_index']==2]))
        stopped,devices,_=self.make('interrupt',duration=3.,interrupt=16.)
        with self.assertRaises(KeyboardInterrupt): stopped.run_session()
        state=json.loads((stopped.root/'capture.json').read_text(encoding='utf-8')); self.assertEqual('interrupted',state['status']); self.assertEqual('interrupted',state['attempts'][0]['status'])
        self.integrity(stopped.root/state['attempts'][0]['path']); self.assertTrue(devices[0].closed)
        resumed,_,_=self.make('interrupt'); state=resumed.run_session(resume=True); self.assertEqual('interrupted',state['attempts'][0]['status']); self.assertTrue(state['day_complete'])
    def test_night_cross_date_root_and_incompatible_schema(self):
        work=self.root/'select-work'; original=guided.default_output_root(work,today='20261011'); original.mkdir(parents=True)
        (original/'capture.json').write_text(json.dumps({'protocol_version':guided.PROTOCOL_VERSION,'day_complete':True,'night_complete':False}),encoding='utf-8')
        incompatible=work/'real-labeled-capture-20261012'; incompatible.mkdir(); (incompatible/'capture.json').write_text('{"protocol_version":"old-complex"}',encoding='utf-8')
        self.assertEqual(original,guided.default_output_root(work,night=True,today='20261012'))
        (original/'capture.json').write_text(json.dumps({'protocol_version':guided.PROTOCOL_VERSION,'night_complete':True}),encoding='utf-8')
        with self.assertRaises(ValueError): guided.default_output_root(work,night=True,today='20261012')
    def test_segment9_reboot_gate_requires_new_identity(self):
        missing=self.root/'synthetic-missing-boot.bin'; missing.write_bytes(b''.join(line for line in RAW.read_bytes().splitlines(keepends=True) if b'"type":"boot"' not in line))
        app,devices,_=self.make('bad-reboot',source=lambda i: RAW if i==0 else missing)
        with self.assertRaises(ValueError): app.run_session(redo=['9'])
        gate=json.loads((app.root/'startup-02/gate.json').read_text(encoding='utf-8')); self.assertFalse(gate['passed']); self.integrity(app.root/'startup-02'); self.assertTrue(all(d.closed for d in devices))
    def test_distance_full_window_validity_and_quality_warnings(self):
        directory=self.root/'synthetic-probe'; directory.mkdir()
        original=next(json.loads(line) for line in RAW.read_text(encoding='utf-8').splitlines() if '"cnh_frame"' in line); frames=[]
        for distance in (.5,.5,2.,2.,2.):
            frame=copy.deepcopy(original); frame['distance_mm']=[int(distance*1000)]*64
            frame['target_status']=[5]*64; frame['nb_target']=[1]*64; frames.append({'sensor':frame})
        path=directory/'frames.jsonl'; path.write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8'); ref=guided.summarize_distance(directory)
        self.assertEqual(5,ref['frames_with_center']); self.assertFalse(ref['stable']); self.assertIsNone(ref['distance_m'])
        for frame in frames: frame['sensor']['target_status']=[0]*64
        path.write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8'); self.assertIsNone(guided.summarize_distance(directory)['distance_m'])
        app,_,_=self.make('quality'); summary=Decoder().summary(); summary.update(frames=10,missing_sequences=3,observed_sensor_hz=3.81,cnh_layout_counts={'8x8x16':10})
        warnings,_=app.quality_warnings({'summary':summary},{'distance_reference':{'stable':False}},directory); self.assertIn('丢帧>2%',warnings); self.assertIn('帧率异常',warnings)
        warnings,_=app.quality_warnings({'summary':summary},{'distance_reference':{'stable':True,'distance_m':1.}},directory)
        self.assertTrue(any('中央距离无法与准备读数核对' in warning for warning in warnings))
    def test_approximate_labels_are_condition_weak(self):
        base={'protocol_version':guided.PROTOCOL_VERSION,'sensor_height_m':1.2,'sensor_height_source':'default_estimate','tilt_deg':0,'tilt_source':'nominal_horizontal_assumption','status':'complete'}
        for index in (1,10):
            rows=annotation.approximate_labels(dict(base,condition_index=index,target_distance_m=None)); self.assertTrue(all(r['label']=='clear' and r['independent_ground_truth'] is False for r in rows.values()))
        for index in (4,9,11,5,6,7,8):
            rows=annotation.approximate_labels(dict(base,condition_index=index,target_distance_m=None)); self.assertTrue(all(r['label']=='UNKNOWN' for r in rows.values()),str(index))
        rows=annotation.approximate_labels(dict(base,condition_index=1,target_distance_m=None),attempt_status='needs_review'); self.assertTrue(all(r['label']=='UNKNOWN' for r in rows.values()))
        for index,condition,response in ((1,'corridor_empty','silent'),(4,'corridor_empty','silent'),(9,'corridor_empty','silent'),(10,'corridor_empty','silent'),(11,'corridor_empty','silent'),(2,'plane','calibration'),(5,'corridor_target_present','alarm'),(6,'corridor_target_present','alarm'),(8,'corridor_target_present','alarm'),(7,'outside_target','no_strong_alarm')):
            rows=annotation.approximate_labels(dict(base,condition_index=index,target_distance_m=None))
            self.assertTrue(all(r['condition_label']==condition and r['expected_response']==response for r in rows.values()))
        rows=annotation.approximate_labels(dict(base,condition_index=8,target_distance_m=1.2,target_height_m=1.3,rod_diameter_m=None))
        self.assertTrue(all(r['label']=='UNKNOWN' for r in rows.values()))
if __name__=='__main__': unittest.main(verbosity=2)
