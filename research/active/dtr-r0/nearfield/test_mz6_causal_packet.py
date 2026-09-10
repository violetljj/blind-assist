"""Focused causal/history and warning-episode tests for the new MZ6 mechanism."""
import unittest
from unittest.mock import patch
import numpy as np
import cv2

from mz6_causal_packet import CausalPacket, mean3, correspondences
from mz6_sequence_evaluate import episodes, compare_episodes


class CausalTests(unittest.TestCase):
    def test_observed_translation_and_textureless_abstention(self):
        rng=np.random.default_rng(9)
        image=rng.integers(0,256,(180,320),dtype=np.uint8)
        moved=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,0]]),(320,180))
        matches=correspondences(image,moved)
        self.assertGreater(matches.sum(),30)
        self.assertGreater(matches.sum(),correspondences(image,moved,True).sum())
        self.assertEqual(correspondences(np.zeros_like(image),moved).sum(),0)

    def test_raw_history_expires_and_resets(self):
        gray=np.zeros((180,320),np.uint8)
        ranges=np.zeros((64,2));valid=np.zeros((64,2),bool)
        ranges[0]=[1.,3.];valid[0]=True
        single=ranges.copy();single[0]=[3.,0.]
        one=valid.copy();one[0]=[True,False]
        matches=np.zeros((64,64),int);matches[0,0]=3
        state=CausalPacket()
        with patch('mz6_causal_packet.correspondences',return_value=matches):
            state.step('a',0,gray,ranges,valid)
            for index in (1,2):
                r,v,c=state.step('a',index,gray,single,one)
                self.assertEqual(len(c),1);self.assertEqual(r[0,0],1.)
            r,v,c=state.step('a',3,gray,single,one)
            self.assertEqual(c,[]);np.testing.assert_array_equal(v,one)
            state.step('a',4,gray,ranges,valid)
            self.assertEqual(state.step('b',0,gray,single,one)[2],[])

    def test_flow_and_background_are_both_required(self):
        gray=np.zeros((180,320),np.uint8)
        ranges=np.zeros((64,2));valid=np.zeros((64,2),bool)
        ranges[0]=[1.,3.];valid[0]=True
        single=np.zeros_like(ranges);one=np.zeros_like(valid)
        single[1]=[2.,0.];one[1,0]=True
        match=np.zeros((64,64),int);match[0,1]=3
        state=CausalPacket();state.step('a',0,gray,ranges,valid)
        with patch('mz6_causal_packet.correspondences',return_value=match):
            self.assertEqual(state.step('a',1,gray,single,one)[2],[])
        state=CausalPacket();state.step('a',0,gray,ranges,valid)
        single[1,0]=3.
        with patch('mz6_causal_packet.correspondences',return_value=np.zeros_like(match)):
            self.assertEqual(state.step('a',1,gray,single,one)[2],[])
        state=CausalPacket();state.step('a',0,gray,ranges,valid)
        with patch('mz6_causal_packet.correspondences',return_value=match):
            self.assertEqual(len(state.step('a',1,gray,single,one)[2]),1)

    def test_mean_is_causal_and_clip_local(self):
        logits=np.arange(24,dtype=np.float32).reshape(6,4)
        clips=np.array(['a']*4+['b']*2)
        expected=mean3(logits,clips)
        np.testing.assert_array_equal(mean3(logits[:3],clips[:3]),expected[:3])
        changed=logits.copy();changed[3:]=-999
        np.testing.assert_array_equal(mean3(changed,clips)[:3],expected[:3])
        np.testing.assert_array_equal(expected[4],logits[4])

    def test_episode_censor_and_union(self):
        truth=np.zeros((7,4),bool);truth[1:3,1]=True;truth[3:5,0]=True
        delayed=np.zeros_like(truth);delayed[2:4,1]=True;delayed[4:6,0]=True
        clips=np.array(['a']*7)
        perfect=episodes(truth,truth,clips);candidate=episodes(delayed,truth,clips)
        union=next(r for r in candidate if r['event']=='BODY_ANY')
        self.assertEqual((union['start'],union['end_exclusive'],union['delay_frames']),(1,5,1))
        self.assertEqual(union['clearance_status'],'UNCLEARED')
        self.assertEqual(compare_episodes(candidate,perfect)['delayed_or_lost_episodes'],3)


if __name__=='__main__':unittest.main()
