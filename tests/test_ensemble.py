import sys
from pathlib import Path
import unittest
import tempfile
import hashlib
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmark'))
from bench_ensemble import fuse, evaluate, make_review, score
from bench_core import save_json, read_json


class EnsembleTests(unittest.TestCase):
    def test_review_export_score_and_reject_unconfirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save_json(root/'run.json',dict(selected_models=['a','b'],settings=dict(max_det=300,conf=.25)))
            save_json(root/'dataset.json',[dict(id='day_east_f1',group='day',camera='east',frame=1)])
            image=np.zeros((20,20,3),np.uint8)
            for model in ('a','b'):
                folder=root/'ensemble-review'/model/'pass_1'/'day_east_f1'
                save_json(folder/'detections.json',dict(frame_sha256=hashlib.sha256(image.data).hexdigest(),width=20,height=20,boxes=[[1,1,10,10,.9,0]]))
                cv2.imwrite(str(folder/'source.png'),image)
            make_review(root)
            bundle=read_json(root/'ensemble.json')
            save_json(root/'reviewed.json',bundle)
            with self.assertRaises(ValueError):
                score(root,root/'reviewed.json')
            bundle['status']='human_reviewed'
            bundle['frames'][0]['reviewed']=True
            save_json(root/'reviewed.json',bundle)
            score(root,root/'reviewed.json')
            rows=read_json(root/'accuracy.json')['metrics']
            self.assertTrue(all(r['mAP50_95']==1 for r in rows if r['label']=='all'))

    def test_fuse_preserves_singletons_and_classes(self):
        proposals=fuse({'a':[[0,0,10,10,.9,0],[0,0,10,10,.8,1]],'b':[[1,1,11,11,.8,0],[30,30,40,40,.7,0]]})
        self.assertEqual(len(proposals),3)
        self.assertEqual(proposals[0]['models'],['a','b'])

    def test_same_model_cannot_vote_twice(self):
        self.assertEqual(len(fuse({'a':[[0,0,10,10,.9,0],[0,0,10,10,.8,0]]})),2)

    def test_perfect_and_duplicate_predictions(self):
        truth={'img':[dict(box=[0,0,10,10],class_id=0)]}
        r=evaluate({'img':[[0,0,10,10,.9,0],[0,0,10,10,.8,0]]},truth,0,.5)
        self.assertEqual(r['ap'],1)
        self.assertEqual(r['precision'],.5)
        self.assertEqual(r['fp'],1)

    def test_missing_and_wrong_class(self):
        truth={'img':[dict(box=[0,0,10,10],class_id=0)]}
        self.assertEqual(evaluate({'img':[[0,0,10,10,.9,1]]},truth,0,.5)['ap'],0)
        self.assertIsNone(evaluate({'img':[]},truth,1,.5)['ap'])
