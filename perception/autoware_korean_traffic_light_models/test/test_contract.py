import unittest
import numpy as np
from autoware_korean_traffic_light_models.adapter_core import ROI, classify, decode, ELEMENTS, preprocess

class Backend:
    batch=6
    def __init__(self):self.calls=0
    def infer(self,x):
        self.calls+=1
        assert x.shape==(6,3,20,70) and x.dtype==np.float32
        return np.tile([0,0,1,0,0,0],(6,1)).astype(np.float32)

class Contract(unittest.TestCase):
    def setUp(self):
        self.r=ROI(100,0,0,0,70,20)
        self.image=np.random.default_rng(13).integers(0,256,(20,70,3),dtype=np.uint8)
    def test_six_class_mapping(self):
        for i in range(6):
            with self.subTest(i=i):
                out=decode(np.eye(6)[i],self.r)
                if i==5:self.assertEqual(out['reason'],'learned_reject')
                else:self.assertEqual([(e['color'],e['shape']) for e in out['elements']],list(ELEMENTS[i]))
    def test_invalid_probabilities(self):
        for p in [np.zeros(5),np.zeros(6),np.ones(6),np.full(6,np.nan),np.full(6,np.inf),np.array([-1,0,0,0,0,2])]:
            with self.subTest(p=p):self.assertEqual(decode(p,self.r)['label'],'unknown')
    def test_confidence_and_margin(self):
        self.assertEqual(decode([.79,.21,0,0,0,0],self.r)['label'],'unknown')
        self.assertEqual(decode([.8,.2,0,0,0,0],self.r)['label'],'red')
        self.assertEqual(decode([.8,.2,0,0,0,0],self.r,min_margin=.7)['label'],'unknown')
    def test_batch_and_duplicates(self):
        b=Backend();rs=[ROI(i,0,0,0,70,20) for i in range(7)]
        self.assertEqual(len(classify(self.image,rs,b)),7);self.assertEqual(b.calls,2)
        out=classify(self.image,[self.r,self.r],b)
        self.assertEqual(len(out),1);self.assertEqual(out[0]['reason'],'duplicate_ids')
    def test_invalid_rois_and_uniform(self):
        for r in [ROI(1,1,0,0,70,20),ROI(1,0,-1,0,70,20),ROI(1,0,0,0,2,20),ROI(1,0,0,0,71,20)]:
            self.assertEqual(classify(self.image,[r],Backend())[0]['label'],'unknown')
        self.assertEqual(classify(np.zeros_like(self.image),[self.r],Backend())[0]['reason'],'uniform_crop')
    def test_backend_error_abstains_entire_message(self):
        class Wrong(Backend):
            def infer(self,x):return np.zeros((6,5))
        self.assertEqual(classify(self.image,[self.r],Wrong())[0]['reason'],'inference_error')
    def test_preprocessing(self):
        out=preprocess(self.image)
        np.testing.assert_array_equal(out,self.image.transpose(2,0,1).astype(np.float32)/255)

if __name__=='__main__':unittest.main()
