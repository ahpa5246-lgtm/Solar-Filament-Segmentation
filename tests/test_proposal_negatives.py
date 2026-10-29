import numpy as np
from pycocotools import mask as coco

def test_actual_false_proposal_negative_excludes_all_known_annotation_context():
 from solar.proposal_negatives import safe_negative_feature
 image=np.full((128,128),120,np.uint8);proposal=np.zeros_like(image);proposal[45:60,45:60]=1
 encode=lambda a:coco.encode(np.asfortranarray(a))
 helpers={'square_bounds':lambda mask:(30,30,80,80)}
 protected=np.zeros_like(image);protected[72:75,72:75]=1
 assert safe_negative_feature(image,encode(proposal),protected,helpers) is None
 protected[:]=0
 feature=safe_negative_feature(image,encode(proposal),protected,helpers)
 assert feature.shape==(3,256,256) and feature.dtype==np.uint8
 assert feature[2].max()==255
