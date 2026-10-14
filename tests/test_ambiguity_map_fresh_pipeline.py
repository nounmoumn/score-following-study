import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'scripts/ambiguity_map_fresh_pipeline.py';S=importlib.util.spec_from_file_location('fresh',P);m=importlib.util.module_from_spec(S);S.loader.exec_module(m)
def test_fit_predict_linear_unregularized_shape():
 x=np.arange(5.)[:,None];y=1+2*x[:,0];model=m.fit(x,y,np.ones(5));p=m.predict(x,model);assert p.shape==(5,);assert np.isfinite(p).all()
def test_work_equal_case_equal_mean():
 c=[{'work':'a','rows':[{},{}]},{'work':'a','rows':[{}]},{'work':'b','rows':[{},{}]}];np.testing.assert_allclose(m.wmean(np.array([0,2,10,4,8]),c,['a','b']),[5.5,6])
def test_pair_order_is_frozen():assert m.PAIRS==(('chroma','pitch'),('pitch','fused'),('chroma','fused'))
