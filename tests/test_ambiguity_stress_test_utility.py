import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/'scripts/ambiguity_stress_test_utility.py'
S=importlib.util.spec_from_file_location('stress',P);m=importlib.util.module_from_spec(S);S.loader.exec_module(m)

def test_ridge_train_only_scaling_and_prediction():
 x=np.array([[0.],[1.],[2.],[3.]]);y=2*x[:,0]+1;w=np.ones(4)
 train,pred,fit=m.ridge_predict(x,y,w,np.array([[4.]]),alpha=0.)
 np.testing.assert_allclose(train,y,atol=1e-12)
 np.testing.assert_allclose(pred,[9.],atol=1e-12)
 assert fit['mean']==[1.5]

def test_work_means_equal_cases_then_rows():
 cases=[{'work':'a','rows':[{},{}]},{'work':'a','rows':[{}]},{'work':'b','rows':[{},{}]}]
 got=m.work_means(np.array([0.,2.,10.,4.,8.]),cases,['a','b'])
 np.testing.assert_allclose(got,[5.5,6.])

def test_capped_difference_sign():
 chroma,pitch=8.,2.
 assert min(chroma,m.CAP)-min(pitch,m.CAP)==3.
