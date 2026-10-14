import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'scripts/p3_review_ingest.py'
S=importlib.util.spec_from_file_location('p3_review_ingest',P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_complete_and_independent_schema():
    row=dict(review_id='x',reviewer_id='A',score_position_before='1',score_position_after='2',
             judgment='ok',evidence='notes',reviewed_at='now',status='complete')
    assert M.validate([row],'A')==[]
    row['reviewer_id']='B';assert any('reviewer_id' in x for x in M.validate([row],'A'))
def test_rejects_missing_decreasing_and_pending():
    row=dict(review_id='x',reviewer_id='A',score_position_before='2',score_position_after='1',
             judgment='',evidence='',reviewed_at='',status='pending')
    errors=M.validate([row],'A')
    assert any('decreases' in x for x in errors)
    assert any('status' in x for x in errors)
    assert any('evidence' in x for x in errors)
