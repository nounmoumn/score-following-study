import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from p3_short_signal import sustained_start,summarize


def test_sustained_onset_may_confirm_after_two_second_window():
    error=np.zeros(50);error[19:24]=1
    assert sustained_start(error,0,20)==19
    error[:]=0;error[20:25]=1
    assert sustained_start(error,0,20) is None


def test_transient_is_not_sustained_and_alarm_table_uses_same_population():
    error=np.zeros(50);error[5:9]=1
    assert sustained_start(error,0,20) is None
    rows=[dict(any_positive=True,sustained=True),dict(any_positive=True,sustained=False),
          dict(any_positive=False,sustained=False)]
    result=summarize(rows,'any_positive')
    assert (result['tp'],result['fp'],result['fn'],result['tn'])==(1,1,0,1)
    assert result['false_alarm_rate']==.5
