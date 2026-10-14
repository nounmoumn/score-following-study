import numpy as np
from score_following.onset import work_family,onset_outcome


def test_work_family_movement_repeat_and_bach_pair():
    assert work_family('Beethoven/Piano_Sonatas/17-1_no_repeat')==work_family('Beethoven/Piano_Sonatas/17-3')
    assert work_family('Bach/Fugue/bwv_846')==work_family('Bach/Prelude/bwv_846')
    assert work_family('Chopin/Sonata_2/2nd_no_repeat')=='Chopin/Sonata_2'
    assert work_family('Schumann/Toccata_repeat')=='Schumann/Toccata'


def test_new_failure_excludes_old_and_exposure_crossing():
    e=np.zeros(120);e[60:65]=1
    assert onset_outcome(e,30)['status']=='new_failure'
    e[20]=1;assert onset_outcome(e,30)['status']=='pre_unhealthy'
    e[20]=0;e[40]=1;assert onset_outcome(e,30)['status']=='exposure_crossing'


def test_dwell_and_future_boundary():
    e=np.zeros(120);e[60:64]=1
    assert onset_outcome(e,30)['status']=='no_failure'
    e[95:100]=1
    assert onset_outcome(e,30)['failure_frame']==95
    e[95]=0;assert onset_outcome(e,30)['status']=='no_failure'
