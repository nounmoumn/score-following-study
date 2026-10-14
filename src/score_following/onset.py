"""New sustained failures, separate from recovery and exposure-period failures."""
import re
import numpy as np


def work_family(folder):
    parts=folder.split('/')
    parts[-1]=re.sub(r'_(?:no_.*repeat|no_trio|repeat)$','',parts[-1])
    if parts[0]=='Bach' and parts[1] in ['Fugue','Prelude']:
        return 'Bach/'+parts[2]
    if len(parts)>=3 and parts[1] in ['Piano_Sonatas','Keyboard_Sonatas','Sonatas']:
        return '/'.join(parts[:2]+[parts[2].split('-')[0]])
    if len(parts)>=3 and (parts[1].startswith('Sonata_') or parts[1] in ['Kreisleriana','Gaspard_de_la_Nuit','Miroirs','Images_Book_1','Pour_le_Piano']):
        return '/'.join(parts[:2])
    return '/'.join(parts)


def onset_outcome(error,start,exposure_frames=20,horizon_frames=50,threshold=.5,dwell=5):
    """Eligibility uses history only; exposure failure is retained as a distinct label."""
    error=np.asarray(error);end=start+exposure_frames-1
    healthy=start>=20 and bool(np.all(error[start-20:start]<=threshold))
    if not healthy:return dict(eligible=False,status='pre_unhealthy',failure_frame=None)
    # Any threshold crossing during exposure loses the clean post-exposure risk set.
    if np.any(error[start:end+1]>threshold):
        return dict(eligible=True,status='exposure_crossing',failure_frame=None)
    stop=end+horizon_frames
    if stop>=len(error):raise ValueError('incomplete follow-up')
    for j in range(end+1,stop-dwell+2):
        if np.all(error[j:j+dwell]>threshold):return dict(eligible=True,status='new_failure',failure_frame=j)
    return dict(eligible=True,status='no_failure',failure_frame=None)
