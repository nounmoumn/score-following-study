"""Small independent known-answer controls for the exploratory alignment helper."""
from p3_smd_alignment_pilot import align, events

def main():
    score=[(float(i),{p}) for i,p in enumerate((60,62,64,65,67))]
    _,mapping,_=align(score,score)
    assert mapping=={i:i for i in range(5)},mapping
    warped=[(t*t+3,p) for t,p in score]
    _,mapping,_=align(score,warped)
    assert mapping=={i:i for i in range(5)},mapping
    inserted=score[:2]+[(1.5,{99})]+score[2:]
    _,mapping,_=align(score,inserted)
    assert all(mapping[j]==j-1 for j in range(3,6)),mapping
    octave=[(t,{p+12 for p in pitches}) for t,pitches in score]
    _,mapping,_=align(score,octave,chroma=True)
    assert mapping=={i:i for i in range(5)},mapping
    grouped=events([{'time':t,'pitch':p} for t,p in ((0,60),(.04,64),(.08,67))], 'time',.045)
    assert grouped==[(0.0,{60,64}),(.08,{67})],grouped
    print('5 known-answer alignment controls passed')

if __name__=='__main__': main()
