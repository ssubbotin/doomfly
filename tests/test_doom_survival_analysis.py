import json
from pathlib import Path

import pytest


PILOT=Path(__file__).resolve().parents[1]/'outputs/doom-learning/physiology-v6/survival-pilot'


def _cohort_fixture(out,seeds):
    """Relabel complete historical records to test cohort metadata only."""
    original=json.loads((PILOT/'results.json').read_text())
    cohort={**original,'protocol':{**original['protocol'],'training_seeds':seeds},
        'episodes':[]}
    out.mkdir()
    for seed in seeds:
        cohort['episodes'].extend({**episode,'replicate':seed}
            for episode in original['episodes'])
        for source in PILOT.glob('*/*/episode.json'):
            branch,episode,_=source.relative_to(PILOT).parts
            mode=branch.split('-',1)[1]
            target=out/f'{seed}-{mode}'/episode/'episode.json'
            target.parent.mkdir(parents=True)
            target.write_text(source.read_text())
    (out/'results.json').write_text(json.dumps(cohort))


@pytest.mark.parametrize('seeds',[[42051],[42051,42052,42053]])
def test_analysis_reports_actual_cohort_size_without_frame_pseudoreplication(tmp_path,seeds):
    from doom_learning_v6.analyze import analyze
    out=tmp_path/'cohort';_cohort_fixture(out,seeds)
    analyze(out)
    report=json.loads((out/'analysis.json').read_text())
    assert report['sampling']=={'training_replicates':len(seeds),
        'held_out_starts':2,'uncertainty_unit':'training replicate',
        'game_frames_are_independent_samples':False}
    assert 'One training replica' not in report['reason']
    assert report['survival_learning_demonstrated'] is False
    assert report['announcement_ready'] is False
