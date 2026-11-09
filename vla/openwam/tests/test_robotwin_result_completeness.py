import json
import pytest
from benchmarks.robotwin.export_results_csv import export_from_results_jsonl

@pytest.mark.parametrize('case',['complete','missing','duplicate','unexpected'])
def test_strict_episode_counts(tmp_path,case):
    records=[dict(task='a',mode='demo_clean',seed=s,success=True) for s in range(20)]
    if case=='missing':records.pop()
    if case=='duplicate':records.append(records[0])
    if case=='unexpected':records[0]['task']='b'
    (tmp_path/'results.jsonl').write_text('\n'.join(map(json.dumps,records)))
    result=export_from_results_jsonl(tmp_path,tmp_path/'results.csv',
        dict(tasks='a',mode='demo_clean',test_num='20'),True)
    assert (result==0)==(case=='complete')
