"""Summarize completed rollouts separately from infrastructure exceptions."""
import collections
import json
import re
import sys
from pathlib import Path
root=Path(sys.argv[1])
counts=collections.defaultdict(lambda: [0,0])
results=root/'results.jsonl'
if results.exists():
    for line in results.read_text().splitlines():
        row=json.loads(line)
        key=(row['task'],row['mode'])
        counts[key][0]+=1
        counts[key][1]+=int(row['success'])
summary={}
for mode in sorted({mode for _,mode in counts}):
    jobs={task:{'episodes':n,'successes':s,'success_rate':s/n} for (task,m),(n,s) in counts.items() if m==mode}
    summary[mode]={'tasks':jobs,'completed_episodes':sum(v['episodes'] for v in jobs.values()),
                   'mean_task_success_rate':sum(v['success_rate'] for v in jobs.values())/len(jobs)}
errors=[]
for path in root.glob('worker*/run*.log'):
    for line in path.read_text(errors='replace').splitlines():
        if 'ROLLOUT_EXCEPTION' in line:
            errors.append({'log':str(path.relative_to(root)),'message':line})
summary['rollout_exception_attempts']=len(errors)
summary['note']='Rates use normally completed rollouts only; consult summary.tsv and strict exporter for completeness.'
(root/'aggregate.json').write_text(json.dumps(summary,indent=2))
(root/'exceptions.json').write_text(json.dumps(errors,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('demo_clean','demo_randomized')},indent=2))
for mode in ('demo_clean','demo_randomized'):
    if mode in summary: print(mode,summary[mode]['completed_episodes'],summary[mode]['mean_task_success_rate'])
