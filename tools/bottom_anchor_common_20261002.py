"""Separate cycle state/budget; old experiments and STOP remain untouched."""
import datetime as dt
import fcntl
import json
from pathlib import Path
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/bottom_anchor_cycle_20261002'
PRIOR=ROOT/'output/teacher_temporal_cycle_20261001'
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'

def exclusions(plan):
    """All pending review groups, including newly flagged groups; never infer labels."""
    entries=json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']
    entries=entries+[plan['additional_exclusion']]+plan.get('additional_review_exclusions',[])
    return {(r['session'],int(r['frame'])) for r in entries}

def write_state(**values):
    with (OUT/'state.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=OUT/'state.json';state=json.loads(path.read_text()) if path.exists() else {}
        state.update(values);state['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        temp=OUT/f'.state-{uuid.uuid4().hex}.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(path)

def budget(seconds=1200):
    plan=json.loads((OUT/'plan.json').read_text())
    end=min(time.time()+seconds,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():
        assert time.time()<end and not (OUT/'STOP').exists(),'Cycle/job budget reached or STOP requested'
    check()
    return plan,end,check
