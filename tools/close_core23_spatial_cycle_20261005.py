"""Record verified early closure after app automation pause, without changing models."""
import datetime as dt,fcntl,json,shutil,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'cycle_handoff';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')

def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    auto=tomllib.loads(Path('/mnt/c/Users/55395/.codex/automations/pymousestudio/automation.toml').read_text());assert auto['status']=='PAUSED'
    plan=json.loads((CYCLE/'plan.json').read_text());lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for pid in [3761322,3761673,3761988,3762142]:assert not Path(f'/proc/{pid}').exists(),pid
    tests=json.loads((OUT/'tests.json').read_text());assert tests['tests']==53 and tests['failures']==tests['errors']==0
    now=dt.datetime.now(dt.timezone.utc);start=dt.datetime.fromisoformat(plan['start_utc'].replace('Z','+00:00'))
    status=dict(stage='complete_early_no_production_promotion',completed_utc=now.isoformat(),elapsed_hours=(now-start).total_seconds()/3600,max_authorized_hours=16,automation_id=auto['id'],automation_status=auto['status'],own_gpu_jobs_exited=True,gpu_lock_free=True,tests=53,production_changed=False,labels_changed=False,visual_review=dict(camA='Full candidate has two overlapping confident queries and erroneous foot positions on torso. Reversed input looks stable, not necessarily correct.',camD='Forepaw near head remains; detector changes do not repair anatomy.',synthetic_dual='Two confident boxes retained in fixed first dual scene. Full-frame subjects are small, so this overview cannot establish fine paw accuracy; quantitative errors and original predictions remain required.',reviewed=['review_detection_full_context_pose_v1/00_real_camA_full.jpg','review_detection_full_context_pose_v1/03_real_camD_full.jpg','review_detection_full_context_pose_v1/08_synthetic_camC_full.jpg','review_full_detector_full_order_v1/00_real_order.jpg','review_full_detector_full_order_v1/03_synthetic_order.jpg','cycle_handoff/synthetic_dual_all_views.jpg']))
    write(OUT/'completion.json',status);shutil.copy2(OUT/'completion.json',DEST/'cycle_handoff/completion.json')
    for arm in ['paired','full']:
        path=CYCLE/f'review_full_detector_{arm}_order_v1/results.json';r=json.loads(path.read_text());assert r['original_order_replay_cases']==0;r.update(max_original_order_delta=None,replay_scope='Not applicable: changed detector weights. Actual candidate cached/RGB equality checked by validator, not old checkpoint equality.');write(path,r);shutil.copy2(path,DEST/path.parent.name/path.name)
    next_action='This bounded cycle is CLOSED EARLY with evidence, no further autonomous jobs. Existing pymousestudio automation verified PAUSED. 53 focused tests and final RGB gates passed, ownPIDs exited, GPU lock free. Deliver cycle_handoff/index.html and docs/pymousestudio_core23_spatial_cycle_results_20261005.md. Do not rerun experiments, restart automation or promote models without a new user direction. Recommended future hypotheses are documented, not implemented promises.'
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage=status['stage'],updated_utc=now.isoformat(),completed_utc=now.isoformat(),active_job=None,active_pid=None,next=next_action,automation_status='PAUSED',elapsed_hours=status['elapsed_hours']);state['completed'].append('final_visual_review_own_jobs_exited_and_automation_paused');write(CYCLE/'state.json',state);plan.update(next_command=next_action,status='complete_early',completed_utc=now.isoformat());write(CYCLE/'plan.json',plan)
    for base in [OUT,DEST/'cycle_handoff']:
        page=base/'index.html';s=page.read_text();s=s.replace('<h2>主要发现</h2>',f'<p><strong>已提前结束：</strong>{now.strftime("%Y-%m-%d %H:%M UTC")}，累计约{status["elapsed_hours"]:.2f}小时。自动循环已暂停，自有GPU作业已退出。<a href="completion.json">结束核验</a></p><h2>主要发现</h2>');page.write_text(s,encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False))

if __name__=='__main__':main()
