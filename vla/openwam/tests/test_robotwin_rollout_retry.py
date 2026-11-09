"""Infrastructure exceptions must neither consume a new scene nor count as failure."""
from types import SimpleNamespace
import pytest
from benchmarks.robotwin.episode_worker import _make_dispatcher_eval_policy

@pytest.mark.parametrize('failures',[1,3])
def test_retry_same_seed_without_false_result(failures):
    seeds=[]; reports=[]; calls=[]
    class Env:
        eval_video_path=None
        render_freq=0
        step_lim=1
        plan_success=True
        def setup_demo(self,seed,**kwargs):
            seeds.append(seed); self.take_action_cnt=0; self.eval_success=False
        def play_once(self): return {'info':{}}
        def check_success(self): return True
        def close_env(self,**kwargs): pass
        def set_instruction(self,**kwargs): pass
        def get_obs(self): return {}
    def rollout(env,*args):
        calls.append(1)
        if len(calls)<=failures: raise RuntimeError('temporary transport failure')
        env.take_action_cnt=1;env.eval_success=True
    module=SimpleNamespace(UnStableError=type('UnStableError',(Exception,),{}),
        eval_function_decorator=lambda p,f:rollout if f=='eval' else lambda m:None,
        generate_episode_descriptions=lambda *args:[{'seen':['instruction']}])
    seed_requests=iter([{'seed':100000},{'drain':True}])
    client=SimpleNamespace(request_seed=lambda:next(seed_requests),request_commit=lambda s:{'commit':True},
        report_result=lambda s,**kw:reports.append((s,kw)),heartbeat=lambda:None)
    run=_make_dispatcher_eval_policy(module,client)
    args=dict(task_name='test',task_config='demo_clean',policy_name='test',clear_cache_freq=5,render_freq=0)
    if failures==3:
        with pytest.raises(RuntimeError):run('test',Env(),args,None,100000,instruction_type='seen')
        assert reports==[]
    else:
        run('test',Env(),args,None,100000,instruction_type='seen')
        assert len(reports)==1 and reports[0][1]['success']
    assert set(seeds)=={100000}
    assert len(calls)==min(failures+1,3)
