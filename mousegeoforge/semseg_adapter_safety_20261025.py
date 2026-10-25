"""Prospective TRAIN-only pair stop rule; never a convergence or DEV selection gate."""
import math
STRATA=[('real',True),('real',False),('synthetic',True),('synthetic',False)]
def grouped(rows):
 assert len(rows)==256 and len({r['id'] for r in rows})==256
 groups={'all':rows}
 for d,b in STRATA:
  group=[r for r in rows if (r['domain'],r['bottom'])==(d,b)];assert len(group)==64;groups[f'{d}/{b}']=group
 result={}
 for k,g in groups.items():
  values={metric:sum(r['native'][metric] for r in g)/len(g) for metric in ['iou','dice','boundary_f1']};values['loss']=sum(r['loss'] for r in g)/len(g);assert all(math.isfinite(v) and v>=0 for v in values.values());result[k]=values
 return result
def violations(current,baseline):
 assert {r['id'] for r in current}=={r['id'] for r in baseline}
 a,b=grouped(current),grouped(baseline);failures=[]
 for group in a:
  for metric,tol in [('iou',.01),('dice',.01),('boundary_f1',.02)]:
   if a[group][metric]<b[group][metric]-tol:failures.append(group+'/'+metric)
  if a[group]['loss']>max(2*b[group]['loss'],b[group]['loss']+.02):failures.append(group+'/loss')
 return failures
def persistent_stop(current_failures,previous_failures):
 return bool(set(current_failures)&set(previous_failures))
