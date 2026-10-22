"""Bounded Week 9 release checks/benchmarks; never fit models or score final test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from src.hybrid import week7
from src.hybrid.diversity import rerank_for_diversity
from src.hybrid.recommender import Recommendation, recommend_for_user
from src.hybrid.group_recommender import recommend_group
from src.hybrid.release import load_release, RatingHistory, ServingSession, ServingMetrics
from src.hybrid.serving_cache import SERVING_CACHE, RankingCache
from src.scripts.week8_context import construct_groups, validation_labels


def summary(values):
    result=dict(n=len(values),median=statistics.median(values),minimum=min(values),maximum=max(values))
    # Explicit conservative sample floors; empirical quantiles, not SLA estimates.
    for label,q,floor in [('p90',.90,100),('p95',.95,200),('p99',.99,1000)]:
        result[label]=float(np.quantile(values,q)) if len(values)>=floor else None
    return result


def memory():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
                ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        api=ctypes.WinDLL('psapi');kernel=ctypes.WinDLL('kernel32')
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        row=Counters();row.cb=ctypes.sizeof(row)
        if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(row),row.cb):raise ctypes.WinError()
        return dict(rss_bytes=row.WorkingSetSize,peak_rss_bytes=row.PeakWorkingSetSize,method='Windows process working set')
    import resource
    factor=1 if sys.platform=='darwin' else 1024
    return dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor,method='getrusage process peak RSS')


def source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('reports/week9/benchmark.json'))
    parser.add_argument('--cold-runs',type=int,default=3)
    parser.add_argument('--warm-runs',type=int,default=30)
    parser.add_argument('--load-requests',type=int,default=200)
    args=parser.parse_args()
    if min(args.cold_runs,args.warm_runs,args.load_requests)<1:parser.error('Positive bounded counts required')
    if args.output.exists():parser.error('Refusing overwrite')
    root=Path.cwd();out=args.output.parent
    def save(report):args.output.with_suffix('.progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    s=perf_counter();bundle=load_release(root,trusted_manifest=root/'docs/week9-artifact-manifest.json');load=perf_counter()-s
    profiles=bundle['reference_ratings'].copy();movies=bundle['movies']
    specs,selection=construct_groups(profiles);by_name={name:(users,frame) for name,users,frame in specs}
    sparse,frame=by_name['sparse_member'];cold,cold_frame=by_name['cold_member']
    cases=[('existing',[23],profiles),('sparse',[sparse[-1]],frame),('cold',[cold[-1]],cold_frame)]
    cases += [(name,*by_name[name]) for name in ['similar_pair','one_outlier','conflicting_pair','larger_five']]
    old=source('week9_baseline',out/'baseline/week7.py')
    report=dict(scope='Frozen profile histories/full catalog, no final-test evaluation',artifact_load_seconds=load,
                environment=dict(python=sys.version,platform=sys.platform),selection=selection,cases=[],startup=[],memory_start=memory())
    code="import time,json; s=time.perf_counter(); import src.hybrid.week7 as w; i=time.perf_counter()-s; s=time.perf_counter(); w.load_bundle(); print(json.dumps(dict(imports=i,artifact=time.perf_counter()-s)))"
    for _ in range(3):
        s=perf_counter();child=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True,timeout=90)
        report['startup'].append(dict(process_wall=perf_counter()-s,**json.loads(child.stdout)))
    s=perf_counter();raw=pd.read_csv('data/ratings.csv');report['dataset_load_seconds']=perf_counter()-s
    # Audit only profile boundaries against the frozen bundle. No final-test slice scored.
    from src.hybrid.ml_reranker import chronological_split
    rebuilt=[]
    for _,rows in raw.groupby('userId',sort=False):
        if len(rows)>=10:rebuilt.append(chronological_split(rows,train_fraction=.10)[0])
    expected=pd.concat(rebuilt,ignore_index=True)
    pd.testing.assert_frame_equal(expected,profiles)
    del raw,expected,rebuilt
    report['profile_split_exact']=True
    def call(users,ratings,diagnostics=True):
        t={};s=perf_counter()
        if len(users)>1:result=recommend_group(users,ratings,movies,bundle=bundle,timings=t if diagnostics else None)
        else:
            rows=week7.recommend_week7(users[0],ratings,movies,50,bundle=bundle,timings=t if diagnostics else None)
            d=perf_counter();result=rerank_for_diversity([Recommendation(r.movie_id,r.title,r.genres,r.ml_score) for r in rows],10)
            if diagnostics:t['diversity']=perf_counter()-d
        return perf_counter()-s,result,t
    for name,users,ratings in cases:
        record=dict(case=name,members=len(users),phases=[],cold=[],warm=[],update=[],evicted=[])
        for _ in range(args.cold_runs):
            SERVING_CACHE.clear();seconds,reference,t=call(users,ratings)
            record['cold'].append(seconds);record['phases'].append(t)
        for _ in range(args.warm_runs):
            seconds,result,t=call(users,ratings)
            assert result==reference
            record['warm'].append(seconds)
        record['warm_phases_last']=t
        history=RatingHistory(ratings);metrics=ServingMetrics();session=ServingSession(bundle,history,metrics=metrics)
        for operation in ['add','edit','remove']:
            row=next(r for r in reference if r.movie_id not in set(ratings.loc[ratings.userId==users[0],'movieId']))
            change=dict(user_id=users[0],movie_id=row.movie_id,operation=operation)
            if operation!='remove':change.update(rating=4. if operation=='add' else 1.,timestamp=10000000000)
            s=perf_counter();session.update(**change);update_time=perf_counter()-s
            _,snapshot=history.snapshot();before=SERVING_CACHE.info();seconds,result,t=call(users,snapshot)
            # Complete fresh recomputation equals cached updated pipeline for all members.
            if len(users)==1:
                fresh=week7.recommend_week7(users[0],snapshot,movies,len(movies),bundle=bundle,use_cache=False)
                cached=week7.recommend_week7(users[0],snapshot,movies,len(movies),bundle=bundle)
                assert fresh==cached
            else:
                SERVING_CACHE.clear();assert call(users,snapshot)[1]==result
            record['update'].append(dict(operation=operation,helper_seconds=update_time,inference_seconds=seconds,
                                         before_cache=before,after_cache=SERVING_CACHE.info(),exact_fresh=True))
        for i in range(17):SERVING_CACHE.put(('week9-evict',i),((i,0.),))
        before=SERVING_CACHE.info();seconds,result,t=call(users,ratings)
        assert result==reference and SERVING_CACHE.info()['misses']-before['misses']==len(users)
        record['evicted'].append(seconds)
        if len(users)==1:
            baseline=old.recommend_week7(users[0],ratings,movies,len(movies),bundle=bundle,use_cache=False)
            current=week7.recommend_week7(users[0],ratings,movies,len(movies),bundle=bundle,use_cache=False)
            assert [asdict(r) for r in baseline]==[asdict(r) for r in current]
            old_diversity=rerank_for_diversity([Recommendation(r.movie_id,r.title,r.genres,r.ml_score) for r in baseline[:50]],10)
            assert old_diversity==reference==recommend_for_user(users[0],ratings,movies,week7_bundle=bundle)
            record['complete_frozen_scores_and_diversity_exact']=True
        for key in ['cold','warm','evicted']:record[key]=summary(record[key])
        record['cache']=SERVING_CACHE.info();record['metrics']=metrics.snapshot()
        report['cases'].append(record);save(report)
        print(name,record['cold']['median'],record['warm']['median'],flush=True)
    # Bounded real concurrent individual requests; no monkeypatches in timed path.
    SERVING_CACHE.clear();session=ServingSession(bundle,RatingHistory(profiles))
    reference=session.individual(23)
    def request(_):
        s=perf_counter();result=session.individual(23);assert result==reference;return perf_counter()-s
    s=perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:latencies=list(pool.map(request,range(args.load_requests)))
    elapsed=perf_counter()-s
    report['concurrency']=dict(requests=args.load_requests,threads=4,seconds=elapsed,throughput=args.load_requests/elapsed,
                               latency=summary(latencies),cache=SERVING_CACHE.info(),exact=True)
    bounded=RankingCache()
    for i in range(20):bounded.put(i,tuple((j,float(j)) for j in range(12500)))
    report['bounded_cache_fill']=bounded.info();report['memory_peak']=memory()
    assert bounded.info()['candidates']==200000 and bounded.info()['entries']==16
    # Validation-only deterministic repeat checks on a declared small cohort.
    labels=validation_labels(bundle);checks=[]
    for uid in [23,69,84]:
        subset=movies.loc[movies.movieId.isin(labels.loc[labels.userId==uid,'movieId'])]
        a=week7.score_week7_candidates(uid,profiles,subset,bundle=bundle,use_cache=False)
        b=week7.score_week7_candidates(uid,profiles,subset,bundle=bundle,use_cache=False)
        assert a==b
        checks.append(dict(user=uid,candidates=len(a),exact=True))
    report['validation_reproducibility']=dict(scope='Three prespecified validation-only candidate cohorts; not a historical metric reproduction',checks=checks)
    report['artifact_sha256']=week7.file_digest(week7.DEFAULT_BUNDLE)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
