"""Full-catalog lifecycle, memory and timing diagnostics; no model fitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
import tracemalloc
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.hybrid import week7, recommender
from src.hybrid.group_recommender import recommend_group
from src.hybrid.rating_updates import update_rating
from src.hybrid.serving_cache import SERVING_CACHE, RankingCache
from src.scripts.week8_context import construct_groups


def summary(samples):
    return dict(samples=samples,median=statistics.median(samples),minimum=min(samples),maximum=max(samples),
                population_std=statistics.pstdev(samples))


def load_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('reports/week8/freeze/diagnostics.json'))
    args=parser.parse_args()
    if args.runs<2: parser.error('At least two repetitions required')
    if args.output.exists(): parser.error('Refusing overwrite')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    start=perf_counter();bundle=week7.load_bundle();load_time=perf_counter()-start
    profiles=bundle['reference_ratings'].copy();movies=bundle['movies']
    specs,selection=construct_groups(profiles)
    by_name={name:(users,frame) for name,users,frame in specs}
    existing=23 if 23 in set(profiles.userId) else int(profiles.userId.iloc[0])
    sparse_users,sparse_frame=by_name['sparse_member']; cold_users,cold_frame=by_name['cold_member']
    cases=[('existing',[existing],profiles),('sparse',[sparse_users[-1]],sparse_frame),
           ('cold',[cold_users[-1]],cold_frame)]
    cases += [(name,*by_name[name]) for name in ('similar_pair','one_outlier','conflicting_pair','larger_five')]
    original=load_source('week8_before_finalization',args.output.parent/'baseline/week7.py')
    report=dict(runs=args.runs,scope='Frozen profile histories; full frozen catalog; no final-test scoring',
        bundle_load_seconds=load_time,cases=[],startup=[],
        catalog=dict(rows=len(movies),unique=int(movies.movieId.nunique()),duplicates=int(movies.movieId.duplicated().sum()),
                     missing_title=int(movies.title.isna().sum()),missing_genres=int(movies.genres.isna().sum())))
    child="import time,json; s=time.perf_counter(); import src.hybrid.week7 as w; i=time.perf_counter()-s; s=time.perf_counter(); w.load_bundle(); print(json.dumps(dict(imports=i,bundle=time.perf_counter()-s)))"
    for _ in range(args.runs):
        start=perf_counter();p=subprocess.run([sys.executable,'-c',child],capture_output=True,text=True,check=True)
        report['startup'].append(dict(wall=perf_counter()-start,**json.loads(p.stdout)))

    def call(users,frame):
        t={};start=perf_counter()
        if len(users)>1:
            result=recommend_group(users,frame,movies,bundle=bundle,timings=t)
        else:
            original_fn=week7.recommend_week7
            def measured(*a,**kw): return original_fn(*a,**kw,timings=t)
            diversity_fn=recommender.rerank_for_diversity
            def diversity(*a,**kw):
                d=perf_counter();result=diversity_fn(*a,**kw);t['diversity']=perf_counter()-d;return result
            with patch.object(week7,'recommend_week7',measured),patch.object(recommender,'rerank_for_diversity',diversity):
                result=recommender.recommend_for_user(users[0],frame,movies,10,week7_bundle=bundle)
        return perf_counter()-start,result,t

    for name,users,frame in cases:
        measurements={phase:[] for phase in ('cold','warm','modified','evicted')};phases={p:[] for p in measurements}
        for _ in range(args.runs):
            SERVING_CACHE.clear()
            seconds,first,t=call(users,frame);measurements['cold'].append(seconds);phases['cold'].append(t)
            before=SERVING_CACHE.info();seconds,repeat,t=call(users,frame)
            assert first==repeat and SERVING_CACHE.info()['hits']-before['hits']==len(users)
            measurements['warm'].append(seconds);phases['warm'].append(t)
            uid=users[0];h=frame.loc[frame.userId==uid]
            if h.empty:
                revised=update_rating(frame,user_id=uid,movie_id=first[0].movie_id,operation='add',rating=5.,timestamp=1)
            else:
                row=h.iloc[0]
                revised=update_rating(frame,user_id=uid,movie_id=int(row.movieId),operation='edit',rating=.5 if row.rating!=.5 else 5.)
            seconds,updated,t=call(users,revised);measurements['modified'].append(seconds);phases['modified'].append(t)
            # Fill the existing bounded LRU with cheap diagnostic entries to
            # force eviction without changing capacity or scoring algorithms.
            for i in range(SERVING_CACHE.max_entries+1): SERVING_CACHE.put(('diagnostic',i),((i,0.),))
            before=SERVING_CACHE.info();seconds,evicted,t=call(users,frame)
            assert evicted==first and SERVING_CACHE.info()['misses']-before['misses']==len(users)
            measurements['evicted'].append(seconds);phases['evicted'].append(t)
        record=dict(case=name,users=users,latency={p:summary(v) for p,v in measurements.items()},
                    phases=phases,cache_after=SERVING_CACHE.info())
        if len(users)==1:
            baseline=original.recommend_week7(users[0],frame,movies,len(movies),bundle=bundle,use_cache=False)
            current=week7.recommend_week7(users[0],frame,movies,len(movies),bundle=bundle,use_cache=False)
            assert [asdict(r) for r in baseline]==[asdict(r) for r in current]
            with patch.object(week7,'recommend_week7',original.recommend_week7):
                old=recommender.recommend_for_user(users[0],frame,movies,10,week7_bundle=bundle)
            assert old==call(users,frame)[1]
            record['exact_full_score_and_diversity_preservation']=True
            # Comparable original/current warm timings, same profile/catalog.
            baseline_warm=[];current_warm=[]
            for _ in range(args.runs):
                s=perf_counter();original.recommend_week7(users[0],frame,movies,50,bundle=bundle);baseline_warm.append(perf_counter()-s)
                s=perf_counter();week7.recommend_week7(users[0],frame,movies,50,bundle=bundle);current_warm.append(perf_counter()-s)
            record['comparable_score_prefix_warm']=dict(before=summary(baseline_warm),after=summary(current_warm))
        report['cases'].append(record)
        args.output.with_suffix('.progress.json').write_text(json.dumps(report,indent=2))
        print(name,{p:round(v['median'],3) for p,v in record['latency'].items()},flush=True)
    # Real serving concurrency: immutable inputs, all requests share one key.
    SERVING_CACHE.clear();reference=call([existing],profiles)[1]
    start=perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent=list(pool.map(lambda _: week7.recommend_week7(existing,profiles,movies,50,bundle=bundle),range(8)))
    assert all(rows==concurrent[0] for rows in concurrent)
    report['concurrency']=dict(requests=8,threads=4,seconds=perf_counter()-start,exact_equality=True,cache=SERVING_CACHE.info())
    # Memory is measured separately to avoid tracemalloc distorting latency.
    tracemalloc.start();bounded=RankingCache();base=tracemalloc.get_traced_memory()[0]
    for i in range(20): bounded.put(str(i),tuple((j,float(j)) for j in range(12500)))
    live,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert bounded.info()['candidates']<=200000 and bounded.info()['entries']<=16
    report['memory']=dict(cache=bounded.info(),incremental_live_bytes=live-base,peak_bytes=peak,
        methodology='Separate synthetic fill of production bounds; traced Python allocations, not RSS')
    # Adversarial supplied catalog should not lose any supported eligible IDs.
    augmented=pd.concat([movies,movies.iloc[:2],pd.DataFrame({'movieId':[999999999]})],ignore_index=True)
    t={};ranked=week7.score_week7_candidates(existing,profiles,augmented,bundle=bundle,timings=t)
    expected=set(movies.movieId)-set(profiles.loc[profiles.userId==existing,'movieId'])
    assert {mid for mid,_ in ranked}==expected and all(np.isfinite(s) for _,s in ranked)
    report['catalog']['adversarial_counts']=t['candidate_counts']
    report['catalog']['exact_supported_candidate_coverage']=True
    report['model_integrity']=week7.file_digest(week7.DEFAULT_BUNDLE)
    args.output.write_text(json.dumps(report,indent=2))
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
