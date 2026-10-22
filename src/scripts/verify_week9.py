from pathlib import Path
import json,hashlib,subprocess,sys,statistics
from time import perf_counter
from dataclasses import asdict
root=Path.cwd();sys.path.insert(0,str(root))
from src.hybrid import week7
from src.hybrid.group_recommender import recommend_group
from src.hybrid.serving_cache import SERVING_CACHE
from src.scripts.week8_context import construct_groups
from src.scripts.benchmark_week9 import source,summary
from concurrent.futures import ThreadPoolExecutor
bundle=week7.load_bundle(resource_root=root)
profiles=bundle['reference_ratings'];movies=bundle['movies']
out=Path(sys.argv[1]) if len(sys.argv)>1 else root/'reports/week9'
out.mkdir(parents=True,exist_ok=True)
for name in ['group-preservation.json','group-concurrency.json','matched-before-after.json','existing-cold-serial.json']:
    if (out/name).exists():raise FileExistsError('Refusing verification report overwrite: '+name)
specs,_=construct_groups(profiles);lookup={name:(users,ratings) for name,users,ratings in specs}
oldgroup=source('src.hybrid._week9_group_baseline',root/'reports/week9/baseline/group_recommender.py')
groups=[]
for name,users,ratings in specs:
    for strategy in ['mean','least_misery','fairness_aware']:
        a=oldgroup.recommend_group(users,ratings,movies,bundle=bundle,strategy=strategy)
        b=recommend_group(users,ratings,movies,bundle=bundle,strategy=strategy)
        assert [asdict(r) for r in a]==[asdict(r) for r in b]
        groups.append(dict(case=name,strategy=strategy,complete_output_exact=True))
    print('Group golden',name,flush=True)
(out/'group-preservation.json').write_text(json.dumps(groups,indent=2),encoding='utf-8')
users,ratings=lookup['similar_pair']
reference=recommend_group(users,ratings,movies,bundle=bundle)
def group_request(_):
    start=perf_counter();actual=recommend_group(users,ratings,movies,bundle=bundle)
    assert actual==reference
    return perf_counter()-start
start=perf_counter()
with ThreadPoolExecutor(max_workers=4) as pool:concurrent=list(pool.map(group_request,range(8)))
(out/'group-concurrency.json').write_text(json.dumps(dict(requests=8,threads=4,seconds=perf_counter()-start,
    latency=summary(concurrent),exact=True),indent=2),encoding='utf-8')
old=source('week9_old_scores',root/'reports/week9/baseline/week7.py')
sparse,frame=lookup['sparse_member'];cold,coldframe=lookup['cold_member']
timings=[]
for name,uid,ratings in [('existing',23,profiles),('sparse',sparse[-1],frame),('cold',cold[-1],coldframe)]:
    SERVING_CACHE.clear();a=old.recommend_week7(uid,ratings,movies,len(movies),bundle=bundle,use_cache=False)
    b=week7.recommend_week7(uid,ratings,movies,len(movies),bundle=bundle,use_cache=False)
    assert [asdict(r) for r in a]==[asdict(r) for r in b]
    old.recommend_week7(uid,ratings,movies,50,bundle=bundle)
    before=[];after=[]
    for _ in range(30):
        s=perf_counter();old.recommend_week7(uid,ratings,movies,50,bundle=bundle);before.append(perf_counter()-s)
        s=perf_counter();week7.recommend_week7(uid,ratings,movies,50,bundle=bundle);after.append(perf_counter()-s)
    timings.append(dict(case=name,before=summary(before),after=summary(after),exact_full_scores=True,
                        scope='Paired alternating warm score-prefix only; diversity excluded'))
(out/'matched-before-after.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
# Existing-user cold samples initially overlapped the full test run. Replace
# the reported timing with a separate serial run; preserve original evidence.
from src.hybrid.recommender import recommend_for_user
samples=[]
for _ in range(3):
    SERVING_CACHE.clear();s=perf_counter();recommend_for_user(23,profiles,movies,week7_bundle=bundle);samples.append(perf_counter()-s)
warm=[]
for _ in range(30):
    s=perf_counter();recommend_for_user(23,profiles,movies,week7_bundle=bundle);warm.append(perf_counter()-s)
(out/'existing-cold-serial.json').write_text(json.dumps(dict(latency=summary(samples),samples=samples,warm=summary(warm),
    reason='Initial existing cold benchmark overlapped full tests; serial rerun with no other task test/install jobs'),indent=2),encoding='utf-8')
print('Complete score preservation and serial timing checks passed',flush=True)
