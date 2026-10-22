from pathlib import Path
import sys, json, os, tempfile, hashlib
from dataclasses import asdict
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
output=Path(sys.argv[2]).resolve()
if output.exists():raise FileExistsError('Refusing smoke report overwrite')
from src.hybrid.release import load_release,RatingHistory,ServingSession
from src.scripts.week8_context import construct_groups
from src.hybrid.week7 import score_week7_candidates
os.chdir(tempfile.gettempdir())
bundle=load_release(root,trusted_manifest=root/'docs/week9-artifact-manifest.json')
profiles=bundle['reference_ratings'];movies=bundle['movies']
specs,_=construct_groups(profiles);lookup={n:(u,f) for n,u,f in specs}
sparse,frame=lookup['sparse_member'];cold,coldframe=lookup['cold_member']
result={}
for name,uid,ratings in [('existing',23,profiles),('sparse',sparse[-1],frame),('cold',cold[-1],coldframe)]:
    session=ServingSession(bundle,RatingHistory(ratings))
    result[name]=[asdict(r) for r in session.individual(uid)]
    ranked=score_week7_candidates(uid,ratings,movies,bundle=bundle)
    result[name+'_full_score_sha256']=hashlib.sha256(json.dumps(ranked).encode()).hexdigest()
for name in ['similar_pair','one_outlier','larger_five','conflicting_pair']:
    users,ratings=lookup[name];session=ServingSession(bundle,RatingHistory(ratings))
    for strategy in ['mean','least_misery','fairness_aware']:
        result[name+'_'+strategy]=[asdict(r) for r in session.group(users,strategy=strategy)]
output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Verified cross-cwd individual/group serving; saved',output)
