"""Conflict-focused frozen Group V1 audit; no training or final-test evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from src.hybrid.week7 import load_bundle
from src.hybrid.group_recommender import recommend_group
from src.hybrid.serving_cache import SERVING_CACHE
from src.scripts.evaluate_group_v1 import proxy_metrics
from src.scripts.week8_context import construct_groups, validation_labels


def extended_metrics(rows, candidate_count, validation, users, k):
    metrics = proxy_metrics(rows)
    if not rows:
        return metrics
    values = np.array([[dict(r.member_scores)[uid] for uid in sorted(users)] for r in rows])
    metrics.update(mean_member_variance=float(values.var(axis=1).mean()),
        movies_below_relative_quartile=int((values < .25).any(axis=1).sum()),
        candidate_coverage=len(rows)/candidate_count,
        mean_member_std=float(values.std(axis=1).mean()))
    labels = validation.loc[validation.userId.isin(users)].drop_duplicates(['userId','movieId'])
    observed = labels.loc[labels.movieId.isin([r.movie_id for r in rows])]
    metrics['independent_validation'] = dict(observed_member_movie_labels=len(observed),
        possible_member_movie_labels=len(users)*len(rows), observed_fraction=len(observed)/(len(users)*len(rows)),
        mean_observed_rating=float(observed.rating.mean()) if len(observed) else None,
        member_labels={str(uid): int((observed.userId==uid).sum()) for uid in users},
        caveat='Sparse exposure-selected individual ratings, missing labels unknown, not group acceptance')
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('reports/week8/freeze/conflicts.json'))
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing to overwrite existing report')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    bundle=load_bundle(); profiles=bundle['reference_ratings'].copy(); movies=bundle['movies']
    specs,selection=construct_groups(profiles)
    validation=validation_labels(bundle)
    report=dict(protocol='Profile-only group selection; isolated chronological 70–80% labels used only for independent observed-rating summaries. Final-test metrics not rerun.',
        selection=selection, validation_rows=len(validation), default_strategy='mean',
        heuristic='70/30 unchanged, no tuning or strategy promotion', groups=[])
    for label,users,frame in specs:
        watched=set(frame.loc[frame.userId.isin(users),'movieId'])
        count=len(set(movies.movieId)-watched)
        SERVING_CACHE.clear()
        group=dict(category=label,users=users,history_counts={str(uid):int((frame.userId==uid).sum()) for uid in users},
            candidate_count=count,profile_pair_evidence=selection['group_evidence'][label],results=[])
        # Small common-rated validation probe diagnoses profile criterion stability
        # independently. It never changes selected groups or serving configuration.
        probes=[]
        for a in range(len(users)):
            for b in range(a+1,len(users)):
                common=validation.loc[validation.userId==users[a]].merge(
                    validation.loc[validation.userId==users[b]],on='movieId',suffixes=('_a','_b'))
                probes.append(dict(users=[users[a],users[b]],common=len(common),
                    correlation=float(common.rating_a.corr(common.rating_b)) if len(common)>=3
                        and common.rating_a.std()>0 and common.rating_b.std()>0 else None))
        group['independent_pair_validation']=probes
        for k in (5,10,20):
            slates={}
            for strategy in ('mean','least_misery','fairness_aware'):
                t={}; start=perf_counter()
                rows=recommend_group(users,frame,movies,bundle=bundle,limit=k,strategy=strategy,timings=t)
                elapsed=perf_counter()-start
                assert len(rows)==min(k,count)
                assert not watched.intersection(r.movie_id for r in rows)
                slates[strategy]=[r.movie_id for r in rows]
                examples=sorted(rows,key=lambda r:(-r.disagreement,r.movie_id))[:3]
                group['results'].append(dict(k=k,strategy=strategy,seconds=elapsed,
                    cache_mode='cold' if k==5 and strategy=='mean' else 'warm',
                    metrics=extended_metrics(rows,count,validation,users,k),slate=slates[strategy],
                    disagreement_examples=[asdict(r) for r in examples],timings=t))
            mean=slates['mean']
            for row in group['results'][-3:]:
                row['ordered_changes_vs_mean']=sum(a!=b for a,b in zip(mean,row['slate']))
                row['set_jaccard_vs_mean']=len(set(mean)&set(row['slate']))/len(set(mean)|set(row['slate']))
        report['groups'].append(group)
        args.output.with_suffix('.progress.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(label,users,'correlations',[round(p['correlation'],3) for p in group['profile_pair_evidence']],flush=True)
    report['limitations']=['No observed group-choice labels; group acceptance accuracy unknown',
        'Pearson estimates on small co-rated sets are noisy; selection cohort is deterministic, not representative',
        'Percentiles are relative preferences, not probabilities; validation exposure labels are sparse and nonrandom',
        'Objectives and diversity can drive proxy differences; no winner selected']
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Saved',args.output,flush=True)


if __name__=='__main__': main()
