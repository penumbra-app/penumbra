"""Reproducible profile-only group construction and isolated validation labels."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from src.hybrid.week7 import file_digest


def validation_labels(bundle, path=Path('data/ratings.csv')):
    """Use only 70–80% labels. Total counts/order define frozen boundaries;
    final 20% labels never enter selection, metrics or any model fitting.
    """
    if file_digest(path) != bundle['ratings_sha256']:
        raise ValueError('Ratings source differs from the frozen split provenance')
    frame = pd.read_csv(path)
    slices = []
    for _, rows in frame.groupby('userId', sort=False):
        if len(rows) < 10:
            continue
        ordered = rows.sort_values(['timestamp', 'movieId'], kind='stable')
        slices.append(ordered.iloc[int(len(rows)*.70):int(len(rows)*.80)].copy())
    # No test slice is produced or retained by this function.
    return pd.concat(slices, ignore_index=True)


def construct_groups(profiles, *, pool_size=100, minimum_common=8):
    counts = profiles.groupby('userId').size()
    users = sorted(int(uid) for uid in counts.index[counts >= 25])[:pool_size]
    dictionaries = {uid: rows.groupby('movieId').rating.mean().to_dict()
                    for uid, rows in profiles.loc[profiles.userId.isin(users)].groupby('userId')}
    pairs = []
    for a, b in combinations(users, 2):
        common = sorted(dictionaries[a].keys() & dictionaries[b].keys())
        if len(common) < minimum_common:
            continue
        x, y = [np.array([dictionaries[uid][mid] for mid in common]) for uid in (a, b)]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        pairs.append(dict(users=[a, b], correlation=float(np.corrcoef(x, y)[0, 1]), common=len(common)))
    if not pairs:
        raise ValueError('No sufficiently supported co-rated profile pairs')
    pairs.sort(key=lambda p: (p['correlation'], p['users']))
    lookup = {tuple(p['users']): p for p in pairs}
    similar = pairs[-1]['users']
    conflicting = pairs[0]['users']
    if pairs[0]['correlation'] >= 0:
        raise ValueError('No negatively correlated pair; cannot claim conflict coverage')
    moderate = min(pairs, key=lambda p: (abs(p['correlation']-.3), p['users']))['users']
    def correlations(members):
        return [lookup[tuple(sorted(pair))] for pair in combinations(members, 2)
                if tuple(sorted(pair)) in lookup]
    eligible_outliers = [(np.mean([p['correlation'] for p in correlations(similar+[uid])
                                   if uid in p['users']]), uid)
                         for uid in users if uid not in similar
                         and len([p for p in correlations(similar+[uid]) if uid in p['users']]) == 2]
    _, outlier = min(eligible_outliers)
    third = min((np.mean([p['correlation'] for p in correlations(conflicting+[uid])]), uid)
                for uid in users if uid not in conflicting and len(correlations(conflicting+[uid])) == 3)[1]
    larger = list(dict.fromkeys(conflicting + similar + [outlier, third]))
    larger += [uid for uid in users if uid not in larger][:max(0, 5-len(larger))]
    sparse, cold = int(counts.index.max())+1, int(counts.index.max())+2
    source = profiles.loc[profiles.userId == conflicting[1]].sort_values(['timestamp','movieId'], kind='stable').iloc[:1].copy()
    source['userId'] = sparse
    sparse_frame = pd.concat([profiles, source], ignore_index=True)
    specs = [('similar_pair', similar, profiles), ('moderate_pair', moderate, profiles),
             ('conflicting_pair', conflicting, profiles), ('conflicting_triple', conflicting+[third], profiles),
             ('one_outlier', similar+[outlier], profiles), ('larger_five', larger[:5], profiles),
             ('sparse_member', [conflicting[0], sparse], sparse_frame),
             ('cold_member', [conflicting[0], cold], profiles),
             ('multiple_cold', [conflicting[0], cold, cold+1], profiles)]
    metadata = dict(selection='First 100 ascending IDs with >=25 profile ratings; pairwise Pearson on >=8 co-rated profile movies; ties by IDs',
        eligible_user_count=len(users), supported_pairs=len(pairs), minimum_common=minimum_common,
        similar='Maximum supported correlation', moderate='Supported correlation closest to +0.3',
        conflict='Minimum supported correlation; require negative',
        outlier='Most negative mean correlation with the highly similar pair',
        pairs=pairs, group_evidence={label: correlations(members) for label,members,_ in specs})
    return specs, metadata
