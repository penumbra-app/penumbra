from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import pytest
from src.hybrid import week7
from src.hybrid.group_recommender import recommend_group, percentile_preferences
from src.hybrid.rating_updates import update_rating
from src.hybrid.serving_cache import SERVING_CACHE
from tests.hybrid.test_week8 import serving


@pytest.mark.parametrize('operation', ['add','edit','remove'])
def test_rating_lifecycle_individual_group_and_unrelated(serving,operation):
    frame,movies,bundle,_=serving
    before=frame.copy(deep=True)
    solo=lambda uid,ratings: week7.recommend_week7(uid,ratings,movies,100,bundle=bundle)
    untouched=solo(4,frame)
    group_untouched=recommend_group([4,5],frame,movies,bundle=bundle)
    solo(1,frame);recommend_group([1,3],frame,movies,bundle=bundle)
    mid=10 if operation=='add' else 1
    options={} if operation=='remove' else dict(rating=5.,timestamp=1000)
    changed=update_rating(frame,user_id=1,movie_id=mid,operation=operation,**options)
    pd.testing.assert_frame_equal(frame,before)
    individual=solo(1,changed)
    grouped=recommend_group([1,3],changed,movies,bundle=bundle,limit=100)
    assert (mid in {r.movie_id for r in individual}) == (operation=='remove')
    assert (mid in {r.movie_id for r in grouped}) == (operation=='remove')
    assert solo(4,changed)==untouched
    assert recommend_group([4,5],changed,movies,bundle=bundle)==group_untouched
    SERVING_CACHE.clear()
    assert solo(1,changed)==individual
    assert recommend_group([1,3],changed,movies,bundle=bundle,limit=100)==grouped


def test_five_signal_update_contract_with_real_content(serving,monkeypatch):
    frame,movies,bundle,_=serving
    from src.hybrid.content_adapter import build_content_model_from_frames
    bundle['selected_config_path']=str(Path('src/content/selected_config.json'))
    mf_digest=joblib.hash(bundle['collaborative_model'])
    def features(frame):
        h=frame.loc[frame.userId==1]
        content=build_content_model_from_frames(h,bundle['movies'],use_selected_content=True,
            selected_config_path=bundle['selected_config_path'])
        return week7.build_feature_rows(1,h,bundle['reference_ratings'],movies,[4,5,6,7,8],content,bundle['collaborative_model'])[1]
    original=features(frame)
    revised=update_rating(frame,user_id=1,movie_id=1,operation='edit',rating=5.,timestamp=10000)
    updated=features(revised)
    assert np.array_equal(original[:,[0,3,4]],updated[:,[0,3,4]])
    assert not np.array_equal(original[:,1],updated[:,1])
    assert not np.array_equal(original[:,2],updated[:,2])
    assert joblib.hash(bundle['collaborative_model'])==mf_digest


@pytest.mark.parametrize('options', [dict(operation='add',movie_id=1,rating=4.,timestamp=1),
    dict(operation='edit',movie_id=99,rating=4.),dict(operation='remove',movie_id=99),
    dict(operation='edit',movie_id=1,rating=np.nan),dict(operation='edit',movie_id=1,rating=6.),
    dict(operation='add',movie_id=99,rating=4.),dict(operation='add',movie_id=99,rating=4.,timestamp=-1),
    dict(operation='remove',movie_id=1,rating=4.),dict(operation='wrong',movie_id=1)])
def test_bad_updates_fail_without_mutating(serving,options):
    frame,*_=serving;before=frame.copy(deep=True)
    with pytest.raises(ValueError):update_rating(frame,user_id=1,**options)
    pd.testing.assert_frame_equal(frame,before)


def test_duplicate_events_require_explicit_resolution(serving):
    frame,*_=serving
    duplicate=pd.concat([frame,frame.loc[frame.userId==1].iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='Ambiguous'):
        update_rating(duplicate,user_id=1,movie_id=1,operation='edit',rating=5.)


def test_new_user_and_index_stability(serving):
    frame,_,bundle,_=serving
    before=frame.loc[frame.userId==1].copy()
    new=update_rating(frame,user_id=9,movie_id=4,operation='add',rating=4.,timestamp=12)
    pd.testing.assert_frame_equal(new.loc[new.userId==1],before)
    assert str(new.userId.dtype)==str(frame.userId.dtype)
    assert len(new.loc[new.userId==9])==1


def test_large_multiple_cold_groups_are_bounded_and_deterministic(serving):
    frame,movies,bundle,_=serving
    users=list(range(3,21))
    first=recommend_group(users,frame,movies,bundle=bundle)
    assert first==recommend_group(users[::-1],frame,movies,bundle=bundle)
    assert SERVING_CACHE.info()['entries']<=16
    assert SERVING_CACHE.info()['candidates']<=200000
    assert all(len(r.member_scores)==18 for r in first)


def test_identical_histories_no_disagreement(serving):
    frame,movies,bundle,_=serving
    twin=frame.loc[frame.userId==1].copy();twin['userId']=3
    frame=pd.concat([frame,twin],ignore_index=True)
    assert all(r.disagreement==0 for r in recommend_group([1,3],frame,movies,bundle=bundle))


def test_normalization_invariant_to_positive_raw_scale():
    values=np.array([1.,7.,7.,9.])
    assert np.array_equal(percentile_preferences(values),percentile_preferences(values*100000+5000))


def test_concurrent_real_serving_bookkeeping(serving):
    frame,movies,bundle,_=serving
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:recommend_group([1,3],frame,movies,bundle=bundle),range(8)))
    assert all(rows==results[0] for rows in results)
    assert SERVING_CACHE.info()['entries']<=16


def test_catalog_missing_predictions_and_fallback_observability(serving):
    frame,movies,bundle,_=serving
    t={};week7.score_week7_candidates(1,frame,pd.concat([movies,movies.iloc[:2],pd.DataFrame({'movieId':[999]})]),bundle=bundle,timings=t)
    assert t['candidate_counts']['supplied_rows']==23
    assert t['candidate_counts']['eligible']==17
    assert t['feature_candidates']==17
    content=SimpleNamespace(predict=lambda **kw: [])
    with pytest.raises(ValueError,match='count'):
        week7.build_feature_rows(1,frame,bundle['reference_ratings'],movies,[4],content,None)
    content.predict=lambda **kw:[SimpleNamespace(movie_id=5,predicted_score=3.)]
    with pytest.raises(ValueError,match='order'):
        week7.build_feature_rows(1,frame,bundle['reference_ratings'],movies,[4],content,None)
    content.predict=lambda **kw:[SimpleNamespace(movie_id=4,predicted_score=np.nan)]
    t={};_,values=week7.build_feature_rows(1,frame,bundle['reference_ratings'],movies,[4],content,None,timings=t)
    assert values[0,2]==2.5 and t['content_nonfinite_predictions']==1


def test_corrupt_artifact_and_integrity_before_deserialization(tmp_path,monkeypatch):
    path=tmp_path/'bad.joblib';path.write_bytes(b'broken artifact')
    with pytest.raises(ValueError,match='Cannot load'):
        week7.load_bundle(path)
    called=[]
    monkeypatch.setattr(week7.joblib,'load',lambda *a:called.append(True))
    with pytest.raises(ValueError,match='checksum'):
        week7.load_bundle(path,expected_sha256='wrong')
    assert not called


def test_frozen_preset_change_rejected_even_with_warm_cache(serving):
    frame,movies,bundle,_=serving
    path=Path(bundle['selected_config_path'])
    bundle['selected_config_sha256']=week7.file_digest(path)
    week7.recommend_week7(1,frame,movies,bundle=bundle)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError,match='Frozen Content preset changed'):
        week7.recommend_week7(1,frame,movies,bundle=bundle)


def test_nonfinite_ids_rejected_and_missing_mf_is_observable(serving):
    frame,movies,bundle,_=serving
    invalid=movies.astype({'movieId':float});invalid.loc[0,'movieId']=np.inf
    with pytest.raises(ValueError,match='IDs'):
        week7.recommend_week7(1,frame,invalid,bundle=bundle)
    t={};scores=week7.collaborative_scores(None,1,[4,5],timings=t)
    assert scores=={4:2.5,5:2.5} and t['mf_missing_predictions']==2


def test_validation_only_loader_discards_final_labels(tmp_path):
    from src.scripts.week8_context import validation_labels
    frame=pd.DataFrame({'userId':[1]*10,'movieId':range(1,11),
                        'rating':[3.]*8+[123456.,654321.], 'timestamp':range(10)})
    path=tmp_path/'ratings.csv';frame.to_csv(path,index=False)
    bundle={'ratings_sha256':week7.file_digest(path)}
    result=validation_labels(bundle,path)
    assert result.movieId.tolist()==[8] and result.rating.tolist()==[3.]
    with pytest.raises(ValueError,match='provenance'):
        validation_labels({'ratings_sha256':'wrong'},path)


def test_profile_conflict_group_selection_is_reproducible():
    from src.scripts.week8_context import construct_groups
    rows=[]
    for uid in range(1,7):
        for mid in range(1,31):
            base=1+(mid%5)
            rating=6-base if uid in (2,4) else (1+(2*mid%5) if uid==6 else base)
            rows.append(dict(userId=uid,movieId=mid,rating=float(rating),timestamp=mid))
    frame=pd.DataFrame(rows)
    groups,metadata=construct_groups(frame)
    repeated,again=construct_groups(frame.iloc[::-1])
    assert [(name,u) for name,u,_ in groups]==[(name,u) for name,u,_ in repeated]
    assert metadata['group_evidence']['conflicting_pair'][0]['correlation']==pytest.approx(-1.,abs=1e-14)
    assert len(dict((name,u) for name,u,_ in groups)['larger_five'])==5


def test_real_content_unknown_history_and_duplicate_history_contract(serving,monkeypatch,capsys):
    frame,movies,bundle,_=serving
    from src.hybrid.content_adapter import build_content_model_from_frames
    monkeypatch.setattr(week7,'build_content_model_from_frames',build_content_model_from_frames)
    bundle['selected_config_path']=str(Path('src/content/selected_config.json'))
    extended=update_rating(frame,user_id=1,movie_id=999999,operation='add',rating=4.,timestamp=100)
    result=week7.recommend_week7(1,extended,movies,100,bundle=bundle)
    assert 999999 not in {r.movie_id for r in result} and len(result)==17
    recommend_group([1,3],extended,movies,bundle=bundle)
    assert capsys.readouterr().out==''
    duplicate=pd.concat([frame,frame.loc[frame.userId==1].iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate ratings'):
        week7.recommend_week7(1,duplicate,movies,bundle=bundle)


@pytest.mark.parametrize('change',['nan','fractional_id','timestamp','schema'])
def test_malformed_individual_histories_fail_clearly(serving,change):
    frame,movies,bundle,_=serving
    if change=='nan':frame.loc[frame.userId==1,'rating']=np.nan
    elif change=='fractional_id':movies=movies.astype({'movieId':float});movies.loc[0,'movieId']=1.2
    elif change=='timestamp':frame.loc[frame.userId==1,'timestamp']=-1
    else:frame=frame.drop(columns='rating')
    with pytest.raises(ValueError):week7.recommend_week7(1,frame,movies,bundle=bundle)
