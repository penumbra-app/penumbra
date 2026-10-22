import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from tests.hybrid.test_week8 import serving
from src.hybrid import release, week7
from src.hybrid.release import RatingHistory, ServingSession, ServingMetrics


def test_owned_history_and_revision(serving):
    ratings, _, bundle, _ = serving
    store = RatingHistory(ratings)
    revision, snapshot = store.snapshot()
    snapshot.loc[:, 'rating'] = 0
    assert store.snapshot()[1].equals(ratings)
    assert revision == 0
    assert store.update(expected_revision=0,user_id=1,movie_id=1,operation='edit',rating=5.) == 1
    with pytest.raises(ValueError,match='revision conflict'):
        store.update(expected_revision=0,user_id=1,movie_id=1,operation='remove')
    assert store.snapshot()[0] == 1
    assert ratings.loc[(ratings.userId==1)&(ratings.movieId==1),'rating'].item() == 2.


def test_mid_request_update_has_consistent_snapshot(serving,monkeypatch):
    ratings, _, bundle, _ = serving
    session=ServingSession(bundle,RatingHistory(ratings))
    reference=session.individual(1)
    week7.SERVING_CACHE.clear()
    entered=Event(); resume=Event()
    original=week7.build_content_model_from_frames
    def blocked(*args,**kwargs):
        entered.set()
        assert resume.wait(10)
        return original(*args,**kwargs)
    monkeypatch.setattr(week7,'build_content_model_from_frames',blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(session.individual,1)
        assert entered.wait(10)
        session.update(user_id=1,movie_id=1,operation='edit',rating=5.)
        resume.set()
        assert pending.result(timeout=10) == reference
    updated=session.individual(1)
    assert updated != reference
    assert updated == release.recommend_for_user(1,session.history.snapshot()[1],bundle['movies'],week7_bundle=bundle)


def test_metrics_and_default_group_preserve_outputs(serving):
    ratings,movies,bundle,_=serving
    metrics=ServingMetrics();session=ServingSession(bundle,RatingHistory(ratings),metrics=metrics)
    assert session.individual(1)==release.recommend_for_user(1,ratings,movies,week7_bundle=bundle)
    assert session.group([1,3])==release.recommend_group([1,3],ratings,movies,bundle=bundle)
    with pytest.raises(ValueError):session.group([1,1])
    with pytest.raises(ValueError):session.individual(1,-1)
    with pytest.raises(ValueError):session.update(user_id=1,movie_id=1,operation='add',rating=5.,timestamp=1)
    report=metrics.snapshot()
    assert report['individual_requests']==2 and report['individual_errors']==1
    assert report['group_requests']==2 and report['group_errors']==1
    assert report['update_errors']==1
    assert all('user' not in key for key in report)


@pytest.mark.parametrize('change',['artifact','config','features','schema','dependency','python','escape','missing'])
def test_release_failures_precede_deserialization(tmp_path,monkeypatch,change):
    from pathlib import Path
    manifest=json.loads(Path('docs/week9-artifact-manifest.json').read_text())
    manifest['artifact']={'path':'bundle','sha256':week7.file_digest(Path('docs/week9-artifact-manifest.json'))}
    manifest['config']={'path':'preset','sha256':week7.file_digest(Path('docs/week9-artifact-manifest.json'))}
    data=Path('docs/week9-artifact-manifest.json').read_bytes()
    (tmp_path/'bundle').write_bytes(data);(tmp_path/'preset').write_bytes(data)
    if change in ['artifact','config']:manifest[change]['sha256']='0'*64
    if change=='features':manifest['features']=[]
    if change=='schema':manifest['schema_version']=2
    if change=='dependency':manifest['packages']['numpy']='0'
    if change=='python':manifest['python']='0.0.0'
    if change=='escape':manifest['artifact']['path']='../outside'
    if change=='missing':(tmp_path/'bundle').unlink()
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    loader=Mock(side_effect=AssertionError('Must not deserialize'))
    monkeypatch.setattr(release,'load_bundle',loader)
    with pytest.raises((ValueError,FileNotFoundError)):
        release.load_release(tmp_path,trusted_manifest=path)
    loader.assert_not_called()


def test_resource_root_normalizes_windows_paths(serving,tmp_path,monkeypatch):
    _,_,bundle,_=serving
    bundle=dict(bundle,schema_version=1,selected_config_path='config\\preset.json')
    bundle['collaborative_model'].is_fitted_=True
    config=tmp_path/'config/preset.json';config.parent.mkdir();config.write_text('{}')
    bundle['selected_config_sha256']=week7.file_digest(config)
    artifact=tmp_path/'bundle';artifact.write_bytes(b'local synthetic placeholder')
    monkeypatch.setattr(week7.joblib,'load',lambda path:bundle)
    loaded=week7.load_bundle(artifact,resource_root=tmp_path)
    assert loaded['selected_config_path']==str(config.resolve())
    assert bundle['selected_config_path']=='config\\preset.json'
    bundle['selected_config_path']='../outside.json'
    with pytest.raises(ValueError,match='within resource_root'):week7.load_bundle(artifact,resource_root=tmp_path)
