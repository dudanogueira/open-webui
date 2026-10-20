"""Unit tests for the Weaviate vector store's connection setup (no live server needed)."""

from unittest.mock import MagicMock

import pytest

from open_webui.env import VERSION
from open_webui.retrieval.vector.dbs import weaviate as weaviate_db


@pytest.fixture
def connect_kwargs(monkeypatch):
    calls = {}

    def fake_connect_to_custom(**kwargs):
        calls.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(weaviate_db.weaviate, 'connect_to_custom', fake_connect_to_custom)
    weaviate_db.WeaviateClient()
    return calls


def test_sends_integration_header(connect_kwargs):
    assert connect_kwargs['headers'] == {'X-Weaviate-Client-Integration': f'open-webui/{VERSION}'}


@pytest.mark.parametrize(
    'name, expected',
    [
        ('', None),
        ('none', '_UncompressedConfigCreate'),
        ('pq', '_PQConfigCreate'),
        ('bq', '_BQConfigCreate'),
        ('sq', '_SQConfigCreate'),
        ('rq', '_RQConfigCreate'),
        ('bogus', None),
    ],
)
def test_quantizer_config(name, expected):
    quantizer = weaviate_db._quantizer_config(name)
    assert (type(quantizer).__name__ if quantizer else None) == expected


@pytest.fixture
def mt_client(monkeypatch):
    from open_webui.retrieval.vector.dbs import weaviate_multitenancy

    monkeypatch.setattr(weaviate_db.weaviate, 'connect_to_custom', lambda **kwargs: MagicMock())
    return weaviate_multitenancy.WeaviateClient()


@pytest.mark.parametrize(
    'name, collection',
    [
        ('user-memory-a1b2c3d4-dead-beef-0000-111122223333', 'OpenWebUI_memories'),
        ('file-a1b2c3d4-dead-beef-0000-111122223333', 'OpenWebUI_files'),
        ('web-search-a1b2c3d4-dead-beef-0000-111122223333-abc123', 'OpenWebUI_web_search'),
        ('a' * 63, 'OpenWebUI_hash_based'),
        ('a1b2c3d4-dead-beef-0000-111122223333', 'OpenWebUI_knowledge'),
    ],
)
def test_mt_routing_keeps_name_as_tenant(mt_client, name, collection):
    assert mt_client._get_collection_and_tenant(name) == (collection, name)


@pytest.mark.parametrize('name', ['', 'has space', 'x' * 65, 'dots.not.allowed'])
def test_mt_rejects_invalid_tenant_names(mt_client, name):
    with pytest.raises(ValueError):
        mt_client._get_collection_and_tenant(name)


def test_mt_shared_collection_never_treated_as_legacy(mt_client):
    # 'openWebUI_files' sanitizes to the shared 'OpenWebUI_files' collection.
    mt_client.client.collections.exists.return_value = True
    assert mt_client._legacy_collection_name('openWebUI_files') is None
    assert mt_client._legacy_collection_name('file-abc') == 'File_abc'
