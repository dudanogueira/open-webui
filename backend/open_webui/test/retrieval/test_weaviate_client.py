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
