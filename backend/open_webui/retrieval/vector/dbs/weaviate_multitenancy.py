"""
NOTE: This vector database integration is community-supported and maintained on a best-effort basis.

Weaviate backend that stores Open WebUI's collections as tenants of a few shared multi-tenant
collections instead of one Weaviate collection each. Per-file and per-memory collections are
tiny and numerous, which Weaviate handles far better as tenants than as separate collections.
"""

import logging
import re
from typing import Any

import weaviate
from open_webui.config import WEAVIATE_COLLECTION_PREFIX
from open_webui.retrieval.vector.dbs.weaviate import WeaviateClient as _SingleTenantWeaviateClient

log = logging.getLogger(__name__)

# Weaviate's tenant name rule. Tenant names are stored as given (case-sensitive, no
# normalization), so distinct Open WebUI collection names never share a tenant.
_TENANT_NAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


class WeaviateClient(_SingleTenantWeaviateClient):
    """Multi-tenant Weaviate backend (ENABLE_WEAVIATE_MULTITENANCY_MODE=true).

    Backward compatibility: a collection that already exists as its own Weaviate collection
    (created before multi-tenancy mode was enabled) keeps being read, written and deleted
    there, so enabling the mode never hides existing data. Only collections that do not
    exist yet are created as tenants. Nothing is migrated automatically.
    """

    def __init__(self):
        super().__init__()
        prefix = self._sanitize_collection_name(WEAVIATE_COLLECTION_PREFIX)
        self.MEMORY_COLLECTION = f'{prefix}_memories'
        self.KNOWLEDGE_COLLECTION = f'{prefix}_knowledge'
        self.FILE_COLLECTION = f'{prefix}_files'
        self.WEB_SEARCH_COLLECTION = f'{prefix}_web_search'
        self.HASH_BASED_COLLECTION = f'{prefix}_hash_based'
        self._mt_collections = {
            self.MEMORY_COLLECTION,
            self.KNOWLEDGE_COLLECTION,
            self.FILE_COLLECTION,
            self.WEB_SEARCH_COLLECTION,
            self.HASH_BASED_COLLECTION,
        }
        # Shared collections known to exist; cleared by reset().
        self._ready_mt_collections: set[str] = set()

    def _get_collection_and_tenant(self, collection_name: str) -> tuple[str, str]:
        """
        Maps an Open WebUI collection name to (shared collection, tenant).

        WARNING: This mapping relies on current Open WebUI naming conventions for collection
        names ("user-memory-", "file-", "web-search-" prefixes and 63-char hashes). If those
        change, data would be routed to a different shared collection than before.
        """
        if not isinstance(collection_name, str) or not _TENANT_NAME_RE.match(collection_name):
            raise ValueError(f'Invalid collection name for Weaviate multi-tenancy: {collection_name!r}')

        if collection_name.startswith('user-memory-'):
            return self.MEMORY_COLLECTION, collection_name
        elif collection_name.startswith('file-'):
            return self.FILE_COLLECTION, collection_name
        elif collection_name.startswith('web-search-'):
            return self.WEB_SEARCH_COLLECTION, collection_name
        # Hash-based collections (YouTube and web URLs)
        elif len(collection_name) == 63 and all(c in '0123456789abcdef' for c in collection_name):
            return self.HASH_BASED_COLLECTION, collection_name
        else:
            return self.KNOWLEDGE_COLLECTION, collection_name

    def _legacy_collection_name(self, collection_name: str) -> str | None:
        """Name of the pre-multi-tenancy collection for `collection_name`, if it exists."""
        try:
            sane_collection_name = self._sanitize_collection_name(collection_name)
        except ValueError:
            return None
        # A crafted name can sanitize to a shared collection's name; never treat a shared
        # collection as one tenant's legacy collection, which would expose every tenant.
        if sane_collection_name in self._mt_collections:
            return None
        if not self.client.collections.exists(sane_collection_name):
            return None
        return sane_collection_name

    def _mt_collection_exists(self, mt_collection: str) -> bool:
        if mt_collection in self._ready_mt_collections:
            return True
        if self.client.collections.exists(mt_collection):
            self._ready_mt_collections.add(mt_collection)
            return True
        return False

    def _ensure_mt_collection(self, mt_collection: str) -> None:
        if self._mt_collection_exists(mt_collection):
            return
        try:
            self._create_collection(
                mt_collection,
                multi_tenancy_config=weaviate.classes.config.Configure.multi_tenancy(
                    enabled=True,
                    auto_tenant_creation=True,
                    auto_tenant_activation=True,
                ),
            )
            log.info('Multi-tenant collection %s created', mt_collection)
        except Exception:
            # Another worker may have created it first.
            if not self.client.collections.exists(mt_collection):
                raise
        self._ready_mt_collections.add(mt_collection)

    def _open_collection(self, collection_name: str) -> Any:
        legacy = self._legacy_collection_name(collection_name)
        if legacy:
            return self.client.collections.get(legacy)

        mt_collection, tenant = self._get_collection_and_tenant(collection_name)
        if not self._mt_collection_exists(mt_collection):
            return None
        collection = self.client.collections.get(mt_collection)
        # Querying a tenant that does not exist is an error in Weaviate, not an empty result.
        if not collection.tenants.exists(tenant):
            return None
        return collection.with_tenant(tenant)

    def _open_collection_for_write(self, collection_name: str) -> Any:
        legacy = self._legacy_collection_name(collection_name)
        if legacy:
            return self.client.collections.get(legacy)

        mt_collection, tenant = self._get_collection_and_tenant(collection_name)
        self._ensure_mt_collection(mt_collection)
        # The tenant is created on first write by auto_tenant_creation.
        return self.client.collections.get(mt_collection).with_tenant(tenant)

    def has_collection(self, collection_name: str) -> bool:
        return self._open_collection(collection_name) is not None

    def delete_collection(self, collection_name: str) -> None:
        legacy = self._legacy_collection_name(collection_name)
        if legacy:
            self.client.collections.delete(legacy)

        mt_collection, tenant = self._get_collection_and_tenant(collection_name)
        if not self._mt_collection_exists(mt_collection):
            return
        collection = self.client.collections.get(mt_collection)
        if collection.tenants.exists(tenant):
            collection.tenants.remove([tenant])

    def reset(self) -> None:
        super().reset()
        self._ready_mt_collections.clear()
