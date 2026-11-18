import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest


@pytest.fixture
def postgres_url():
    url = os.environ.get("RASTA_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set RASTA_TEST_POSTGRES_URL to exercise PostgreSQL")
    schema = "test_" + uuid4().hex
    identifier = sql.Identifier(schema)
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query))
    params["options"] = (params.get("options", "") + " -csearch_path=" + schema).strip()
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(identifier))
        try:
            yield urlunsplit(parts._replace(query=urlencode(params)))
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(identifier))
