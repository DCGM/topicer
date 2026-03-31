import pytest
import requests
from urllib.parse import urljoin


@pytest.mark.integration
def test_config(base_url):
    endpoint = urljoin(f"{base_url.rstrip('/')}/", "configs")
    response = requests.get(urljoin(base_url, endpoint))
    assert response.status_code == 200, f"{endpoint} is not 200."
    config = response.json()

    assert isinstance(config, list)
    assert len(config) > 0
