
import pytest
from topicer.database.weaviate_service import WeaviateService
from unittest.mock import MagicMock

# -------------------- UNIT TESTS --------------------

@pytest.mark.unit
def test_weaviate_service_cleanup_mock_unit(mock_service):
    service, mock_client = mock_service

    # Verifify manual close
    service.close()

    assert mock_client.close.call_count == 1
    assert service._client is None


@pytest.mark.unit
def test_service_close_idempotency_unit(mock_service):
    service, mock_client = mock_service

    # Close it multiple times
    service.close()
    service.close()
    service.close()

    assert mock_client.close.call_count == 1
    assert service._client is None


@pytest.mark.unit
def test_weaviate_service_context_manager_unit(mocker):
    mock_connect = mocker.patch(
        "topicer.database.weaviate_service.weaviate.connect_to_custom")
    mock_client = MagicMock()
    mock_connect.return_value = mock_client

    with WeaviateService() as service:
        assert service._client is not None

    # After exiting the context, close should have been called
    assert service._client is None
    assert mock_client.close.call_count == 1

@pytest.mark.unit
def test_weaviate_service_destructor_mock_unit(mock_service):
    service, mock_client = mock_service
    
    # Verify that before deletion the client exists
    assert service._client is not None
    
    # Manually invoke the destructor
    # In reality, this is done by the Garbage Collector when there are no references to the object
    service.__del__()

    # Verify that the destructor initiated the close
    assert mock_client.close.call_count == 1
    assert service._client is None
    
@pytest.mark.integration
def test_weaviate_service_context_manager_integration(integration_service):
    # integration_service gives us an already created service,
    # but to test the context manager we will create our own here
    # to see the full 'with' cycle.
    from topicer.database.weaviate_service import WeaviateService
    
    with WeaviateService() as service:
        assert service._client.is_connected() is True
        # Let's try a real query (the collection was created in the fixture)
        coll = service._client.collections.use(service.chunks_collection)
        assert coll.exists() is True
    
    # After exiting the 'with'
    assert service._client is None
    
@pytest.mark.integration
def test_error_during_query_closes_connection_integration(integration_service):
    from topicer.database.weaviate_service import WeaviateService
    
    service_ref = None
    try:
        with WeaviateService() as service:
            service_ref = service
            # Let's raise an artificial error in the middle of working with the service
            raise ValueError("Simulated error during work")
    except ValueError:
        pass

    # Even with the error, the connection must be closed
    assert service_ref._client is None
    
@pytest.mark.integration
def test_integration_explicit_close_and_reconnect_integration(integration_service):
    service = integration_service
    
    # Verify that the connection is alive
    assert service._client.is_connected() is True
    
    # Close it
    service.close()
    
    # Verify that our class cleaned up the reference
    assert service._client is None
    
    # Verify idempotency in a real environment
    service.close() # Should not crash
    assert service._client is None
    
    # Reconnect by calling connect
    service.connect()
    assert service._client.is_connected() is True
    service.close()
    assert service._client is None
    