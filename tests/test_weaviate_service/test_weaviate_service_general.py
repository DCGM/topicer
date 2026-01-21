
import pytest
from topicer.database.weaviate_service import WeaviateService
from unittest.mock import AsyncMock

# -------------------- UNIT TESTS --------------------

@pytest.mark.unit
async def test_weaviate_service_cleanup_mock_unit(mock_service):
    service, mock_client = mock_service

    # Verify manual close
    await service.close()

    assert mock_client.close.call_count == 1
    assert service._client is None


@pytest.mark.unit
async def test_service_close_idempotency_unit(mock_service):
    service, mock_client = mock_service

    # Close it multiple times
    await service.close()
    await service.close()
    await service.close()

    assert mock_client.close.call_count == 1
    assert service._client is None


@pytest.mark.unit
async def test_weaviate_service_context_manager_unit(mocker):
    mock_connect = mocker.patch(
        "topicer.database.weaviate_service.weaviate.use_async_with_custom")
    mock_client = AsyncMock()
    mock_connect.return_value = mock_client

    async with WeaviateService() as service:
        assert service._client is not None

    # After exiting the context, close should have been called
    assert service._client is None
    assert mock_client.close.call_count == 1
    
# -------------------- INTEGRATION TESTS --------------------
    
@pytest.mark.integration
async def test_weaviate_service_context_manager_integration(integration_service):
    # Integration_service gives us an already created service,
    # but to test the context manager we will create our own here
    # to see the full 'with' cycle.
    # However, we use the integration fixture to create the collection beforehand.
    
    async with WeaviateService() as service:
        assert service._client.is_connected() is True
        # Let's try a real query (the collection was created in the fixture)
        coll = service._client.collections.use(service.chunks_collection)
        assert await coll.exists() is True
    
    # After exiting the 'with'
    assert service._client is None
    
@pytest.mark.integration
async def test_error_during_query_closes_connection_integration():
    
    service_ref = None
    try:
        async with WeaviateService() as service:
            service_ref = service
            # Let's raise an artificial error in the middle of working with the service
            raise ValueError("Simulated error during work")
    except ValueError:
        pass

    # Even with the error, the connection must be closed
    assert service_ref._client is None
    
@pytest.mark.integration
async def test_integration_explicit_close_and_reconnect_integration(integration_service):
    service = integration_service
    
    # Verify that the connection is alive
    assert service._client.is_connected() is True
    
    # Close it
    await service.close()
    
    # Verify that our class cleaned up the reference
    assert service._client is None
    
    # Verify idempotency in a real environment
    await service.close() # Should not crash
    assert service._client is None
    
    # Reconnect by calling connect
    await service.connect()
    assert service._client.is_connected() is True
    await service.close()
    assert service._client is None
    