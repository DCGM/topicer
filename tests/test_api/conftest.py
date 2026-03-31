import pytest

from topicer import WeaviateService


def pytest_addoption(parser):
    parser.addoption(
        "--api-url",
        action="store",
        default="http://127.0.0.1:8000/v1",
        help="Base URL for the API to be tested.",
    )
    parser.addoption(
        "--db-host",
        action="store",
        default="localhost",
        help="Database host for integration tests.",
    )

    parser.addoption(
        "--db-port",
        action="store",
        default="8080",
        help="Database port for integration tests.",
    )

    parser.addoption(
        "--db-grpc-port",
        action="store",
        default="50051",
        help="Database gRPC port for integration tests.",
    )


# Create a fixture that returns the base URL based on the environment
@pytest.fixture(scope="session")
def base_url(request):
    api_url = request.config.getoption("--api-url")

    return api_url


@pytest.fixture(scope="function")
async def database_connection(request):
    # Setup code for database connection
    service = WeaviateService(
        host=request.config.getoption("--db-host"),
        rest_port=int(request.config.getoption("--db-port")),
        grpc_port=int(request.config.getoption("--db-grpc-port")),
    )
    async with service:
        yield service

