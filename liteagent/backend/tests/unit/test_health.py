"""
Unit tests for health check system.
Tests liveness, readiness, and dependency health checks.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from app.core.health import (
    HealthStatus,
    HealthCheck,
    HealthCheckResult,
    HealthCheckRegistry,
    DatabaseHealthCheck,
    ExternalServiceHealthCheck,
    DiskSpaceHealthCheck,
    MemoryHealthCheck,
    CompositeHealthCheck,
)


class TestHealthStatus:
    """Tests for health status enum."""

    def test_status_values(self):
        """Test health status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"

    def test_status_is_healthy(self):
        """Test is_healthy property."""
        assert HealthStatus.HEALTHY.is_ok is True
        assert HealthStatus.DEGRADED.is_ok is True
        assert HealthStatus.UNHEALTHY.is_ok is False


class TestHealthCheckResult:
    """Tests for health check result."""

    def test_create_healthy_result(self):
        """Test creating a healthy result."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
        )
        assert result.status == HealthStatus.HEALTHY
        assert result.is_healthy is True

    def test_create_unhealthy_result(self):
        """Test creating an unhealthy result."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.UNHEALTHY,
            message="Connection failed",
            error="ConnectionError",
        )
        assert result.status == HealthStatus.UNHEALTHY
        assert result.is_healthy is False
        assert result.error == "ConnectionError"

    def test_result_has_timestamp(self):
        """Test result includes timestamp."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert result.checked_at is not None

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = HealthCheckResult(
            name="db",
            status=HealthStatus.HEALTHY,
            message="Connected",
            latency_ms=5.2,
        )
        data = result.to_dict()

        assert data["name"] == "db"
        assert data["status"] == "healthy"
        assert data["message"] == "Connected"
        assert data["latency_ms"] == 5.2


class TestHealthCheck:
    """Tests for base health check."""

    def test_health_check_has_name(self):
        """Test health check has a name."""
        # Use DiskSpaceHealthCheck as a concrete implementation
        check = DiskSpaceHealthCheck(name="test-check", path="/")
        assert check.name == "test-check"

    def test_health_check_has_timeout(self):
        """Test health check has timeout."""
        check = DiskSpaceHealthCheck(name="test", path="/", timeout_seconds=5.0)
        assert check.timeout_seconds == 5.0


class TestDatabaseHealthCheck:
    """Tests for database health check."""

    @pytest.mark.asyncio
    async def test_healthy_database(self):
        """Test healthy database check."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=Mock())

        check = DatabaseHealthCheck(
            name="postgres",
            get_session=lambda: mock_session,
        )
        result = await check.check()

        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_unhealthy_database(self):
        """Test unhealthy database check."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=Exception("Connection refused"))

        check = DatabaseHealthCheck(
            name="postgres",
            get_session=lambda: mock_session,
        )
        result = await check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection refused" in result.error


class TestExternalServiceHealthCheck:
    """Tests for external service health check."""

    @pytest.mark.asyncio
    async def test_healthy_service(self):
        """Test healthy external service."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        check = ExternalServiceHealthCheck(
            name="api",
            url="https://api.example.com/health",
            client=mock_client,
        )
        result = await check.check()

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_service(self):
        """Test unhealthy external service."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_response)

        check = ExternalServiceHealthCheck(
            name="api",
            url="https://api.example.com/health",
            client=mock_client,
        )
        result = await check.check()

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_service_timeout(self):
        """Test external service timeout."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=TimeoutError("Request timed out"))

        check = ExternalServiceHealthCheck(
            name="api",
            url="https://api.example.com/health",
            client=mock_client,
        )
        result = await check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.error.lower()


class TestDiskSpaceHealthCheck:
    """Tests for disk space health check."""

    @pytest.mark.asyncio
    async def test_sufficient_disk_space(self):
        """Test sufficient disk space."""
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = Mock(
                total=100_000_000_000,  # 100GB
                used=50_000_000_000,  # 50GB
                free=50_000_000_000,  # 50GB
            )

            check = DiskSpaceHealthCheck(
                name="disk",
                path="/",
                min_free_percent=10,
            )
            result = await check.check()

            assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_low_disk_space(self):
        """Test low disk space warning."""
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = Mock(
                total=100_000_000_000,  # 100GB
                used=95_000_000_000,  # 95GB
                free=5_000_000_000,  # 5GB (5%)
            )

            check = DiskSpaceHealthCheck(
                name="disk",
                path="/",
                min_free_percent=10,
            )
            result = await check.check()

            assert result.status == HealthStatus.DEGRADED


class TestMemoryHealthCheck:
    """Tests for memory health check."""

    @pytest.mark.asyncio
    async def test_sufficient_memory(self):
        """Test sufficient memory available."""
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = Mock(
                total=16_000_000_000,  # 16GB
                available=8_000_000_000,  # 8GB
                percent=50.0,
            )

            check = MemoryHealthCheck(
                name="memory",
                max_used_percent=90,
            )
            result = await check.check()

            assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_high_memory_usage(self):
        """Test high memory usage warning."""
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = Mock(
                total=16_000_000_000,
                available=1_000_000_000,
                percent=93.0,
            )

            check = MemoryHealthCheck(
                name="memory",
                max_used_percent=90,
            )
            result = await check.check()

            assert result.status == HealthStatus.DEGRADED


class TestHealthCheckRegistry:
    """Tests for health check registry."""

    def test_register_check(self):
        """Test registering a health check."""
        registry = HealthCheckRegistry()
        check = DiskSpaceHealthCheck(name="test", path="/")
        registry.register(check)

        assert "test" in registry.checks

    def test_register_multiple_checks(self):
        """Test registering multiple health checks."""
        registry = HealthCheckRegistry()
        registry.register(DiskSpaceHealthCheck(name="check1", path="/"))
        registry.register(MemoryHealthCheck(name="check2"))

        assert len(registry.checks) == 2

    @pytest.mark.asyncio
    async def test_run_all_checks(self):
        """Test running all registered checks."""
        registry = HealthCheckRegistry()

        check1 = Mock(spec=HealthCheck)
        check1.name = "check1"
        check1.check = AsyncMock(return_value=HealthCheckResult(
            name="check1",
            status=HealthStatus.HEALTHY,
        ))

        check2 = Mock(spec=HealthCheck)
        check2.name = "check2"
        check2.check = AsyncMock(return_value=HealthCheckResult(
            name="check2",
            status=HealthStatus.HEALTHY,
        ))

        registry.register(check1)
        registry.register(check2)

        results = await registry.run_all()

        assert len(results) == 2
        assert all(r.is_healthy for r in results)

    @pytest.mark.asyncio
    async def test_overall_status_healthy(self):
        """Test overall status when all checks pass."""
        registry = HealthCheckRegistry()

        check = Mock(spec=HealthCheck)
        check.name = "check"
        check.check = AsyncMock(return_value=HealthCheckResult(
            name="check",
            status=HealthStatus.HEALTHY,
        ))
        registry.register(check)

        status = await registry.get_overall_status()
        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_overall_status_unhealthy(self):
        """Test overall status when any check fails."""
        registry = HealthCheckRegistry()

        healthy_check = Mock(spec=HealthCheck)
        healthy_check.name = "healthy"
        healthy_check.check = AsyncMock(return_value=HealthCheckResult(
            name="healthy",
            status=HealthStatus.HEALTHY,
        ))

        unhealthy_check = Mock(spec=HealthCheck)
        unhealthy_check.name = "unhealthy"
        unhealthy_check.check = AsyncMock(return_value=HealthCheckResult(
            name="unhealthy",
            status=HealthStatus.UNHEALTHY,
        ))

        registry.register(healthy_check)
        registry.register(unhealthy_check)

        status = await registry.get_overall_status()
        assert status == HealthStatus.UNHEALTHY


class TestCompositeHealthCheck:
    """Tests for composite health check."""

    @pytest.mark.asyncio
    async def test_all_healthy(self):
        """Test composite when all checks healthy."""
        check1 = Mock(spec=HealthCheck)
        check1.name = "check1"
        check1.check = AsyncMock(return_value=HealthCheckResult(
            name="check1", status=HealthStatus.HEALTHY
        ))

        check2 = Mock(spec=HealthCheck)
        check2.name = "check2"
        check2.check = AsyncMock(return_value=HealthCheckResult(
            name="check2", status=HealthStatus.HEALTHY
        ))

        composite = CompositeHealthCheck(
            name="composite",
            checks=[check1, check2],
        )
        result = await composite.check()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.details["checks"]) == 2

    @pytest.mark.asyncio
    async def test_one_unhealthy(self):
        """Test composite when one check unhealthy."""
        healthy = Mock(spec=HealthCheck)
        healthy.name = "healthy"
        healthy.check = AsyncMock(return_value=HealthCheckResult(
            name="healthy", status=HealthStatus.HEALTHY
        ))

        unhealthy = Mock(spec=HealthCheck)
        unhealthy.name = "unhealthy"
        unhealthy.check = AsyncMock(return_value=HealthCheckResult(
            name="unhealthy", status=HealthStatus.UNHEALTHY
        ))

        composite = CompositeHealthCheck(
            name="composite",
            checks=[healthy, unhealthy],
        )
        result = await composite.check()

        assert result.status == HealthStatus.UNHEALTHY
