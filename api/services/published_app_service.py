"""
Published App Service - Business logic for app publishing and external exposure.

This service provides methods for publishing apps for external consumption,
managing versions, and controlling access to published apps.
"""

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select

from extensions.ext_database import db
from models.model import App
from models.published_app import (
    PublishedApp,
    PublishedAppStatus,
    PublishedAppVersion,
    PublishedAppVisibility,
    PublishedAppWebhook,
)


class PublishedAppNotFoundError(Exception):
    """Raised when a published app is not found."""

    def __init__(self, message: str = "Published app not found."):
        self.message = message
        super().__init__(self.message)


class PublishedAppAlreadyExistsError(Exception):
    """Raised when trying to publish an app that is already published."""

    def __init__(self, message: str = "This app is already published."):
        self.message = message
        super().__init__(self.message)


class PublishedAppService:
    """Service for managing published apps."""

    # ==========================================
    # Publishing Methods
    # ==========================================

    @classmethod
    def publish_app(
        cls,
        app: App,
        name: str,
        created_by: str,
        description: str | None = None,
        icon: str | None = None,
        icon_background: str | None = None,
        version: str = "1.0.0",
        changelog: str | None = None,
        visibility: PublishedAppVisibility = PublishedAppVisibility.PRIVATE,
        default_rate_limit_rpm: int = 60,
        default_rate_limit_rph: int = 1000,
        default_rate_limit_rpd: int | None = None,
        free_quota_per_consumer: int | None = None,
        terms_of_service: str | None = None,
        privacy_policy: str | None = None,
        documentation_url: str | None = None,
        support_email: str | None = None,
    ) -> PublishedApp:
        """
        Publish an app for external consumption.

        Args:
            app: The app to publish
            name: Public name for the published app
            created_by: ID of the user publishing the app
            description: Public description
            icon: Icon URL or identifier
            icon_background: Icon background color
            version: Initial version string
            changelog: Initial changelog
            visibility: Visibility level (private, unlisted, public)
            default_rate_limit_rpm: Default requests per minute limit
            default_rate_limit_rph: Default requests per hour limit
            default_rate_limit_rpd: Default requests per day limit
            free_quota_per_consumer: Free quota for new consumers
            terms_of_service: Terms of service text
            privacy_policy: Privacy policy text
            documentation_url: URL to documentation
            support_email: Support email address

        Returns:
            The created PublishedApp

        Raises:
            PublishedAppAlreadyExistsError: If app is already published
        """
        # Check if already published
        existing = cls.get_published_app_by_app_id(app.id, app.tenant_id)
        if existing:
            raise PublishedAppAlreadyExistsError()

        # Generate unique slug
        slug = PublishedApp.generate_slug(name)

        published_app = PublishedApp(
            tenant_id=app.tenant_id,
            app_id=app.id,
            slug=slug,
            name=name,
            description=description,
            icon=icon,
            icon_background=icon_background,
            version=version,
            changelog=changelog,
            status=PublishedAppStatus.DRAFT.value,
            visibility=visibility.value,
            default_rate_limit_rpm=default_rate_limit_rpm,
            default_rate_limit_rph=default_rate_limit_rph,
            default_rate_limit_rpd=default_rate_limit_rpd,
            free_quota_per_consumer=free_quota_per_consumer,
            terms_of_service=terms_of_service,
            privacy_policy=privacy_policy,
            documentation_url=documentation_url,
            support_email=support_email,
            created_by=created_by,
        )

        db.session.add(published_app)
        db.session.commit()

        # Create initial version
        cls.create_version(
            published_app=published_app,
            version=version,
            created_by=created_by,
            changelog=changelog,
            is_current=True,
        )

        return published_app

    @classmethod
    def go_live(cls, published_app_id: str, published_by: str) -> PublishedApp:
        """
        Set a published app to live status.

        Args:
            published_app_id: The published app ID
            published_by: ID of the user publishing

        Returns:
            The updated PublishedApp
        """
        published_app = cls.get_published_app(published_app_id)
        if not published_app:
            raise PublishedAppNotFoundError()

        published_app.status = PublishedAppStatus.PUBLISHED.value
        published_app.published_by = published_by
        published_app.published_at = datetime.utcnow()
        db.session.commit()

        return published_app

    @classmethod
    def pause(cls, published_app_id: str) -> PublishedApp:
        """Temporarily pause a published app."""
        published_app = cls.get_published_app(published_app_id)
        if not published_app:
            raise PublishedAppNotFoundError()

        published_app.status = PublishedAppStatus.PAUSED.value
        db.session.commit()

        return published_app

    @classmethod
    def deprecate(cls, published_app_id: str) -> PublishedApp:
        """Mark a published app as deprecated."""
        published_app = cls.get_published_app(published_app_id)
        if not published_app:
            raise PublishedAppNotFoundError()

        published_app.status = PublishedAppStatus.DEPRECATED.value
        db.session.commit()

        return published_app

    @classmethod
    def archive(cls, published_app_id: str) -> PublishedApp:
        """Permanently archive a published app."""
        published_app = cls.get_published_app(published_app_id)
        if not published_app:
            raise PublishedAppNotFoundError()

        published_app.status = PublishedAppStatus.ARCHIVED.value
        db.session.commit()

        return published_app

    @classmethod
    def unpublish(cls, published_app_id: str) -> bool:
        """
        Completely unpublish an app (remove from external access).

        Args:
            published_app_id: The published app ID

        Returns:
            True if unpublished successfully
        """
        published_app = cls.get_published_app(published_app_id)
        if not published_app:
            return False

        # Delete webhooks
        db.session.execute(
            PublishedAppWebhook.__table__.delete().where(
                PublishedAppWebhook.published_app_id == published_app_id
            )
        )

        # Delete versions
        db.session.execute(
            PublishedAppVersion.__table__.delete().where(
                PublishedAppVersion.published_app_id == published_app_id
            )
        )

        # Delete the published app
        db.session.delete(published_app)
        db.session.commit()

        return True

    # ==========================================
    # CRUD Methods
    # ==========================================

    @classmethod
    def get_published_app(cls, published_app_id: str) -> PublishedApp | None:
        """Get a published app by ID."""
        return db.session.get(PublishedApp, published_app_id)

    @classmethod
    def get_published_app_by_slug(cls, slug: str) -> PublishedApp | None:
        """Get a published app by its public slug."""
        return db.session.scalar(
            select(PublishedApp).where(PublishedApp.slug == slug)
        )

    @classmethod
    def get_published_app_by_app_id(
        cls, app_id: str, tenant_id: str
    ) -> PublishedApp | None:
        """Get a published app by its internal app ID."""
        return db.session.scalar(
            select(PublishedApp).where(
                PublishedApp.app_id == app_id,
                PublishedApp.tenant_id == tenant_id,
            )
        )

    @classmethod
    def get_published_apps(
        cls,
        tenant_id: str,
        status: PublishedAppStatus | None = None,
        visibility: PublishedAppVisibility | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Sequence[PublishedApp], int]:
        """
        Get published apps for a tenant with optional filtering.

        Returns:
            Tuple of (published_apps, total_count)
        """
        query = select(PublishedApp).where(PublishedApp.tenant_id == tenant_id)

        if status:
            query = query.where(PublishedApp.status == status.value)
        if visibility:
            query = query.where(PublishedApp.visibility == visibility.value)

        # Get total count
        count_query = select(PublishedApp.id).where(PublishedApp.tenant_id == tenant_id)
        if status:
            count_query = count_query.where(PublishedApp.status == status.value)
        if visibility:
            count_query = count_query.where(PublishedApp.visibility == visibility.value)
        total = len(db.session.scalars(count_query).all())

        # Get paginated results
        query = query.order_by(PublishedApp.created_at.desc())
        query = query.offset(offset).limit(limit)
        apps = db.session.scalars(query).all()

        return apps, total

    @classmethod
    def get_public_apps(
        cls,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[PublishedApp]:
        """Get all publicly visible and live published apps."""
        query = select(PublishedApp).where(
            PublishedApp.status == PublishedAppStatus.PUBLISHED.value,
            PublishedApp.visibility == PublishedAppVisibility.PUBLIC.value,
        )
        query = query.order_by(PublishedApp.total_requests.desc())
        query = query.offset(offset).limit(limit)

        return db.session.scalars(query).all()

    @classmethod
    def update_published_app(
        cls,
        published_app_id: str,
        name: str | None = None,
        description: str | None = None,
        icon: str | None = None,
        icon_background: str | None = None,
        visibility: PublishedAppVisibility | None = None,
        default_rate_limit_rpm: int | None = None,
        default_rate_limit_rph: int | None = None,
        default_rate_limit_rpd: int | None = None,
        free_quota_per_consumer: int | None = None,
        terms_of_service: str | None = None,
        privacy_policy: str | None = None,
        documentation_url: str | None = None,
        support_email: str | None = None,
        custom_domain: str | None = None,
        cors_allowed_origins: str | None = None,
        webhook_url: str | None = None,
    ) -> PublishedApp:
        """Update published app details."""
        published_app = cls.get_published_app(published_app_id)
        if not published_app:
            raise PublishedAppNotFoundError()

        if name is not None:
            published_app.name = name
        if description is not None:
            published_app.description = description
        if icon is not None:
            published_app.icon = icon
        if icon_background is not None:
            published_app.icon_background = icon_background
        if visibility is not None:
            published_app.visibility = visibility.value
        if default_rate_limit_rpm is not None:
            published_app.default_rate_limit_rpm = default_rate_limit_rpm
        if default_rate_limit_rph is not None:
            published_app.default_rate_limit_rph = default_rate_limit_rph
        if default_rate_limit_rpd is not None:
            published_app.default_rate_limit_rpd = default_rate_limit_rpd
        if free_quota_per_consumer is not None:
            published_app.free_quota_per_consumer = free_quota_per_consumer
        if terms_of_service is not None:
            published_app.terms_of_service = terms_of_service
        if privacy_policy is not None:
            published_app.privacy_policy = privacy_policy
        if documentation_url is not None:
            published_app.documentation_url = documentation_url
        if support_email is not None:
            published_app.support_email = support_email
        if custom_domain is not None:
            published_app.custom_domain = custom_domain
        if cors_allowed_origins is not None:
            published_app.cors_allowed_origins = cors_allowed_origins
        if webhook_url is not None:
            published_app.webhook_url = webhook_url

        db.session.commit()
        return published_app

    # ==========================================
    # Version Management
    # ==========================================

    @classmethod
    def create_version(
        cls,
        published_app: PublishedApp,
        version: str,
        created_by: str,
        changelog: str | None = None,
        workflow_version: str | None = None,
        config_snapshot: str | None = None,
        is_current: bool = False,
    ) -> PublishedAppVersion:
        """
        Create a new version for a published app.

        Args:
            published_app: The published app
            version: Version string (e.g., "1.0.1")
            created_by: ID of the user creating the version
            changelog: What changed in this version
            workflow_version: Reference to workflow version
            config_snapshot: JSON snapshot of configuration
            is_current: Whether this is the current version

        Returns:
            The created PublishedAppVersion
        """
        # If setting as current, unset previous current
        if is_current:
            db.session.execute(
                PublishedAppVersion.__table__.update()
                .where(
                    PublishedAppVersion.published_app_id == published_app.id,
                    PublishedAppVersion.is_current == True,  # noqa: E712
                )
                .values(is_current=False)
            )

        version_record = PublishedAppVersion(
            published_app_id=published_app.id,
            version=version,
            workflow_version=workflow_version,
            config_snapshot=config_snapshot,
            changelog=changelog,
            is_current=is_current,
            created_by=created_by,
        )

        db.session.add(version_record)

        # Update main version
        if is_current:
            published_app.version = version
            published_app.changelog = changelog

        db.session.commit()
        return version_record

    @classmethod
    def get_versions(
        cls, published_app_id: str
    ) -> Sequence[PublishedAppVersion]:
        """Get all versions for a published app."""
        return db.session.scalars(
            select(PublishedAppVersion)
            .where(PublishedAppVersion.published_app_id == published_app_id)
            .order_by(PublishedAppVersion.created_at.desc())
        ).all()

    @classmethod
    def get_current_version(
        cls, published_app_id: str
    ) -> PublishedAppVersion | None:
        """Get the current version for a published app."""
        return db.session.scalar(
            select(PublishedAppVersion).where(
                PublishedAppVersion.published_app_id == published_app_id,
                PublishedAppVersion.is_current == True,  # noqa: E712
            )
        )

    @classmethod
    def set_current_version(
        cls, published_app_id: str, version: str
    ) -> PublishedAppVersion:
        """Set a specific version as current."""
        # Find the version
        version_record = db.session.scalar(
            select(PublishedAppVersion).where(
                PublishedAppVersion.published_app_id == published_app_id,
                PublishedAppVersion.version == version,
            )
        )

        if not version_record:
            raise PublishedAppNotFoundError(f"Version {version} not found.")

        # Unset previous current
        db.session.execute(
            PublishedAppVersion.__table__.update()
            .where(
                PublishedAppVersion.published_app_id == published_app_id,
                PublishedAppVersion.is_current == True,  # noqa: E712
            )
            .values(is_current=False)
        )

        # Set new current
        version_record.is_current = True

        # Update main app version
        published_app = cls.get_published_app(published_app_id)
        if published_app:
            published_app.version = version
            published_app.changelog = version_record.changelog

        db.session.commit()
        return version_record

    @classmethod
    def deprecate_version(
        cls, published_app_id: str, version: str, message: str | None = None
    ) -> PublishedAppVersion:
        """Mark a specific version as deprecated."""
        version_record = db.session.scalar(
            select(PublishedAppVersion).where(
                PublishedAppVersion.published_app_id == published_app_id,
                PublishedAppVersion.version == version,
            )
        )

        if not version_record:
            raise PublishedAppNotFoundError(f"Version {version} not found.")

        version_record.is_deprecated = True
        version_record.deprecation_message = message
        db.session.commit()

        return version_record

    # ==========================================
    # Webhook Management
    # ==========================================

    @classmethod
    def add_webhook(
        cls,
        published_app_id: str,
        tenant_id: str,
        name: str,
        url: str,
        events: list[str],
        secret: str | None = None,
    ) -> PublishedAppWebhook:
        """Add a webhook to a published app."""
        import json

        webhook = PublishedAppWebhook(
            published_app_id=published_app_id,
            tenant_id=tenant_id,
            name=name,
            url=url,
            secret=secret,
            events=json.dumps(events),
        )

        db.session.add(webhook)
        db.session.commit()
        return webhook

    @classmethod
    def get_webhooks(cls, published_app_id: str) -> Sequence[PublishedAppWebhook]:
        """Get all webhooks for a published app."""
        return db.session.scalars(
            select(PublishedAppWebhook).where(
                PublishedAppWebhook.published_app_id == published_app_id
            )
        ).all()

    @classmethod
    def delete_webhook(cls, webhook_id: str) -> bool:
        """Delete a webhook."""
        webhook = db.session.get(PublishedAppWebhook, webhook_id)
        if not webhook:
            return False

        db.session.delete(webhook)
        db.session.commit()
        return True

    # ==========================================
    # Analytics Methods
    # ==========================================

    @classmethod
    def increment_request_count(cls, published_app_id: str) -> None:
        """Increment the total request count for a published app."""
        published_app = cls.get_published_app(published_app_id)
        if published_app:
            published_app.total_requests += 1
            db.session.commit()

    @classmethod
    def increment_consumer_count(cls, published_app_id: str) -> None:
        """Increment the total consumer count for a published app."""
        published_app = cls.get_published_app(published_app_id)
        if published_app:
            published_app.total_consumers += 1
            db.session.commit()

    # ==========================================
    # Access Control Helpers
    # ==========================================

    @classmethod
    def is_accessible(cls, published_app: PublishedApp) -> bool:
        """Check if a published app is accessible to the public."""
        return published_app.is_live

    @classmethod
    def is_publicly_discoverable(cls, published_app: PublishedApp) -> bool:
        """Check if a published app should appear in public listings."""
        return published_app.is_public
