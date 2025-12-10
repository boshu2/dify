"""
Scope-based authentication decorators for service API endpoints.

This module provides decorators for enforcing API token scopes
on service API endpoints.
"""

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from flask import request
from werkzeug.exceptions import Forbidden, Unauthorized

from models.api_token_scope import ApiScope
from services.api_token_scope_service import (
    ApiTokenScopeService,
    ScopeNotAllowedError,
    TokenExpiredError,
    TokenIPRestrictionError,
)

P = ParamSpec("P")
R = TypeVar("R")


def require_scope(
    scope: str | ApiScope,
    *,
    check_config: bool = True,
):
    """
    Decorator to require a specific scope for an endpoint.

    This decorator should be used after validate_app_token or validate_dataset_token.
    It checks if the authenticated API token has the required scope.

    Args:
        scope: The scope required for this endpoint
        check_config: Whether to also validate token config (expiration, IP, etc.)

    Usage:
        @validate_app_token
        @require_scope(ApiScope.APP_INVOKE)
        def my_endpoint(app_model):
            ...

    Raises:
        Forbidden: If token doesn't have the required scope
        Unauthorized: If token has expired or IP is not allowed
    """
    def decorator(view_func: Callable[P, R]) -> Callable[P, R]:
        @wraps(view_func)
        def decorated_view(*args: P.args, **kwargs: P.kwargs) -> R:
            # Get token_id from the request context
            # The token is set by validate_app_token or validate_dataset_token
            api_token = getattr(request, '_api_token', None)

            if not api_token:
                # Try to get from authorization header if not set
                from controllers.service_api.wraps import validate_and_get_api_token
                try:
                    # Determine scope type from endpoint
                    scope_type = "app"  # Default to app
                    api_token = validate_and_get_api_token(scope_type)
                except Exception:
                    raise Unauthorized("API token not found in request context.")

            token_id = api_token.id

            # Check scope
            try:
                ApiTokenScopeService.validate_token_scope(token_id, scope)
            except ScopeNotAllowedError as e:
                raise Forbidden(str(e))

            # Check config (expiration, IP, etc.)
            if check_config:
                try:
                    ip_address = request.remote_addr
                    referrer = request.referrer
                    ApiTokenScopeService.validate_token_config(
                        token_id,
                        ip_address=ip_address,
                        referrer=referrer,
                    )
                except TokenExpiredError:
                    raise Unauthorized("API token has expired.")
                except TokenIPRestrictionError as e:
                    raise Forbidden(str(e))

            return view_func(*args, **kwargs)

        return decorated_view

    return decorator


def require_any_scope(*scopes: str | ApiScope, check_config: bool = True):
    """
    Decorator to require any one of the specified scopes.

    Args:
        *scopes: Variable number of scopes, at least one is required
        check_config: Whether to also validate token config

    Usage:
        @validate_app_token
        @require_any_scope(ApiScope.APP_INVOKE, ApiScope.WORKFLOW_RUN)
        def my_endpoint(app_model):
            ...
    """
    def decorator(view_func: Callable[P, R]) -> Callable[P, R]:
        @wraps(view_func)
        def decorated_view(*args: P.args, **kwargs: P.kwargs) -> R:
            api_token = getattr(request, '_api_token', None)

            if not api_token:
                from controllers.service_api.wraps import validate_and_get_api_token
                try:
                    api_token = validate_and_get_api_token("app")
                except Exception:
                    raise Unauthorized("API token not found in request context.")

            token_id = api_token.id

            # Check if token has any of the required scopes
            if not ApiTokenScopeService.has_any_scope(token_id, list(scopes)):
                scope_names = [s.value if isinstance(s, ApiScope) else s for s in scopes]
                raise Forbidden(
                    f"API token requires at least one of these scopes: {', '.join(scope_names)}"
                )

            # Check config
            if check_config:
                try:
                    ApiTokenScopeService.validate_token_config(
                        token_id,
                        ip_address=request.remote_addr,
                        referrer=request.referrer,
                    )
                except TokenExpiredError:
                    raise Unauthorized("API token has expired.")
                except TokenIPRestrictionError as e:
                    raise Forbidden(str(e))

            return view_func(*args, **kwargs)

        return decorated_view

    return decorator


def require_all_scopes(*scopes: str | ApiScope, check_config: bool = True):
    """
    Decorator to require all of the specified scopes.

    Args:
        *scopes: Variable number of scopes, all are required
        check_config: Whether to also validate token config

    Usage:
        @validate_app_token
        @require_all_scopes(ApiScope.CONVERSATION_READ, ApiScope.CONVERSATION_WRITE)
        def my_endpoint(app_model):
            ...
    """
    def decorator(view_func: Callable[P, R]) -> Callable[P, R]:
        @wraps(view_func)
        def decorated_view(*args: P.args, **kwargs: P.kwargs) -> R:
            api_token = getattr(request, '_api_token', None)

            if not api_token:
                from controllers.service_api.wraps import validate_and_get_api_token
                try:
                    api_token = validate_and_get_api_token("app")
                except Exception:
                    raise Unauthorized("API token not found in request context.")

            token_id = api_token.id

            # Check if token has all required scopes
            if not ApiTokenScopeService.has_all_scopes(token_id, list(scopes)):
                scope_names = [s.value if isinstance(s, ApiScope) else s for s in scopes]
                raise Forbidden(
                    f"API token requires all of these scopes: {', '.join(scope_names)}"
                )

            # Check config
            if check_config:
                try:
                    ApiTokenScopeService.validate_token_config(
                        token_id,
                        ip_address=request.remote_addr,
                        referrer=request.referrer,
                    )
                except TokenExpiredError:
                    raise Unauthorized("API token has expired.")
                except TokenIPRestrictionError as e:
                    raise Forbidden(str(e))

            return view_func(*args, **kwargs)

        return decorated_view

    return decorator


def optional_scope_check(scope: str | ApiScope):
    """
    Decorator that adds scope information to request but doesn't enforce.

    This is useful for endpoints that behave differently based on scope.
    The scope status is available via request.has_scope.

    Usage:
        @validate_app_token
        @optional_scope_check(ApiScope.ADMIN_ALL)
        def my_endpoint(app_model):
            if request.has_scope:
                # Admin behavior
                ...
            else:
                # Normal behavior
                ...
    """
    def decorator(view_func: Callable[P, R]) -> Callable[P, R]:
        @wraps(view_func)
        def decorated_view(*args: P.args, **kwargs: P.kwargs) -> R:
            api_token = getattr(request, '_api_token', None)

            has_scope = False
            if api_token:
                has_scope = ApiTokenScopeService.has_scope(api_token.id, scope)

            # Attach to request for use in view
            request.has_scope = has_scope  # type: ignore
            request.checked_scope = scope  # type: ignore

            return view_func(*args, **kwargs)

        return decorated_view

    return decorator


# Convenience decorators for common scope checks
def require_invoke_scope(view_func: Callable[P, R] | None = None):
    """Require app:invoke scope."""
    decorator = require_scope(ApiScope.APP_INVOKE)
    if view_func:
        return decorator(view_func)
    return decorator


def require_invoke_stream_scope(view_func: Callable[P, R] | None = None):
    """Require app:invoke:stream scope."""
    decorator = require_scope(ApiScope.APP_INVOKE_STREAM)
    if view_func:
        return decorator(view_func)
    return decorator


def require_conversation_read_scope(view_func: Callable[P, R] | None = None):
    """Require conversation:read scope."""
    decorator = require_scope(ApiScope.CONVERSATION_READ)
    if view_func:
        return decorator(view_func)
    return decorator


def require_file_upload_scope(view_func: Callable[P, R] | None = None):
    """Require file:upload scope."""
    decorator = require_scope(ApiScope.FILE_UPLOAD)
    if view_func:
        return decorator(view_func)
    return decorator


def require_workflow_scope(view_func: Callable[P, R] | None = None):
    """Require workflow:run scope."""
    decorator = require_scope(ApiScope.WORKFLOW_RUN)
    if view_func:
        return decorator(view_func)
    return decorator


def require_admin_scope(view_func: Callable[P, R] | None = None):
    """Require admin:all scope."""
    decorator = require_scope(ApiScope.ADMIN_ALL)
    if view_func:
        return decorator(view_func)
    return decorator
