from fastapi import APIRouter

from app.providers.datasource import DataSourceFactory
from app.providers.llm import LLMProviderFactory
from app.schemas.provider import ProviderType

router = APIRouter()


@router.get("/provider-types")
async def get_provider_types():
    """Get all available LLM provider types."""
    return {
        "provider_types": [
            {"value": pt.value, "label": pt.value.title()}
            for pt in LLMProviderFactory.get_all_provider_types()
        ]
    }


@router.get("/provider-types/{provider_type}/models")
async def get_models_for_provider(provider_type: ProviderType):
    """Get available models for a specific provider type."""
    models = LLMProviderFactory.get_available_models(provider_type)
    return {"models": models}


@router.get("/datasource-types")
async def get_datasource_types():
    """Get all available data source types."""
    return {
        "datasource_types": [
            {"value": dt.value, "label": dt.value.title()}
            for dt in DataSourceFactory.get_all_types()
        ]
    }
