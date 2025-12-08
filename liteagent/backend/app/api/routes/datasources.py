from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.datasource_service import DataSourceService
from app.schemas.datasource import (
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceResponse,
)
from app.providers.datasource import FileDataSourceProvider

router = APIRouter()


@router.post("/", response_model=DataSourceResponse)
async def create_datasource(
    data: DataSourceCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new data source."""
    service = DataSourceService(db)
    try:
        datasource = await service.create(data)
        return datasource
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload", response_model=DataSourceResponse)
async def upload_file(
    name: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a file as a data source."""
    # Save the file
    file_provider = FileDataSourceProvider()
    content = await file.read()
    filename = await file_provider.save_uploaded_file(file.filename, content)

    # Fetch the content
    result = await file_provider.fetch_content(filename)

    # Create data source record
    from app.models.datasource import DataSource, DataSourceType as DSType

    datasource = DataSource(
        name=name,
        source_type=DSType.FILE,
        content=result.content,
        source_path=filename,
    )
    db.add(datasource)
    await db.commit()
    await db.refresh(datasource)
    return datasource


@router.get("/", response_model=list[DataSourceResponse])
async def list_datasources(
    db: AsyncSession = Depends(get_db),
):
    """List all data sources."""
    service = DataSourceService(db)
    datasources = await service.get_all()
    return datasources


@router.get("/{datasource_id}", response_model=DataSourceResponse)
async def get_datasource(
    datasource_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific data source."""
    service = DataSourceService(db)
    datasource = await service.get_by_id(datasource_id)
    if not datasource:
        raise HTTPException(status_code=404, detail="Data source not found")
    return datasource


@router.patch("/{datasource_id}", response_model=DataSourceResponse)
async def update_datasource(
    datasource_id: str,
    data: DataSourceUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a data source."""
    service = DataSourceService(db)
    datasource = await service.update(datasource_id, data)
    if not datasource:
        raise HTTPException(status_code=404, detail="Data source not found")
    return datasource


@router.delete("/{datasource_id}")
async def delete_datasource(
    datasource_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a data source."""
    service = DataSourceService(db)
    success = await service.delete(datasource_id)
    if not success:
        raise HTTPException(status_code=404, detail="Data source not found")
    return {"status": "deleted"}


@router.post("/{datasource_id}/refresh", response_model=DataSourceResponse)
async def refresh_datasource(
    datasource_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Refresh content for a URL data source."""
    service = DataSourceService(db)
    datasource = await service.refresh_content(datasource_id)
    if not datasource:
        raise HTTPException(
            status_code=400,
            detail="Data source not found or not a URL type",
        )
    return datasource
