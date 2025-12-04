from pydantic import Field
from pydantic_settings import BaseSettings


class HuaweiCloudConfig(BaseSettings):
    """
    Configuration settings for Huawei cloud search service
    """

    HUAWEI_CLOUD_HOSTS: str | None = Field(
        description="Hostname or IP address of the Huawei cloud search service instance",
        default=None,
    )

    HUAWEI_CLOUD_USER: str | None = Field(
        description="Username for authenticating with Huawei cloud search service",
        default=None,
    )

    HUAWEI_CLOUD_PASSWORD: str | None = Field(
        description="Password for authenticating with Huawei cloud search service",
        default=None,
    )

    HUAWEI_CLOUD_VERIFY_CERTS: bool = Field(
        description="Whether to verify SSL certificates when connecting to Huawei cloud search service",
        default=True,
    )
