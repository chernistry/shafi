import uvicorn

from shafi.api.app import create_app
from shafi.config import get_settings
from shafi.config.logging import setup_logging

if __name__ == "__main__":
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_format)
    uvicorn.run(
        create_app(),
        host=settings.app.host,
        port=settings.app.port,
        log_level=settings.app.log_level.lower(),
        log_config=None,
    )
