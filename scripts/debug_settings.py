from shafi.config import get_settings
import os

print(f"ENV EMBED_DIMENSIONS: {os.environ.get('EMBED_DIMENSIONS')}")
print(f"ENV EMBED_MODEL: {os.environ.get('EMBED_MODEL')}")

settings = get_settings()
print(f"Settings dimensions: {settings.embedding.dimensions}")
print(f"Settings model: {settings.embedding.model}")
