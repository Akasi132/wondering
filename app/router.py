"""URL -> the right extractor. Simple string matching, per v1 scope."""

from app.extract import extract_article, extract_youtube, is_youtube_host
from app.models import Document

# Re-exported under the router's own name for callers/tests; the implementation lives in
# app.extract so the router and the extractor share one definition of "is this YouTube".
is_youtube = is_youtube_host


def route(url: str) -> Document:
    if is_youtube(url):
        return extract_youtube(url)
    return extract_article(url)
