"""Live extractor tests. These hit the real network on purpose (Directive 6).

Run with:  python -m pytest tests/test_extract.py -q -s
"""

import pytest

from app.extract import (
    ExtractionError,
    NoCaptionsError,
    extract_article,
    extract_youtube,
    youtube_video_id,
)
from app.router import is_youtube, route

ARTICLE_URL = "https://karpathy.github.io/2015/05/21/rnn-effectiveness/"
YOUTUBE_URL = "https://www.youtube.com/watch?v=aircAruvnKk"
# A real video that is unavailable, so extraction fails as ExtractionError, not NoCaptionsError.
UNAVAILABLE_URL = "https://www.youtube.com/watch?v=BaW_jenozKc"


def test_youtube_video_id_shapes():
    assert youtube_video_id("https://www.youtube.com/watch?v=aircAruvnKk") == "aircAruvnKk"
    assert youtube_video_id("https://youtu.be/aircAruvnKk") == "aircAruvnKk"
    assert youtube_video_id("https://www.youtube.com/shorts/aircAruvnKk") == "aircAruvnKk"


def test_is_youtube():
    assert is_youtube(YOUTUBE_URL)
    assert is_youtube("https://youtu.be/aircAruvnKk")
    assert not is_youtube(ARTICLE_URL)


def test_extract_article_live():
    doc = extract_article(ARTICLE_URL)
    assert doc.source_type == "article"
    assert doc.title.strip()
    assert len(doc.text) > 500
    print(f"\n[article] title={doc.title!r} chars={len(doc.text)}")
    print(doc.text[:300])


def test_extract_youtube_live():
    doc = extract_youtube(YOUTUBE_URL)
    assert doc.source_type == "youtube"
    assert len(doc.text) > 500
    assert doc.anchors, "expected timestamp anchors"
    print(f"\n[youtube] title={doc.title!r} chars={len(doc.text)} anchors={len(doc.anchors)}")
    print(doc.text[:300])


def test_no_captions_raises():
    """No caption track for the requested language -> NoCaptionsError, no ASR fallback.

    NOTE: YouTube auto-generates English captions for nearly every public video, so a real
    'captions fully disabled' video was not findable. Asking a real video for a language it
    has no track in exercises the same NoTranscriptFound -> NoCaptionsError mapping.
    """
    with pytest.raises(NoCaptionsError) as excinfo:
        extract_youtube(YOUTUBE_URL, languages=("zu",))
    print(f"\n[no-captions] {excinfo.value}")


def test_unavailable_video_raises_extraction_error():
    with pytest.raises(ExtractionError) as excinfo:
        extract_youtube(UNAVAILABLE_URL)
    assert not isinstance(excinfo.value, NoCaptionsError)
    print(f"\n[unavailable] {excinfo.value}")


def test_router_dispatches_by_source():
    assert route(ARTICLE_URL).source_type == "article"
    assert route(YOUTUBE_URL).source_type == "youtube"
