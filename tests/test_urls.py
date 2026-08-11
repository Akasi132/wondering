"""Adversarial, fully OFFLINE tests for URL/text helpers.

Scope: app.extract.youtube_video_id / is_youtube_host / _host / _normalize / _timestamp /
_transcript_text, plus app.router.is_youtube. Nothing here touches the network or needs an
API key. tests/test_extract.py owns the happy paths and the live calls; this file owns the
edge cases.

This file used to carry "BUG:" characterization tests that pinned known defects (minutes
running past 59, negative timestamps rendering as "-1:59", NBSP surviving before a newline,
and the router accepting *.youtube.com hosts the id parser could not parse). Those defects
are fixed in app/, so every one of those tests now asserts the CORRECT behaviour instead.
"""

import pytest

from app.extract import (
    TIMESTAMP_MARKER_SECONDS,
    ExtractionError,
    _host,
    _normalize,
    _timestamp,
    _transcript_text,
    is_youtube_host,
    youtube_video_id,
)
from app.router import is_youtube

VID = "aircAruvnKk"  # 11 chars, valid id alphabet
VID2 = "dQw4w9WgXcQ"


class Snippet:
    """Stand-in for youtube_transcript_api's FetchedTranscriptSnippet (text, start, duration)."""

    def __init__(self, text, start, duration=1.0):
        self.text = text
        self.start = start
        self.duration = duration


# --------------------------------------------------------------------- youtube_video_id (accept)


@pytest.mark.parametrize(
    "url, expected",
    [
        # watch?v=
        (f"https://www.youtube.com/watch?v={VID}", VID),
        (f"https://youtube.com/watch?v={VID}", VID),
        (f"http://www.youtube.com/watch?v={VID}", VID),
        # youtu.be short links
        (f"https://youtu.be/{VID}", VID),
        (f"https://www.youtu.be/{VID}", VID),
        (f"https://youtu.be/{VID}?si=Ab3-_dEfGhIjKlMn", VID),
        (f"https://youtu.be/{VID}?t=30", VID),
        (f"https://youtu.be/{VID}/", VID),  # trailing slash
        # /shorts/ /embed/ /live/ /v/
        (f"https://www.youtube.com/shorts/{VID}", VID),
        (f"https://www.youtube.com/shorts/{VID}/", VID),  # trailing slash
        (f"https://www.youtube.com/shorts/{VID}?feature=share", VID),
        (f"https://www.youtube.com/embed/{VID}", VID),
        (f"https://www.youtube.com/embed/{VID}?start=30&rel=0", VID),
        (f"https://www.youtube.com/live/{VID}", VID),
        (f"https://www.youtube.com/live/{VID}?si=xyz", VID),
        (f"https://www.youtube.com/v/{VID}", VID),
        # alternate hosts
        (f"https://m.youtube.com/watch?v={VID}", VID),
        (f"https://m.youtube.com/shorts/{VID}", VID),
        (f"https://music.youtube.com/watch?v={VID}", VID),
        # host casing is normalised
        (f"https://WWW.YOUTUBE.COM/watch?v={VID}", VID),
        (f"https://YouTu.Be/{VID}", VID),
        (f"HTTPS://Www.YouTube.com/watch?v={VID}", VID),
        # extra query params, in either order
        (f"https://www.youtube.com/watch?v={VID}&t=30s", VID),
        (f"https://www.youtube.com/watch?v={VID}&list=PLabcdefg&index=4", VID),
        (f"https://www.youtube.com/watch?t=30s&v={VID}", VID),  # v= is NOT the first param
        (f"https://www.youtube.com/watch?feature=share&list=PLxyz&v={VID}&t=1m", VID),
        # ids that stress the allowed alphabet (underscore, hyphen, digits, mixed case)
        ("https://www.youtube.com/watch?v=_-Ab3cD4eF5", "_-Ab3cD4eF5"),
        (f"https://youtu.be/{VID2}", VID2),
    ],
)
def test_youtube_video_id_accepts(url, expected):
    assert youtube_video_id(url) == expected


def test_youtube_video_id_takes_first_v_when_repeated():
    # parse_qs keeps every value; the extractor takes the first one.
    assert youtube_video_id(f"https://www.youtube.com/watch?v={VID}&v={VID2}") == VID


@pytest.mark.parametrize(
    "url",
    [
        # trailing slash on /watch is now tolerated (path is rstripped before comparison)
        f"https://www.youtube.com/watch/?v={VID}",
        f"https://youtube.com/watch/?v={VID}&t=10",
        # ANY youtube.com subdomain is accepted, not just www./m./music.
        f"https://gaming.youtube.com/watch?v={VID}",
        f"https://gaming.youtube.com/shorts/{VID}",
        f"https://tv.youtube.com/watch?v={VID}",
        f"https://kids.youtube.com/watch?v={VID}",
        # an explicit port in the netloc no longer breaks host matching
        f"https://youtube.com:443/watch?v={VID}",
        f"https://www.youtube.com:443/shorts/{VID}",
        f"http://m.youtube.com:8080/watch?v={VID}",
        f"https://youtu.be:443/{VID}",
    ],
)
def test_youtube_video_id_accepts_previously_unsupported_shapes(url):
    """These four shapes used to raise ExtractionError; the shared host helper fixed them.

    Each is a URL a user could realistically paste, and each now resolves to the video id
    instead of dying with "Could not find a YouTube video id".
    """
    assert youtube_video_id(url) == VID


# --------------------------------------------------------------------- youtube_video_id (reject)


@pytest.mark.parametrize(
    "url",
    [
        # not YouTube at all
        "https://example.com/watch?v=aircAruvnKk",
        "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
        "https://vimeo.com/123456789",
        "not a url at all",
        # lookalike hosts must not yield an id either
        "https://notyoutube.com/watch?v=aircAruvnKk",
        "https://youtube.com.evil.com/watch?v=aircAruvnKk",
        "https://myyoutu.be/aircAruvnKk",
        "https://youtu.beer/aircAruvnKk",
        # youtube.com only in the userinfo: the real host is evil.com, so no id
        "https://youtube.com@evil.com/watch?v=aircAruvnKk",
        # YouTube host, no video id
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch/",  # /watch with no v= at all
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/watch?list=PLabcdefghij",
        "https://youtu.be/",
        "https://youtu.be",
        # bare channel / playlist / feed URLs
        "https://www.youtube.com/@veritasium",
        "https://www.youtube.com/c/veritasium",
        "https://www.youtube.com/user/1veritasium",
        "https://www.youtube.com/channel/UCHnyfMqiRRG1u-2MsSQLbXA",
        "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab",
        "https://www.youtube.com/results?search_query=neural+networks",
        "https://www.youtube.com/feed/subscriptions",
        # wrong length
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch?v=aircAruvnK",  # 10
        "https://www.youtube.com/watch?v=aircAruvnKkk",  # 12
        "https://youtu.be/aircAruvnK",  # 10
        "https://youtu.be/aircAruvnKkk",  # 12
        "https://www.youtube.com/shorts/aircAruvnKkk",  # 12
        "https://gaming.youtube.com/watch?v=aircAruvnKkk",  # 12, on a subdomain
        "https://youtube.com:443/watch?v=aircAruvnKkk",  # 12, with a port
        # invalid characters (11 chars, but outside [A-Za-z0-9_-])
        "https://www.youtube.com/watch?v=aircAruvnK!",
        "https://www.youtube.com/watch?v=airc.ruvnKk",
        "https://www.youtube.com/watch?v=airc%20uvnKk",  # decodes to a space
        "https://youtu.be/airc+ruvnKk",
        # empty / whitespace input
        "",
        "   ",
        "https://",
    ],
)
def test_youtube_video_id_rejects(url):
    with pytest.raises(ExtractionError):
        youtube_video_id(url)


def test_youtube_video_id_error_message_includes_url():
    url = "https://www.youtube.com/@veritasium"
    with pytest.raises(ExtractionError) as excinfo:
        youtube_video_id(url)
    assert url in str(excinfo.value)


# --------------------------------------------------------------------------------- is_youtube


def test_router_is_youtube_is_the_extractor_helper():
    # One definition, re-exported: they cannot drift apart because they are the same object.
    assert is_youtube is is_youtube_host


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/watch?v=aircAruvnKk",
        "https://www.youtube.com/watch?v=aircAruvnKk",
        "https://m.youtube.com/watch?v=aircAruvnKk",
        "https://music.youtube.com/watch?v=aircAruvnKk",
        "https://gaming.youtube.com/watch?v=aircAruvnKk",
        "https://WWW.YouTube.COM/watch?v=aircAruvnKk",
        "https://youtube.com:443/watch?v=aircAruvnKk",
        "https://youtu.be/aircAruvnKk",
        "https://www.youtu.be/aircAruvnKk",
        "https://YOUTU.BE/aircAruvnKk",
        "http://youtube.com/watch?v=aircAruvnKk",
    ],
)
def test_is_youtube_accepts_real_hosts(url):
    assert is_youtube(url) is True


@pytest.mark.parametrize(
    "url",
    [
        # THE lookalikes called out in the brief
        "https://notyoutube.com/watch?v=aircAruvnKk",
        "https://www.notyoutube.com/watch?v=aircAruvnKk",
        "https://youtube.com.evil.com/watch?v=aircAruvnKk",
        "https://www.youtube.com.evil.com/watch?v=aircAruvnKk",
        # other near-misses
        "https://youtube.co/watch?v=aircAruvnKk",
        "https://youtubecom/watch?v=aircAruvnKk",
        "https://myyoutu.be/aircAruvnKk",
        "https://youtu.beer/aircAruvnKk",
        "https://xyoutu.be/aircAruvnKk",
        "https://fakeyoutube.com/watch?v=aircAruvnKk",
        # youtube only in userinfo / path / query, never the host
        "https://youtube.com@evil.com/watch?v=aircAruvnKk",
        "https://evil.com/youtube.com/watch?v=aircAruvnKk",
        "https://evil.com/?redirect=https://www.youtube.com/watch?v=aircAruvnKk",
        # plain articles
        "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
        "https://example.com",
        "",
        "youtube.com/watch?v=aircAruvnKk",  # no scheme -> urlparse puts it all in .path
    ],
)
def test_is_youtube_rejects_lookalikes(url):
    assert is_youtube(url) is False


def test_userinfo_lookalike_is_judged_on_the_real_host():
    """https://youtube.com@evil.com/ -> the connection goes to evil.com, and _host says so."""
    url = "https://youtube.com@evil.com/watch?v=" + VID
    assert _host(url) == "evil.com"
    assert is_youtube(url) is False
    with pytest.raises(ExtractionError):
        youtube_video_id(url)


def test_userinfo_before_a_real_youtube_host_is_still_youtube():
    # The mirror image of the attack above: the host really is youtube.com, so this is YouTube
    # and the id parses. Only the part after '@' decides.
    url = f"https://evil.com@www.youtube.com/watch?v={VID}"
    assert _host(url) == "youtube.com"
    assert is_youtube(url) is True
    assert youtube_video_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}",
        f"https://YouTube.com/watch/?v={VID}",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://music.youtube.com/watch?v={VID}",
        f"https://gaming.youtube.com/watch?v={VID}",  # arbitrary subdomain
        f"https://gaming.youtube.com/shorts/{VID}",
        f"https://youtube.com:443/watch?v={VID}",  # explicit port
        f"https://www.youtube.com:443/embed/{VID}",
        f"https://youtu.be/{VID}",
        f"https://www.youtu.be/{VID}?t=42",
        f"https://youtu.be:443/{VID}",
        f"https://evil.com@www.youtube.com/live/{VID}",
    ],
)
def test_is_youtube_and_video_id_are_consistent(url):
    """The router and the extractor now share is_youtube_host, so they cannot disagree.

    Replaces the old test_is_youtube_vs_video_id_disagree, which pinned the bug where
    route() handed gaming.youtube.com (and :443 URLs) to extract_youtube(), which then died
    with "Could not find a YouTube video id" instead of extracting anything. If is_youtube()
    says yes for a URL that carries a video id, youtube_video_id() must produce it.
    """
    assert is_youtube(url) is True
    assert youtube_video_id(url) == VID


# --------------------------------------------------------------------------------- _host


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.youtube.com/watch?v=x", "youtube.com"),
        ("https://YouTube.COM/watch", "youtube.com"),  # lowercased
        ("https://youtube.com:443/watch", "youtube.com"),  # port stripped
        ("http://m.youtube.com:8080/watch", "m.youtube.com"),
        ("https://user:pw@example.com/page", "example.com"),  # userinfo stripped
        ("https://youtube.com@evil.com/page", "evil.com"),
        ("https://example.com", "example.com"),
        ("https://www.example.co.uk/a/b", "example.co.uk"),
        ("example.com/watch", ""),  # no scheme -> nothing lands in netloc
        ("", ""),
        ("not a url at all", ""),
    ],
)
def test_host_normalizes(url, expected):
    assert _host(url) == expected


def test_host_strips_only_the_leading_www():
    # Nested "www." is not a thing in practice, but the helper is documented as leading-only.
    assert _host("https://www.www.youtube.com/watch") == "www.youtube.com"


# --------------------------------------------------------------------------------- _normalize


def test_normalize_collapses_three_or_more_newlines_to_two():
    assert _normalize("a\n\n\nb") == "a\n\nb"
    assert _normalize("a\n\n\n\n\n\n\nb") == "a\n\nb"


def test_normalize_preserves_single_and_double_newlines():
    assert _normalize("a\nb") == "a\nb"
    assert _normalize("para one\n\npara two") == "para one\n\npara two"


def test_normalize_strips_spaces_and_tabs_before_newline():
    assert _normalize("a   \nb") == "a\nb"
    assert _normalize("a\t\t\nb") == "a\nb"
    assert _normalize("a \t \nb") == "a\nb"
    # a whitespace-only "blank" line becomes a real blank line, not a third newline
    assert _normalize("a\n   \nb") == "a\n\nb"


def test_normalize_handles_crlf_and_bare_cr():
    assert _normalize("a\r\nb") == "a\nb"
    assert _normalize("a\rb") == "a\nb"
    assert _normalize("a\r\n\r\n\r\n\r\nb") == "a\n\nb"
    assert _normalize("a\r\r\r\rb") == "a\n\nb"
    assert _normalize("a   \r\nb") == "a\nb"  # trailing spaces killed after CRLF folding


def test_normalize_strips_leading_and_trailing_whitespace():
    assert _normalize("   \n\n hello \n\n  ") == "hello"
    assert _normalize("\r\n\r\n\ttext\t\r\n\r\n") == "text"
    assert _normalize("") == ""
    assert _normalize("   \n\t\n  ") == ""
    assert _normalize("\n\n\n") == ""


def test_normalize_is_idempotent():
    messy = "  Title \r\n\r\n\r\n Body line one \t\n \n\n Body line two \r\n  "
    once = _normalize(messy)
    assert _normalize(once) == once
    # Note the surviving leading spaces: _normalize only trims whitespace BEFORE a newline
    # and at the very ends of the string, never indentation at the start of an inner line.
    assert once == "Title\n\n Body line one\n\n Body line two"


def test_normalize_leaves_interior_and_leading_line_whitespace_alone():
    # Only whitespace immediately BEFORE a newline is removed; indentation survives.
    assert _normalize("a \n \n \n b") == "a\n\n b"
    assert _normalize("x\n    indented") == "x\n    indented"
    assert _normalize("double  space") == "double  space"


def test_normalize_strips_non_ascii_spaces_before_newline():
    """Fixed: the regex is [^\\S\\n]+ now, so NBSP and friends no longer survive.

    Articles and captions routinely carry U+00A0; leaving it before a newline left invisible
    trailing whitespace in Document.text.
    """
    assert _normalize("a\u00a0\nb") == "a\nb"  # NBSP
    assert _normalize("a \u00a0\t\nb") == "a\nb"  # NBSP + ASCII whitespace
    assert _normalize("a\u2009\u2003\nb") == "a\nb"  # thin space, em space
    # a NBSP-only line still collapses to one blank line, not a third newline
    assert _normalize("a\n\u00a0\nb") == "a\n\nb"
    # ...but a NBSP that is not before a newline is still content and stays put
    assert _normalize("a\u00a0b") == "a\u00a0b"
    # a trailing NBSP at the very end goes with the usual .strip()
    assert _normalize("a\u00a0") == "a"


# --------------------------------------------------------------------------------- _timestamp


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00"),
        (0.0, "00:00"),
        (0.99, "00:00"),  # truncates, never rounds
        (9, "00:09"),
        (59, "00:59"),
        (59.999, "00:59"),
        (60, "01:00"),
        (65.9, "01:05"),  # 65.9s -> 01:05, NOT 01:06
        (599, "09:59"),
        (600, "10:00"),
        (3599, "59:59"),
    ],
)
def test_timestamp_under_one_hour(seconds, expected):
    assert _timestamp(seconds) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (3600, "01:00:00"),
        (3601, "01:00:01"),
        (3661.5, "01:01:01"),
        (5400, "01:30:00"),
        (7325, "02:02:05"),
        (36000, "10:00:00"),
        (86399, "23:59:59"),
        (360000, "100:00:00"),  # hours are not zero-padded past two digits, just wider
    ],
)
def test_timestamp_past_one_hour_uses_hours(seconds, expected):
    """Fixed: _timestamp() now has an hours component, matching what YouTube displays.

    These strings become Document.anchors and then the `citation` the model must copy
    verbatim (app/pathbuilder.py SYSTEM allows [MM:SS] or [HH:MM:SS]), so a long lecture
    now gets citations that point at a timestamp YouTube actually shows.
    """
    assert _timestamp(seconds) == expected
    hours, minutes, secs = expected.split(":")
    # the minutes field is now minutes-WITHIN-the-hour and never runs past 59
    assert 0 <= int(minutes) <= 59
    assert 0 <= int(secs) <= 59
    assert int(hours) * 3600 + int(minutes) * 60 + int(secs) == int(seconds)


def test_timestamp_hour_boundary_switches_format():
    assert _timestamp(3599) == "59:59"  # MM:SS right up to the boundary
    assert _timestamp(3600) == "01:00:00"  # HH:MM:SS from the boundary on
    assert _timestamp(3599.9) == "59:59"


def test_timestamp_negative_input_clamps_to_zero():
    """Fixed: negative starts are clamped instead of rendering as "-1:59"."""
    assert _timestamp(-1) == "00:00"
    assert _timestamp(-0.5) == "00:00"
    assert _timestamp(-3661) == "00:00"
    assert _timestamp(-1e9) == "00:00"


def test_timestamp_accepts_int_and_float_equivalently():
    assert _timestamp(125) == _timestamp(125.0) == _timestamp(125.75) == "02:05"


# ----------------------------------------------------------------------------- _transcript_text


def test_transcript_text_inserts_markers_and_keeps_order():
    snippets = [
        Snippet("first line", 0.0),
        Snippet("still first chunk", 5.0),
        Snippet("second chunk", 31.0),
        Snippet("third chunk", 61.0),
    ]
    text, markers = _transcript_text(snippets, marker_every=30.0)
    assert text == "[00:00] first line still first chunk [00:31] second chunk [01:01] third chunk"
    assert markers == ["00:00", "00:31", "01:01"]


def test_transcript_text_always_marks_the_first_snippet():
    # Even when the first caption does not start at 0.0 (common: intro music is not captioned).
    text, markers = _transcript_text([Snippet("hello", 12.4)], marker_every=30.0)
    assert markers == ["00:12"]
    assert text.startswith("[00:12] ")
    # and when the first non-empty snippet is preceded by whitespace-only ones
    text, markers = _transcript_text(
        [Snippet("   ", 0.0), Snippet("\n\t", 2.0), Snippet("real words", 4.0)],
        marker_every=30.0,
    )
    assert markers == ["00:04"]
    assert text == "[00:04] real words"


@pytest.mark.parametrize("marker_every", [5.0, 30.0, 120.0, TIMESTAMP_MARKER_SECONDS])
def test_transcript_text_every_returned_marker_appears_in_the_text(marker_every):
    """Directive 8: an anchor the model is never shown is an invitation to invent one.

    pathbuilder.unverifiable_citations() checks citations against Document.anchors, so every
    anchor must be findable in Document.text.
    """
    snippets = [Snippet(f"line {i}", i * 7.5) for i in range(40)]
    text, markers = _transcript_text(snippets, marker_every=marker_every)
    assert markers
    for marker in markers:
        assert f"[{marker}]" in text


def test_transcript_text_does_not_mark_snippets_closer_than_marker_every():
    # Nine snippets one second apart: only the first is inside a fresh marker window.
    snippets = [Snippet(f"w{i}", float(i)) for i in range(9)]
    text, markers = _transcript_text(snippets, marker_every=30.0)
    assert markers == ["00:00"]
    assert text.count("[") == 1
    assert text == "[00:00] w0 w1 w2 w3 w4 w5 w6 w7 w8"


def test_transcript_text_marker_spacing_respects_marker_every():
    snippets = [Snippet(f"w{i}", float(i)) for i in range(0, 100, 4)]
    text, markers = _transcript_text(snippets, marker_every=20.0)
    # 0, 20, 40, 60, 80 fall exactly on the window boundaries and each land a marker
    assert markers == ["00:00", "00:20", "00:40", "01:00", "01:20"]
    assert len(markers) == text.count("[")


def test_transcript_text_uses_hour_markers_for_long_videos():
    # The _timestamp fix flows straight through to the anchors of a 2-hour lecture.
    snippets = [Snippet("late point", 3700.0), Snippet("later still", 7325.0)]
    text, markers = _transcript_text(snippets, marker_every=30.0)
    assert markers == ["01:01:40", "02:02:05"]
    assert text == "[01:01:40] late point [02:02:05] later still"


def test_transcript_text_handles_empty_input():
    assert _transcript_text([], marker_every=30.0) == ("", [])
    assert _transcript_text([Snippet("  ", 0.0), Snippet("", 1.0)], marker_every=30.0) == ("", [])


def test_transcript_text_defaults_to_the_module_marker_interval():
    snippets = [Snippet("a", 0.0), Snippet("b", TIMESTAMP_MARKER_SECONDS - 1), Snippet("c", 600.0)]
    assert _transcript_text(snippets)[1] == _transcript_text(
        snippets, marker_every=TIMESTAMP_MARKER_SECONDS
    )[1]
