# -*- coding: utf-8 -*-

from unittest.mock import patch

from fish_ai.codify import codify, get_instructions


@patch("fish_ai.engine.get_args", lambda: ["# hello"])
@patch("fish_ai.engine.get_response", lambda messages: "echo hello")
def test_successful_codify(capsys):
    codify()
    assert capsys.readouterr().out == "echo hello"
    assert capsys.readouterr().err == ""


@patch("fish_ai.engine.get_args", lambda: ["# hello"])
@patch(
    "fish_ai.engine.get_response", side_effect=Exception("crystal ball failed")
)
def test_unsuccessful_codify(_, caplog):
    codify()
    assert "crystal ball failed" in caplog.text


@patch("fish_ai.engine.get_args", lambda: ["# hello"])
@patch("fish_ai.engine.get_response", lambda messages: None)
def test_codify_empty_response_is_reported(capsys):
    codify()
    out = capsys.readouterr().out
    assert out.startswith("# An error occurred when running fish-ai.")
    assert "empty response" in out


@patch("fish_ai.engine.get_args", lambda: ["# hello"])
@patch(
    "fish_ai.engine.get_response",
    side_effect=ValueError("invalid effort 'turbo' in fish-ai.ini."),
)
def test_codify_config_error_is_printed_in_band(_, capsys):
    codify()
    out = capsys.readouterr().out
    assert out.startswith("# An error occurred when running fish-ai.")
    assert "invalid effort 'turbo'" in out


@patch(
    "fish_ai.engine.get_file_info",
    lambda commandline: ("Cargo.toml", "[package]"),
)
def test_codify_file_content_is_separated_from_request():
    content = get_instructions("check outdated packages")[-1]["content"]
    assert content == (
        "check outdated packages\n\n---\n\n"
        "The content of the file Cargo.toml is:\n\n"
        "[package]"
    )


@patch("fish_ai.engine.get_file_info", lambda commandline: (None, None))
def test_codify_without_file_leaves_request_unchanged():
    content = get_instructions("check outdated packages")[-1]["content"]
    assert content == "check outdated packages"
