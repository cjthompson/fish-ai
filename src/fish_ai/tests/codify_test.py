# -*- coding: utf-8 -*-

from unittest.mock import patch

from fish_ai.codify import codify, get_instructions


@patch("fish_ai.engine.get_keyword_history", lambda text: [])
@patch("fish_ai.engine.get_args", lambda: ["# hello"])
@patch("fish_ai.engine.get_response", lambda messages: "echo hello")
def test_successful_codify(capsys):
    codify()
    assert capsys.readouterr().out == "echo hello"
    assert capsys.readouterr().err == ""


@patch("fish_ai.engine.get_keyword_history", lambda text: [])
@patch("fish_ai.engine.get_args", lambda: ["# hello"])
@patch(
    "fish_ai.engine.get_response", side_effect=Exception("crystal ball failed")
)
def test_unsuccessful_codify(_, caplog):
    codify()
    assert "crystal ball failed" in caplog.text


@patch(
    "fish_ai.engine.get_keyword_history",
    return_value=["docker ps -a", "docker run --rm alpine"],
)
def test_get_instructions_includes_keyword_history(mock_history):
    instructions = get_instructions("list docker containers")
    mock_history.assert_called_once_with("list docker containers")
    content = instructions[0]["content"]
    header = (
        "You may use the following commandline history to match the "
        "user's preferred tools and flags:"
    )
    assert header in content
    assert "flags:\n\ndocker ps -a\ndocker run --rm alpine" in content


@patch("fish_ai.engine.get_keyword_history", return_value=[])
def test_get_instructions_omits_history_section_when_empty(mock_history):
    instructions = get_instructions("list docker containers")
    mock_history.assert_called_once_with("list docker containers")
    content = instructions[0]["content"]
    assert "commandline history" not in content
    assert content.startswith("Respond with a fish shell command")
    assert content.endswith("as parameters.\n")


@patch("fish_ai.engine.get_args", lambda: ["# list docker containers"])
@patch("fish_ai.engine.get_keyword_history", return_value=["docker ps -a"])
@patch("fish_ai.engine.get_response")
def test_codify_passes_keyword_history_to_response(
    mock_get_response, mock_history, capsys
):
    mock_get_response.return_value = "docker ps"
    codify()
    mock_history.assert_called_once_with("list docker containers")
    messages = mock_get_response.call_args.kwargs["messages"]
    assert "docker ps -a" in messages[0]["content"]
    assert capsys.readouterr().out == "docker ps"
