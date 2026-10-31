# -*- coding: utf-8 -*-

from unittest.mock import patch

from fish_ai.autocomplete import autocomplete, get_instructions, get_pipe


@patch("fish_ai.engine.get_args", lambda: ["echo hello", 10])
@patch("fish_ai.engine.get_config", lambda key: {"completions": "1"}.get(key))
@patch(
    "fish_ai.autocomplete.iterfzf",
    lambda iterable, **kwargs: "echo hello world",
)
def test_successful_autocomplete(capsys):
    autocomplete()
    assert capsys.readouterr().out == "echo hello world"
    assert capsys.readouterr().err == ""


@patch("fish_ai.engine.get_args", lambda: ["echo hello", 4])
@patch(
    "fish_ai.autocomplete.iterfzf",
    side_effect=Exception("crystal ball failed"),
)
def test_unsuccessful_autocomplete(_, caplog):
    autocomplete()
    assert "crystal ball failed" in caplog.text


def test_get_pipe():
    assert get_pipe("echo hello") == ""
    assert get_pipe("echo hello | grep world") == "echo hello"
    assert (
        get_pipe("echo hello | grep world | wc -l") == "echo hello | grep world"
    )
    assert (
        get_pipe("echo hello | grep world | wc -l |")
        == "echo hello | grep world | wc -l"
    )
    assert get_pipe("(echo hello | (echo world | wc -l") == "echo world"
    assert get_pipe('echo "|"') == ""
    assert get_pipe('echo "hello | world" | wc -l') == 'echo "hello | world"'
    assert get_pipe("echo '|'") == ""
    assert get_pipe("echo 'hello | world' | wc -l") == "echo 'hello | world'"
    assert get_pipe('echo "hello ( world" | wc -l') == 'echo "hello ( world"'
    assert get_pipe("💩") == ""
    assert get_pipe('az vm list | jq ".[] =|') == "az vm list"
    assert get_pipe('echo "\\"foo" | a') == 'echo "\\"foo"'
    assert (
        get_pipe('echo \'{"foo":"bar}\' | jq .foo') == 'echo \'{"foo":"bar}\''
    )

    assert (
        get_pipe("""set sponsors (gh api graphql -f query='
        query {
            blah blah
        }' | jq -r .data)""")
        == """gh api graphql -f query='
        query {
            blah blah
        }'"""
    )


@patch("fish_ai.engine.get_config", lambda key: None)
@patch(
    "fish_ai.engine.get_file_info",
    lambda commandline: ("Cargo.toml", "[package]"),
)
def test_autocomplete_file_content_is_separated_from_request():
    content = get_instructions("cat Cargo.toml", 14, 5, "")[-1]["content"]
    assert content.startswith(
        "The content of the file Cargo.toml is:\n\n[package]\n\n---\n\n"
    )


@patch("fish_ai.engine.get_args", lambda: ["# hello"])
@patch(
    "fish_ai.engine.get_response",
    side_effect=ValueError("invalid effort 'turbo' in fish-ai.ini."),
)
def test_autocomplete_codify_config_error_is_printed_in_band(_, capsys):
    autocomplete()
    out = capsys.readouterr().out
    assert out.startswith("# An error occurred when running fish-ai.")
    assert "invalid effort 'turbo'" in out


@patch("fish_ai.engine.get_args", lambda: ["echo hello", 10])
@patch(
    "fish_ai.engine.get_config",
    lambda key: {"use_subscription": "true", "effort": "turbo"}.get(key),
)
@patch("fish_ai.autocomplete.iterfzf")
def test_autocomplete_bad_claude_config_is_reported_before_picker(
    mock_iterfzf, capsys
):
    autocomplete()
    out = capsys.readouterr().out
    assert out.startswith("# An error occurred when running fish-ai.")
    assert "invalid effort 'turbo'" in out
    mock_iterfzf.assert_not_called()
