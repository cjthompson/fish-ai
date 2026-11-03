# -*- coding: utf-8 -*-

from subprocess import DEVNULL, PIPE
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fish_ai.engine import (
    _KEYWORD_HISTORY_SCRIPT,
    get_commandline_history,
    get_keyword_history,
    get_keywords,
    get_messages,
    get_response,
)


def _proc(entries, returncode=0):
    return SimpleNamespace(
        returncode=returncode,
        stdout="\0".join(entries).encode("utf-8"),
    )


def test_get_messages():
    instructions = [
        {"role": "system", "content": "System message 1"},
        {"role": "user", "content": "User message 1"},
        {"role": "user", "content": "User message 2"},
    ]
    messages = get_messages(instructions)

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].__contains__(
        "You are a shell scripting assistant working inside a fish shell."
    )
    assert messages[0]["content"].__contains__("System message 1")
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "user"


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.get_logger")
def test_get_commandline_history_disabled(mock_get_logger, mock_get_config):
    mock_get_config.return_value = "0"
    assert (
        get_commandline_history("command", 0)
        == "No commandline history available."
    )
    mock_get_logger.assert_called_once()


def test_get_keywords_removes_stopwords_and_short_words():
    assert get_keywords("List the files for all of us") == ["list", "files"]


def test_get_keywords_lowercases_and_dedupes_in_first_seen_order():
    assert get_keywords("Docker pull DOCKER image pull") == [
        "docker",
        "pull",
        "image",
    ]


def test_get_keywords_caps_at_five():
    assert get_keywords("alpha bravo charlie delta echo foxtrot golf") == [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
    ]


def test_get_keywords_strips_leading_and_trailing_dots_and_dashes():
    assert get_keywords("..config -verbose- .hidden. file.txt") == [
        "config",
        "verbose",
        "hidden",
        "file.txt",
    ]


def test_get_keywords_drops_tokens_without_alphanumerics():
    assert get_keywords("--- ... _ __ ___ -.- git") == ["git"]


def test_get_keywords_empty_input():
    assert get_keywords("") == []


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.get_logger")
@patch("fish_ai.engine.run")
def test_get_keyword_history_disabled(
    mock_run, mock_get_logger, mock_get_config
):
    mock_get_config.return_value = "0"
    assert get_keyword_history("list docker images") == []
    mock_run.assert_not_called()
    mock_get_logger.assert_called_once()
    mock_get_logger.return_value.debug.assert_called_once_with(
        "Keyword history disabled."
    )


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.get_logger")
@patch("fish_ai.engine.run")
def test_get_keyword_history_unset_or_negative_makes_no_subprocess_call(
    mock_run, mock_get_logger, mock_get_config
):
    for value in (None, "", "-3"):
        mock_get_config.return_value = value
        assert get_keyword_history("list docker images") == []
    mock_run.assert_not_called()


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.get_logger")
@patch("fish_ai.engine.run")
def test_get_keyword_history_non_numeric_size_returns_empty(
    mock_run, mock_get_logger, mock_get_config
):
    mock_get_config.return_value = "abc"
    assert get_keyword_history("docker images") == []
    mock_run.assert_not_called()


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.run")
def test_get_keyword_history_ranks_by_keyword_match_count(
    mock_run, mock_get_config
):
    mock_get_config.return_value = "5"
    mock_run.side_effect = [
        _proc(["docker ps", "docker images -a", "docker pull alpine"]),
        _proc(["docker images -a", "podman images"]),
    ]
    assert get_keyword_history("docker images") == [
        "docker images -a",
        "docker ps",
        "docker pull alpine",
        "podman images",
    ]
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0].args[0] == [
        "fish",
        "-c",
        _KEYWORD_HISTORY_SCRIPT,
        "--",
        "5",
        "docker",
    ]
    assert mock_run.call_args_list[0].kwargs == {
        "stdout": PIPE,
        "stderr": DEVNULL,
    }


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.run")
def test_get_keyword_history_dedupes_within_a_keyword_and_ignores_empty_chunks(
    mock_run, mock_get_config
):
    mock_get_config.return_value = "5"
    mock_run.side_effect = [
        SimpleNamespace(
            returncode=0,
            stdout=b"docker ps\0docker ps\0\0 docker ps \0docker run alpine\0",
        ),
        _proc(["docker run alpine"]),
    ]
    assert get_keyword_history("docker images") == [
        "docker run alpine",
        "docker ps",
    ]


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.run")
def test_get_keyword_history_caps_at_history_size(mock_run, mock_get_config):
    mock_get_config.return_value = "2"
    mock_run.side_effect = [_proc(["docker a", "docker b", "docker c"])]
    assert get_keyword_history("docker") == ["docker a", "docker b"]
    assert mock_run.call_args.args[0][4] == "2"


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.run")
def test_get_keyword_history_no_matches(mock_run, mock_get_config):
    mock_get_config.return_value = "5"
    mock_run.side_effect = [_proc([]), _proc([])]
    assert get_keyword_history("docker images") == []
    mock_run.reset_mock()
    assert get_keyword_history("the of") == []
    mock_run.assert_not_called()


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.run")
def test_get_keyword_history_skips_keyword_on_nonzero_returncode(
    mock_run, mock_get_config
):
    mock_get_config.return_value = "5"
    mock_run.side_effect = [
        _proc(["docker ps"], returncode=1),
        _proc(["docker images -a"]),
    ]
    assert get_keyword_history("docker images") == ["docker images -a"]
    assert mock_run.call_count == 2


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.get_logger")
@patch("fish_ai.engine.run")
def test_get_keyword_history_returns_empty_on_exception(
    mock_run, mock_get_logger, mock_get_config
):
    mock_get_config.return_value = "5"
    mock_run.side_effect = OSError("fish not found")
    assert get_keyword_history("docker images") == []
    mock_debug = mock_get_logger.return_value.debug
    assert mock_debug.call_count == 1
    debug_message = mock_debug.call_args.args[0]
    assert "fish not found" in debug_message


@patch("fish_ai.engine.get_config")
def test_bedrock_converse_happy_path(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "bedrock_api": "converse",
            "aws_region": "us-west-2",
            "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_bedrock_client = MagicMock()
    mock_bedrock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "echo hello"}],
            }
        }
    }

    with patch(
        "fish_ai.engine.get_bedrock_client", return_value=mock_bedrock_client
    ) as mock_get_client:
        response = get_response(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        mock_get_client.assert_called_once()
        mock_bedrock_client.converse.assert_called_once()
        call_kwargs = mock_bedrock_client.converse.call_args[1]
        assert (
            call_kwargs["modelId"]
            == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        assert call_kwargs["system"] == [{"text": "You are helpful."}]
        assert call_kwargs["messages"] == [
            {"role": "user", "content": [{"text": "hello"}]}
        ]
        assert response == "echo hello"


@patch("fish_ai.engine.get_config")
def test_bedrock_converse_default_model(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "bedrock_api": "converse",
            "aws_region": "us-west-2",
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_bedrock_client = MagicMock()
    mock_bedrock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "echo hello"}],
            }
        }
    }

    with patch(
        "fish_ai.engine.get_bedrock_client", return_value=mock_bedrock_client
    ):
        get_response(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        call_kwargs = mock_bedrock_client.converse.call_args[1]
        assert (
            call_kwargs["modelId"]
            == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )


@patch("fish_ai.engine.get_config")
def test_bedrock_mantle_is_default_api(mock_get_config):
    """When bedrock_api is not set, mantle should be used."""

    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "aws_region": "us-west-2",
            "api_key": "test-api-key",
            "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_completions = MagicMock()
    mock_completions.choices = [
        MagicMock(message=MagicMock(content="echo hello"))
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completions

    with patch(
        "fish_ai.engine.get_openai_client", return_value=mock_client
    ) as mock_get_client:
        get_response(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        mock_get_client.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()


@patch("fish_ai.engine.get_config")
def test_bedrock_client_uses_profile(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "aws_region": "us-west-2",
            "aws_profile": "my-profile",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_session = MagicMock()

    with patch(
        "boto3.Session", return_value=mock_session
    ) as mock_boto3_session:
        from fish_ai.engine import get_bedrock_client

        get_bedrock_client()
        mock_boto3_session.assert_called_once_with(
            profile_name="my-profile",
            region_name="us-west-2",
        )
        mock_session.client.assert_called_once_with("bedrock-runtime")


@patch("fish_ai.engine.get_config")
def test_bedrock_client_default_credentials(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "aws_region": "us-east-1",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_session = MagicMock()

    with patch(
        "boto3.Session", return_value=mock_session
    ) as mock_boto3_session:
        from fish_ai.engine import get_bedrock_client

        get_bedrock_client()
        mock_boto3_session.assert_called_once_with(
            region_name="us-east-1",
        )
        mock_session.client.assert_called_once_with("bedrock-runtime")


@patch("fish_ai.engine.get_config")
def test_bedrock_default_region(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_session = MagicMock()

    with patch(
        "boto3.Session", return_value=mock_session
    ) as mock_boto3_session:
        from fish_ai.engine import get_bedrock_client

        get_bedrock_client()
        mock_boto3_session.assert_called_once_with(
            region_name="us-east-1",
        )


@patch("fish_ai.engine.get_config")
def test_bedrock_messages_conversion(mock_get_config):
    """Test that OpenAI-format messages are correctly converted to Bedrock
    Converse format with system messages extracted separately."""
    mock_get_config.return_value = None

    from fish_ai.engine import get_messages_for_bedrock

    messages = [
        {"role": "system", "content": "You are a shell assistant."},
        {"role": "user", "content": "How do I list files?"},
        {"role": "assistant", "content": "Use ls command."},
        {"role": "user", "content": "What about hidden files?"},
    ]

    system_prompts, converse_messages = get_messages_for_bedrock(messages)

    assert system_prompts == [{"text": "You are a shell assistant."}]
    assert converse_messages == [
        {"role": "user", "content": [{"text": "How do I list files?"}]},
        {"role": "assistant", "content": [{"text": "Use ls command."}]},
        {"role": "user", "content": [{"text": "What about hidden files?"}]},
    ]


@patch("fish_ai.engine.get_config")
def test_bedrock_converse_thinking_tokens_removed(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "bedrock_api": "converse",
            "aws_region": "us-west-2",
            "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_bedrock_client = MagicMock()
    mock_bedrock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "<think>reasoning</think>echo hello"}],
            }
        }
    }

    with patch(
        "fish_ai.engine.get_bedrock_client", return_value=mock_bedrock_client
    ):
        response = get_response(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        assert response == "echo hello"


@patch("fish_ai.engine.get_config")
def test_bedrock_mantle_happy_path(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "bedrock_api": "mantle",
            "aws_region": "us-west-2",
            "api_key": "test-api-key",
            "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_completions = MagicMock()
    mock_completions.choices = [
        MagicMock(message=MagicMock(content="echo hello"))
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completions

    with patch(
        "fish_ai.engine.get_openai_client", return_value=mock_client
    ) as mock_get_client:
        response = get_response(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        mock_get_client.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert (
            call_kwargs["model"] == "anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        assert response == "echo hello"


@patch("fish_ai.engine.get_config")
def test_bedrock_mantle_constructs_url(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "aws_region": "eu-west-1",
            "api_key": "test-key",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    with patch("openai.OpenAI") as mock_openai_cls:
        from fish_ai.engine import get_openai_client

        get_openai_client()
        mock_openai_cls.assert_called_once_with(
            base_url="https://bedrock-mantle.eu-west-1.api.aws/v1",
            api_key="test-key",
            default_headers=None,
        )


@patch("fish_ai.engine.get_config")
def test_bedrock_mantle_auto_token_when_no_api_key(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "aws_region": "us-west-2",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    import sys

    mock_token_module = MagicMock()
    mock_token_module.provide_token.return_value = "auto-generated-token"
    with patch.dict(
        sys.modules, {"aws_bedrock_token_generator": mock_token_module}
    ):
        with patch("openai.OpenAI") as mock_openai_cls:
            from fish_ai.engine import get_openai_client

            get_openai_client()
            mock_token_module.provide_token.assert_called_once_with(
                region="us-west-2"
            )
            mock_openai_cls.assert_called_once_with(
                base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
                api_key="auto-generated-token",
                default_headers=None,
            )


@patch("fish_ai.engine.get_config")
def test_bedrock_mantle_uses_aws_profile_for_token(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "aws_region": "us-west-2",
            "aws_profile": "my-profile",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    import sys

    mock_credentials = MagicMock()
    mock_session = MagicMock()
    mock_session.get_credentials.return_value = mock_credentials

    mock_token_generator = MagicMock()
    mock_token_generator.get_token.return_value = "profile-token"

    mock_token_module = MagicMock()
    mock_token_module.BedrockTokenGenerator.return_value = mock_token_generator

    mock_botocore_module = MagicMock()
    mock_botocore_module.Session.return_value = mock_session

    with patch.dict(
        sys.modules,
        {
            "aws_bedrock_token_generator": mock_token_module,
            "botocore.session": mock_botocore_module,
            "botocore": MagicMock(),
        },
    ):
        with patch("openai.OpenAI") as mock_openai_cls:
            from fish_ai.engine import get_openai_client

            get_openai_client()
            mock_botocore_module.Session.assert_called_once_with(
                profile="my-profile"
            )
            mock_session.get_credentials.assert_called_once()
            mock_token_generator.get_token.assert_called_once_with(
                credentials=mock_credentials,
                region="us-west-2",
            )
            mock_openai_cls.assert_called_once_with(
                base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
                api_key="profile-token",
                default_headers=None,
            )


@patch("fish_ai.engine.get_config")
def test_bedrock_invalid_api_raises(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "bedrock",
            "bedrock_api": "invalid",
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    import pytest

    with pytest.raises(Exception, match='Unknown bedrock_api "invalid"'):
        get_response(
            [
                {"role": "user", "content": "hello"},
            ]
        )


def create_content_block(block_type, **attributes):
    """
    Create a content block as returned by the Anthropic API. A thinking
    block has no text attribute, just like the real thing.
    """
    block = MagicMock(spec=["type", *attributes.keys()])
    block.type = block_type
    for name, value in attributes.items():
        setattr(block, name, value)
    return block


@patch("fish_ai.engine.get_config")
def test_anthropic_happy_path(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "anthropic",
            "api_key": "test-api-key",
            "model": "claude-haiku-5-5",
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_completions = MagicMock()
    mock_completions.content = [create_content_block("text", text="echo hello")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_completions

    with patch("anthropic.Anthropic", return_value=mock_client):
        response = get_response(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-5-5"
        assert call_kwargs["system"] == "You are helpful."
        assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert "extra_body" not in call_kwargs
        assert response == "echo hello"


@patch("fish_ai.engine.get_config")
def test_anthropic_ignores_thinking_blocks(mock_get_config):
    """Models which are thinking return a thinking block first."""

    def config_side_effect(key):
        config = {
            "provider": "anthropic",
            "api_key": "test-api-key",
            "model": "claude-haiku-5-5",
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_completions = MagicMock()
    mock_completions.content = [
        create_content_block("thinking", thinking="reasoning"),
        create_content_block("text", text="echo hello"),
    ]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_completions

    with patch("anthropic.Anthropic", return_value=mock_client):
        response = get_response(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        assert response == "echo hello"


@patch("fish_ai.engine.get_config")
def test_anthropic_extra_body(mock_get_config):
    def config_side_effect(key):
        config = {
            "provider": "anthropic",
            "api_key": "test-api-key",
            "model": "claude-haiku-5-5",
            "extra_body": '{"thinking": {"type": "disabled"}}',
            "redact": "False",
        }
        return config.get(key)

    mock_get_config.side_effect = config_side_effect

    mock_completions = MagicMock()
    mock_completions.content = [create_content_block("text", text="echo hello")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_completions

    with patch("anthropic.Anthropic", return_value=mock_client):
        get_response(
            [
                {"role": "user", "content": "hello"},
            ]
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.get_openai_client")
def test_atlascloud_default_model(mock_get_openai_client, mock_get_config):
    def config_side_effect(key):
        return {"provider": "atlascloud", "redact": "False"}.get(key)

    mock_get_config.side_effect = config_side_effect
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="echo hello"))
    ]
    mock_get_openai_client.return_value = mock_client

    response = get_response([{"role": "user", "content": "say hello"}])

    assert response == "echo hello"
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "deepseek-ai/deepseek-v4-flash"
    assert kwargs["stream"] is False


@patch("fish_ai.engine.get_config")
@patch("fish_ai.engine.get_openai_client")
def test_cheaperinference_default_model(mock_get_openai_client, mock_get_config):
    def config_side_effect(key):
        return {"provider": "cheaperinference", "redact": "False"}.get(key)

    mock_get_config.side_effect = config_side_effect
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="echo hello"))
    ]
    mock_get_openai_client.return_value = mock_client

    response = get_response([{"role": "user", "content": "say hello"}])

    assert response == "echo hello"
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.4-mini"
    assert kwargs["stream"] is False
