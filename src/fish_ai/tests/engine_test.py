# -*- coding: utf-8 -*-

import logging
from unittest.mock import MagicMock, patch

import pytest

from fish_ai.engine import (
    get_claude_cli_args,
    get_claude_prompt,
    get_commandline_history,
    get_messages,
    get_response,
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


MINIMAL_FLAGS = [
    "--setting-sources",
    "",
    "--disable-slash-commands",
    "--strict-mcp-config",
    "--tools",
    "",
]

SUBSCRIPTION_CONFIG = {
    "provider": "anthropic",
    "use_subscription": "true",
    "model": "claude-sonnet-5-5",
    "redact": "False",
    "minimal_mode": "True",
    "effort": "medium",
}


def config_from(values):
    return lambda key: values.get(key)


def cli_args_for(values):
    with patch("fish_ai.engine.get_config", side_effect=config_from(values)):
        return get_claude_cli_args()


def test_claude_cli_args_empty_by_default():
    assert cli_args_for({}) == []


@pytest.mark.parametrize("value", ["True", "true", "TRUE"])
def test_claude_cli_args_minimal_mode(value):
    assert cli_args_for({"minimal_mode": value}) == MINIMAL_FLAGS


def test_claude_cli_args_minimal_mode_false():
    assert cli_args_for({"minimal_mode": "False"}) == []


def test_claude_cli_args_effort():
    assert cli_args_for({"effort": "medium"}) == ["--effort", "medium"]


def test_claude_cli_args_order():
    values = {
        "minimal_mode": "True",
        "effort": "high",
        "cli_args": "--add-dir '/tmp/my dir'",
    }
    assert cli_args_for(values) == MINIMAL_FLAGS + [
        "--effort",
        "high",
        "--add-dir",
        "/tmp/my dir",
    ]


def test_claude_cli_args_invalid_effort_raises():
    with pytest.raises(ValueError, match="invalid effort 'turbo'"):
        cli_args_for({"effort": "turbo"})


def test_claude_cli_args_unparsable_cli_args_raises():
    with pytest.raises(ValueError, match="could not parse cli_args"):
        cli_args_for({"cli_args": 'a "b'})


def test_claude_cli_args_warns_on_effort_flag(caplog):
    with caplog.at_level(logging.WARNING):
        args = cli_args_for({"cli_args": "--effort=high"})
    assert args == ["--effort=high"]
    assert "cli_args sets --effort" in caplog.text


def test_claude_cli_args_warns_when_minimal_mode_is_overridden(caplog):
    with caplog.at_level(logging.WARNING):
        args = cli_args_for(
            {"minimal_mode": "true", "cli_args": "--tools Bash"}
        )
    assert args[-2:] == ["--tools", "Bash"]
    assert "cli_args sets --tools, which minimal_mode turned off" in caplog.text


@pytest.mark.parametrize("value", ["True", "true"])
@patch("subprocess.run")
@patch("fish_ai.engine.get_config")
def test_subscription_argv(mock_get_config, mock_run, value):
    config = dict(SUBSCRIPTION_CONFIG, use_subscription=value)
    mock_get_config.side_effect = config_from(config)
    mock_run.return_value = MagicMock(returncode=0, stdout="echo hi", stderr="")

    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "example question"},
        {"role": "assistant", "content": "example answer"},
        {"role": "user", "content": "say hi"},
    ]
    assert get_response(messages=messages) == "echo hi"

    argv = mock_run.call_args[0][0]
    assert argv == [
        "claude",
        "-p",
        (
            "--system-prompt=Be brief.\n\n"
            "Previous exchanges:\n"
            "User: example question\n"
            "Assistant: example answer"
        ),
        "--model",
        "claude-sonnet-5-5",
        *MINIMAL_FLAGS,
        "--effort",
        "medium",
    ]
    assert mock_run.call_args.kwargs["input"] == "say hi"


@patch("subprocess.run")
@patch("fish_ai.engine.get_config")
def test_subscription_prompt_starting_with_dash_is_not_an_option(
    mock_get_config, mock_run
):
    mock_get_config.side_effect = config_from(SUBSCRIPTION_CONFIG)
    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

    get_response(messages=[{"role": "user", "content": "-n 5 largest files"}])

    argv = mock_run.call_args[0][0]
    assert not any("-n 5" in arg for arg in argv)
    assert mock_run.call_args.kwargs["input"] == "-n 5 largest files"


def test_claude_prompt_single_user_message():
    messages = [
        {"role": "system", "content": "System A"},
        {"role": "system", "content": "System B"},
        {"role": "user", "content": "the request"},
    ]
    assert get_claude_prompt(messages) == (
        "System A\n\nSystem B",
        "the request",
    )


def test_claude_prompt_without_trailing_user_message():
    messages = [{"role": "system", "content": "System A"}]
    assert get_claude_prompt(messages) == ("System A", "")


@patch("subprocess.run")
@patch("fish_ai.engine.get_config")
def test_subscription_invalid_effort_raises_before_cli_runs(
    mock_get_config, mock_run
):
    config = dict(SUBSCRIPTION_CONFIG, effort="turbo")
    mock_get_config.side_effect = config_from(config)

    with pytest.raises(ValueError, match="invalid effort"):
        get_response(messages=[{"role": "user", "content": "hi"}])
    mock_run.assert_not_called()
