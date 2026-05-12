import argparse

from main import resolve_runtime_config


def _args(**overrides):
    base = dict(
        openai=False,
        provider=None,
        api_mode=None,
        model=None,
        api_key=None,
        base_url=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_runtime_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_ENABLE", "true")
    monkeypatch.setenv("LLM_PROVIDER", "opencode")
    monkeypatch.setenv("LLM_API_MODE", "chat")
    monkeypatch.setenv("OPENCODE_MODEL", "minimax-2.5")
    monkeypatch.setenv("OPENCODE_API_KEY", "k1")
    monkeypatch.setenv("OPENCODE_BASE_URL", "https://example.com/v1")

    cfg = resolve_runtime_config(_args())
    assert cfg.use_openai is True
    assert cfg.provider == "opencode"
    assert cfg.api_mode == "chat"
    assert cfg.model == "minimax-2.5"
    assert cfg.api_key == "k1"
    assert cfg.base_url == "https://example.com/v1"


def test_cli_has_higher_priority_than_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-env")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env-base")

    cfg = resolve_runtime_config(
        _args(
            openai=True,
            provider="opencode",
            api_mode="responses",
            model="minimax-cli",
            api_key="cli-key",
            base_url="https://cli-base",
        )
    )
    assert cfg.use_openai is True
    assert cfg.provider == "opencode"
    assert cfg.api_mode == "responses"
    assert cfg.model == "minimax-cli"
    assert cfg.api_key == "cli-key"
    assert cfg.base_url == "https://cli-base"
