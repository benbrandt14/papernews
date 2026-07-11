"""Tests for the `papernews` CLI entry point (papernews/__main__.py)."""

import pytest

from papernews.__main__ import main


def test_run_missing_config_returns_error(tmp_path, capsys):
    rc = main(["run", "--config", str(tmp_path / "nope.toml")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_run_invokes_flow_with_loaded_config(tmp_path, mocker, capsys):
    cfg = tmp_path / "sources.toml"
    cfg.write_text(
        '[[source]]\nname = "A"\nkind = "rss"\nurl = "http://x"\ncategory = "Sci"\n'
    )
    fake_pdf = tmp_path / "out.pdf"
    run = mocker.patch("papernews.core.main.run_papernews", return_value=fake_pdf)

    rc = main(["run", "--config", str(cfg)])

    assert rc == 0
    config = run.call_args.kwargs["config"]
    assert config.sources[0].name == "A"
    assert str(fake_pdf) in capsys.readouterr().out


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        main([])


def test_providers_lists_presets(capsys, monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "deepseek")
    rc = main(["providers"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "deepseek" in out and "openrouter" in out and "local" in out
    assert "*" in out  # the active provider is marked


def test_check_llm_ok(mocker, capsys):
    backend = mocker.MagicMock()
    backend.check.return_value = (True, "model @ url replied 'ok'")
    mocker.patch("papernews.core.backends.get_backend", return_value=backend)
    rc = main(["check-llm"])
    assert rc == 0
    assert "OK:" in capsys.readouterr().out


def test_check_llm_failure_returns_nonzero(mocker, capsys):
    backend = mocker.MagicMock()
    backend.check.return_value = (False, "ConnectionError: refused")
    mocker.patch("papernews.core.backends.get_backend", return_value=backend)
    rc = main(["check-llm"])
    assert rc == 1
    assert "FAILED:" in capsys.readouterr().out


def test_check_llm_misconfig_returns_nonzero(mocker, capsys):
    mocker.patch(
        "papernews.core.backends.get_backend",
        side_effect=ValueError("needs an API key"),
    )
    rc = main(["check-llm"])
    assert rc == 1
    assert "Misconfigured" in capsys.readouterr().err
