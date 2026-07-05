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
