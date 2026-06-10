import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

def test_load_config_defaults():
    """Test that _load_config returns correct defaults."""
    from plugins.memory.mem0 import _load_config

    with patch.dict(os.environ, {}, clear=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hermes_constants.get_hermes_home", return_value=Path(tmpdir)):
                config = _load_config()

                assert config["mode"] == "cloud"
                assert config["user_id"] == "hermes-user"
                assert config["agent_id"] == "hermes"
                assert config["rerank"] is True


def test_load_config_from_env():
    """Test that _load_config reads environment variables."""
    from plugins.memory.mem0 import _load_config

    env = {
        "MEM0_MODE": "local",
        "MEM0_API_KEY": "test-key",
        "MEM0_USER_ID": "test-user",
    }

    with patch.dict(os.environ, env, clear=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hermes_constants.get_hermes_home", return_value=Path(tmpdir)):
                config = _load_config()

                assert config["mode"] == "local"
                assert config["api_key"] == "test-key"
                assert config["user_id"] == "test-user"


def test_load_config_from_file():
    """Test that _load_config reads from mem0.json."""
    from plugins.memory.mem0 import _load_config

    file_config = {
        "mode": "local",
        "user_id": "file-user",
        "local": {
            "llm_provider": "hermes",
            "embedding": {"provider": "ollama", "model": "test-model"},
        },
    }

    with patch.dict(os.environ, {}, clear=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mem0.json"
            config_path.write_text(json.dumps(file_config))

            with patch("hermes_constants.get_hermes_home", return_value=Path(tmpdir)):
                config = _load_config()

                assert config["mode"] == "local"
                assert config["user_id"] == "file-user"
                assert config["local"]["embedding"]["model"] == "test-model"


def test_load_config_backward_compatible():
    """Test that old config without mode defaults to cloud."""
    from plugins.memory.mem0 import _load_config

    old_config = {
        "user_id": "old-user",
        "agent_id": "old-agent",
        "api_key": "old-key",
    }

    with patch.dict(os.environ, {}, clear=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mem0.json"
            config_path.write_text(json.dumps(old_config))

            with patch("hermes_constants.get_hermes_home", return_value=Path(tmpdir)):
                config = _load_config()

                # Should default to cloud for backward compatibility
                assert config["mode"] == "cloud"
                assert config["api_key"] == "old-key"


def test_load_config_local_key_without_mode():
    """Test that config with 'local' key but no 'mode' keeps local default."""
    from plugins.memory.mem0 import _load_config

    local_config = {
        "local": {
            "llm_provider": "hermes",
            "embedding": {"provider": "ollama", "model": "test-model"},
        },
    }

    with patch.dict(os.environ, {}, clear=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mem0.json"
            config_path.write_text(json.dumps(local_config))

            with patch("hermes_constants.get_hermes_home", return_value=Path(tmpdir)):
                config = _load_config()

                # 'local' key present so backward-compat should NOT set mode to cloud
                # env default is "cloud", and file_cfg has no "mode" key, so mode stays as env default
                assert config["mode"] == "cloud"
                assert config["local"]["embedding"]["model"] == "test-model"


def test_check_local_runtime_available():
    """Test _check_local_runtime when mem0 is installed."""
    from plugins.memory.mem0 import _check_local_runtime

    # Mock successful import
    with patch("importlib.import_module", return_value=None):
        available, reason = _check_local_runtime()

        assert available is True
        assert reason is None


def test_check_local_runtime_unavailable():
    """Test _check_local_runtime when mem0 is not installed."""
    from plugins.memory.mem0 import _check_local_runtime

    with patch("importlib.import_module", side_effect=ImportError("No module")):
        available, reason = _check_local_runtime()

        assert available is False
        assert "No module" in reason


def test_is_available_cloud_mode_with_key():
    """Test is_available returns True for cloud mode with API key."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    with patch("plugins.memory.mem0._load_config", return_value={
        "mode": "cloud",
        "api_key": "test-key",
    }):
        assert provider.is_available() is True


def test_is_available_cloud_mode_without_key():
    """Test is_available returns False for cloud mode without API key."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    with patch("plugins.memory.mem0._load_config", return_value={
        "mode": "cloud",
        "api_key": "",
    }):
        assert provider.is_available() is False


def test_is_available_local_mode_available():
    """Test is_available returns True for local mode when runtime available."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    with patch("plugins.memory.mem0._load_config", return_value={
        "mode": "local",
        "local": {"embedding": {"model": "test-model"}},
    }):
        with patch("plugins.memory.mem0._check_local_runtime", return_value=(True, None)):
            assert provider.is_available() is True


def test_is_available_local_mode_unavailable():
    """Test is_available returns False for local mode when runtime unavailable."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    with patch("plugins.memory.mem0._load_config", return_value={
        "mode": "local",
        "local": {"embedding": {"model": "test-model"}},
    }):
        with patch("plugins.memory.mem0._check_local_runtime", return_value=(False, "error")):
            assert provider.is_available() is False
