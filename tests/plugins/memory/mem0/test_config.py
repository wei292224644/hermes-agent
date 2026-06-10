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


def test_get_hermes_llm_config():
    """Test _get_hermes_llm_config reads from hermes config."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    with patch("plugins.memory.mem0.cfg_get") as mock_cfg_get:
        mock_cfg_get.side_effect = lambda key, default=None: {
            "provider": "openai",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4",
        }.get(key, default)

        config = provider._get_hermes_llm_config()

        assert config["provider"] == "openai"
        assert config["config"]["api_key"] == "test-key"
        assert config["config"]["openai_base_url"] == "https://api.openai.com/v1"
        assert config["config"]["model"] == "gpt-4"


def test_map_hermes_provider_to_mem0():
    """Test _map_hermes_provider_to_mem0 maps correctly."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    assert provider._map_hermes_provider_to_mem0("openai") == "openai"
    assert provider._map_hermes_provider_to_mem0("anthropic") == "anthropic"
    assert provider._map_hermes_provider_to_mem0("ollama") == "ollama"
    assert provider._map_hermes_provider_to_mem0("openrouter") == "openai"
    assert provider._map_hermes_provider_to_mem0("deepseek") == "openai"
    assert provider._map_hermes_provider_to_mem0("unknown") == "openai"


def test_get_llm_config_hermes():
    """Test _get_llm_config with hermes provider."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    with patch.object(provider, "_get_hermes_llm_config", return_value={"provider": "openai", "config": {}}):
        config = provider._get_llm_config({"llm_provider": "hermes"})

        assert config["provider"] == "openai"


def test_get_llm_config_custom():
    """Test _get_llm_config with custom provider."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    local_cfg = {
        "llm": {
            "provider": "ollama",
            "config": {"model": "llama3"},
        }
    }

    config = provider._get_llm_config(local_cfg)

    assert config["provider"] == "ollama"
    assert config["config"]["model"] == "llama3"


# ---------------------------------------------------------------------------
# Task 6: Embedding configuration builder
# ---------------------------------------------------------------------------

def test_get_embedding_config_default():
    """Test _get_embedding_config with defaults."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    config = provider._get_embedding_config({})

    assert config["provider"] == "ollama"
    assert config["config"]["model"] == "qwen3-embedding:4b"
    assert config["config"]["ollama_base_url"] == "http://localhost:11434"


def test_get_embedding_config_custom():
    """Test _get_embedding_config with custom values."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    local_cfg = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test-key",
        }
    }

    config = provider._get_embedding_config(local_cfg)

    assert config["provider"] == "openai"
    assert config["config"]["model"] == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Task 7: Vector store configuration builder
# ---------------------------------------------------------------------------

def test_get_vector_store_config_default():
    """Test _get_vector_store_config with defaults."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    config = provider._get_vector_store_config({})

    assert config["provider"] == "qdrant"
    assert config["config"]["path"] == "~/.hermes/qdrant"
    assert config["config"]["collection_name"] == "mem0"


def test_get_vector_store_config_custom():
    """Test _get_vector_store_config with custom values."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    local_cfg = {
        "vector_store": {
            "provider": "chroma",
            "path": "/custom/path",
            "collection_name": "custom_collection",
        }
    }

    config = provider._get_vector_store_config(local_cfg)

    assert config["provider"] == "chroma"
    assert config["config"]["path"] == "/custom/path"
    assert config["config"]["collection_name"] == "custom_collection"


# ---------------------------------------------------------------------------
# Task 8: Local config builder and _get_client() update
# ---------------------------------------------------------------------------

def test_build_local_config():
    """Test _build_local_config combines all configs."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()
    provider._config = {
        "local": {
            "llm_provider": "hermes",
            "embedding": {"provider": "ollama", "model": "test-model"},
            "vector_store": {"provider": "qdrant", "path": "/test/path"},
        }
    }

    with patch.object(provider, "_get_hermes_llm_config", return_value={"provider": "openai", "config": {}}):
        config = provider._build_local_config()

        assert "llm" in config
        assert "embedder" in config
        assert "vector_store" in config
        assert config["llm"]["provider"] == "openai"
        assert config["embedder"]["provider"] == "ollama"
        assert config["vector_store"]["provider"] == "qdrant"


def test_get_client_local_mode():
    """Test _get_client creates local client."""
    import sys
    from plugins.memory.mem0 import Mem0MemoryProvider
    from unittest.mock import MagicMock

    provider = Mem0MemoryProvider()
    provider._mode = "local"
    provider._config = {"local": {}}

    mock_memory = MagicMock()
    mock_memory.from_config.return_value = MagicMock()

    fake_mem0 = MagicMock()
    fake_mem0.Memory = mock_memory

    with patch.dict(sys.modules, {"mem0": fake_mem0}):
        with patch.object(provider, "_build_local_config", return_value={"llm": {}, "embedder": {}, "vector_store": {}}):
            client = provider._get_client()

            assert client is not None
            mock_memory.from_config.assert_called_once()


def test_get_client_cloud_mode():
    """Test _get_client creates cloud client."""
    import sys
    from plugins.memory.mem0 import Mem0MemoryProvider
    from unittest.mock import MagicMock

    provider = Mem0MemoryProvider()
    provider._mode = "cloud"
    provider._api_key = "test-key"

    mock_client = MagicMock()

    fake_mem0 = MagicMock()
    fake_mem0.MemoryClient = mock_client

    with patch.dict(sys.modules, {"mem0": fake_mem0}):
        client = provider._get_client()

        assert client is not None


# ---------------------------------------------------------------------------
# Task 9: initialize() for mode support
# ---------------------------------------------------------------------------

def test_initialize_local_mode():
    """Test initialize sets up local mode correctly."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    with patch("plugins.memory.mem0._load_config", return_value={
        "mode": "local",
        "user_id": "test-user",
        "agent_id": "test-agent",
        "local": {},
    }):
        provider.initialize("test-session")

        assert provider._mode == "local"
        assert provider._user_id == "test-user"
        assert provider._agent_id == "test-agent"


def test_initialize_cloud_mode():
    """Test initialize sets up cloud mode correctly."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    with patch("plugins.memory.mem0._load_config", return_value={
        "mode": "cloud",
        "api_key": "test-key",
        "user_id": "test-user",
        "agent_id": "test-agent",
    }):
        provider.initialize("test-session")

        assert provider._mode == "cloud"
        assert provider._api_key == "test-key"


# ---------------------------------------------------------------------------
# Task 10: Configuration schema
# ---------------------------------------------------------------------------

def test_get_config_schema():
    """Test get_config_schema returns correct schema."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    schema = provider.get_config_schema()

    # Check that schema contains required fields
    keys = [item["key"] for item in schema]

    assert "mode" in keys
    assert "api_key" in keys
    assert "llm_provider" in keys
    assert "embedding_provider" in keys
    assert "embedding_model" in keys
    assert "vector_store_provider" in keys
    assert "vector_store_path" in keys
    assert "user_id" in keys
    assert "agent_id" in keys
    assert "rerank" in keys

    # Check mode field
    mode_field = next(item for item in schema if item["key"] == "mode")
    assert mode_field["default"] == "cloud"
    assert "cloud" in mode_field["choices"]
    assert "local" in mode_field["choices"]


# ---------------------------------------------------------------------------
# Task 11: save_config() method
# ---------------------------------------------------------------------------

def test_save_config():
    """Test save_config writes to mem0.json."""
    from plugins.memory.mem0 import Mem0MemoryProvider
    from pathlib import Path

    provider = Mem0MemoryProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        values = {"mode": "local", "user_id": "test-user"}

        with patch("utils.atomic_json_write") as mock_write:
            provider.save_config(values, tmpdir)

            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            assert args[0] == Path(tmpdir) / "mem0.json"
            assert args[1]["mode"] == "local"


# ---------------------------------------------------------------------------
# Task 13: Integration test
# ---------------------------------------------------------------------------

def test_integration_local_mode_flow():
    """Test complete local mode flow."""
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()

    # Mock configuration
    config = {
        "mode": "local",
        "user_id": "test-user",
        "agent_id": "test-agent",
        "rerank": True,
        "local": {
            "llm_provider": "hermes",
            "embedding": {
                "provider": "ollama",
                "model": "qwen3-embedding:4b",
                "base_url": "http://localhost:11434",
            },
            "vector_store": {
                "provider": "qdrant",
                "path": "~/.hermes/qdrant",
            },
        },
    }

    with patch("plugins.memory.mem0._load_config", return_value=config):
        with patch("plugins.memory.mem0._check_local_runtime", return_value=(True, None)):
            # Test is_available
            assert provider.is_available() is True

            # Test initialize
            provider.initialize("test-session")
            assert provider._mode == "local"
            assert provider._user_id == "test-user"

            # Test _build_local_config
            local_config = provider._build_local_config()
            assert "llm" in local_config
            assert "embedder" in local_config
            assert "vector_store" in local_config
            assert local_config["embedder"]["config"]["model"] == "qwen3-embedding:4b"
