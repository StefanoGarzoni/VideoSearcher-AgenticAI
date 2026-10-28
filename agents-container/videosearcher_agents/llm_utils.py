from typing import Literal, Optional, Dict
import os
import logging
from pydantic_settings import BaseSettings
from openai import AsyncOpenAI
from agents.extensions.models.litellm_model import LitellmModel
from agents import OpenAIChatCompletionsModel

logger = logging.getLogger(__name__)


class LLMConfig(BaseSettings):
    provider: Literal["litellm", "openai-chat"] = "litellm"
    model: str = "gemini/gemini-2.5-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    extra_headers: Dict[str, str] = {}

    class Config:
        env_prefix = "LLM_"

    @classmethod
    def from_env(cls, agent_type: Optional[str] = None) -> "LLMConfig":
        """
        Load config from environment, supporting both LLM_API_KEY and provider-specific keys.
        LiteLLM automatically checks GEMINI_API_KEY, OPENAI_API_KEY based on model prefix.
        """
        config = cls()
        
        # For LiteLLM, we don't need to explicitly set api_key
        # It will auto-detect GEMINI_API_KEY, OPENAI_API_KEY, etc.
        # But we support LLM_API_KEY for backwards compatibility
        if config.provider == "litellm":
            # LiteLLM handles provider-specific keys internally
            # Only use LLM_API_KEY if explicitly set
            if config.api_key is None:
                # Check for LLM_API_KEY as fallback (backwards compat)
                config.api_key = os.getenv("LLM_API_KEY")
        
        return config

def get_llm_config(agent_type: Optional[str] = None) -> LLMConfig:
    """
    Get LLM configuration with agent-specific overrides.
    
    Supports environment variables:
    - GEMINI_API_KEY, OPENAI_API_KEY (auto-detected by LiteLLM based on model prefix)
    - LLM_MODEL (default model for all agents)
    - LLM_ORCHESTRATOR_MODEL (orchestrator-specific model override)
    - LLM_API_KEY (fallback, for backwards compatibility)
    
    TODO: Future enhancement - support model name-based lookup:
        get_llm_config(model_name="gpt-4o") could search for:
        - LLM_GPT4O_API_KEY or LLM_GPT_4O_API_KEY (normalized name)
        - Predefined model registry with API key mappings
        This would allow orchestrator to configure itself with just a model name.
    
    Examples:
        # Recommended: Provider-specific keys (auto-detected by LiteLLM)
        export GEMINI_API_KEY="your_key"
        export LLM_MODEL="gemini/gemini-2.0-flash-exp"
        
        # Orchestrator override
        export GEMINI_API_KEY="your_key"
        export OPENAI_API_KEY="your_key"
        export LLM_ORCHESTRATOR_MODEL="gpt-4o"  # Orchestrator uses GPT-4
        export LLM_MODEL="gemini/gemini-2.0-flash-exp"  # Agents use Gemini
        
        # Local LLM (advanced)
        export GEMINI_API_KEY="your_key"  # For agents
        export LLM_ORCHESTRATOR_MODEL="openai/local/localhost:9000"  # For orchestrator
        
    Args:
        agent_type: Optional agent type (e.g., "orchestrator", "audio")
                   If provided, reads LLM_{AGENT_TYPE}_MODEL before LLM_MODEL
    
    Returns:
        LLMConfig instance with agent-specific overrides applied
    """
    # Read global defaults first
    global_config = LLMConfig.from_env()
    
    # If no agent_type specified, return global config
    if not agent_type:
        return global_config
    
    # Build agent-specific env prefix
    agent_prefix = f"LLM_{agent_type.upper()}_"
    
    # Check for agent-specific overrides
    config_dict = {
        "provider": os.getenv(f"{agent_prefix}PROVIDER", global_config.provider),
        "model": os.getenv(f"{agent_prefix}MODEL", global_config.model),
        "api_key": os.getenv(f"{agent_prefix}API_KEY", global_config.api_key),
        "base_url": os.getenv(f"{agent_prefix}BASE_URL", global_config.base_url),
    }
    
    logger.debug(f"LLM config for agent_type='{agent_type}': model={config_dict['model']}, provider={config_dict['provider']}")
    
    return LLMConfig(**config_dict)


def resolve_llm_config(
    agent_type: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    use_local_llm: bool = False,
    local_base_url_env: str = "LLM_ORCHESTRATOR_BASE_URL",
    local_default_base_url: str = "http://localhost:9000/v1",
    local_default_model: str = "local-model"
) -> LLMConfig:
    """
    Resolve LLM configuration from explicit overrides with env fallbacks.

    Resolution order:
     1) Start from get_llm_config(agent_type) (env-based defaults and overrides).
     2) Apply explicit parameters (api_key/model/provider/base_url) if provided.
     3) If use_local_llm is True, force provider to openai-chat and set a local
         base URL/model using explicit values or env/default fallbacks.
    
    Args:
        agent_type: Optional agent type for env overrides (e.g., "orchestrator").
        llm_api_key: Explicit API key override.
        llm_model_name: Explicit model name override.
        llm_provider: Explicit provider override.
        llm_base_url: Explicit base URL override.
        use_local_llm: When True, force openai-chat with local endpoint defaults.
        local_base_url_env: Env var to use for local base URL fallback.
        local_default_base_url: Default local base URL.
        local_default_model: Default model name for local endpoints.

    Returns:
        Resolved LLMConfig instance.
    """
    cfg = get_llm_config(agent_type=agent_type)

    effective_model = llm_model_name
    effective_provider = llm_provider
    effective_base_url = llm_base_url

    if use_local_llm:
        effective_provider = "openai-chat"
        effective_base_url = effective_base_url or os.getenv(
            local_base_url_env,
            local_default_base_url
        )
        effective_model = effective_model or local_default_model

    cfg.api_key = llm_api_key or cfg.api_key
    cfg.model = effective_model or cfg.model
    cfg.provider = effective_provider or cfg.provider
    cfg.base_url = effective_base_url or cfg.base_url

    return cfg


def build_llm(cfg: LLMConfig):
    """
    Build LLM model instance from configuration.
    
    For LiteLLM provider, API keys are auto-detected:
    - gemini/* models → checks GEMINI_API_KEY
    - gpt-* models → checks OPENAI_API_KEY
    - Other models → checks provider-specific env vars
    
    Args:
        cfg: LLMConfig instance
        
    Returns:
        OpenAIChatCompletionsModel or LitellmModel instance
    """
    if cfg.provider == "openai-chat":
        # For direct OpenAI client (local LLMs, custom endpoints)
        if not cfg.api_key:
            cfg.api_key = "not-needed"  # Local servers may not need keys
        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        return OpenAIChatCompletionsModel(
            openai_client=client,
            model=cfg.model
        )
    
    # LiteLLM provider - auto-detects API keys from environment
    # No need to explicitly pass api_key, LiteLLM checks:
    # - GEMINI_API_KEY for gemini/* models
    # - OPENAI_API_KEY for gpt-* models
    # - etc.
    return LitellmModel(
        api_key=cfg.api_key,  # Can be None, LiteLLM will check env vars
        model=cfg.model
    )
