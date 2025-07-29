"""
Modular LLM client supporting multiple providers (Claude, OpenAI, Databricks).
"""
import os
import logging
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_explanation(prompt: str, provider: str = "claude", **kwargs) -> str:
    """
    Get LLM explanation from specified provider.
    
    Args:
        prompt: The input prompt for the LLM
        provider: LLM provider ("claude", "openai", "databricks")
        **kwargs: Additional provider-specific parameters
    
    Returns:
        Generated explanation text
    
    Raises:
        ValueError: If provider is not supported
        Exception: If API call fails
    """
    if provider.lower() == "claude":
        return _get_claude_explanation(prompt, **kwargs)
    elif provider.lower() == "openai":
        return _get_openai_explanation(prompt, **kwargs)
    elif provider.lower() == "databricks":
        return _get_databricks_explanation(prompt, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _get_claude_explanation(prompt: str, model: str = "claude-3-5-sonnet-20240620", 
                           max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """
    Get explanation from Claude API.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    system_prompt = """You are an expert forecast analyst. Given a technical analysis of forecast issues, 
    provide a clear, human-readable explanation of what's wrong and why it matters for business planning. 
    Keep your response concise (2-3 sentences) and focus on practical implications."""
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
        
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        raise Exception(f"Failed to get Claude explanation: {e}")


def _get_openai_explanation(prompt: str, model: str = "gpt-3.5-turbo", 
                           max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """
    Get explanation from OpenAI API.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    system_prompt = """You are an expert forecast analyst. Given a technical analysis of forecast issues, 
    provide a clear, human-readable explanation of what's wrong and why it matters for business planning. 
    Keep your response concise (2-3 sentences) and focus on practical implications."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise Exception(f"Failed to get OpenAI explanation: {e}")


def _get_databricks_explanation(prompt: str, endpoint_url: Optional[str] = None,
                               model: str = "databricks-meta-llama-3-1-405b-instruct",
                               max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """
    Get explanation from Databricks Foundation Model API.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests package not installed. Run: pip install requests")
    
    api_key = os.getenv("DATABRICKS_API_KEY")
    if not api_key:
        raise ValueError("DATABRICKS_API_KEY environment variable not set")
    
    # Use provided endpoint or get from environment
    if endpoint_url is None:
        endpoint_url = os.getenv("DATABRICKS_ENDPOINT_URL")
        if not endpoint_url:
            raise ValueError("DATABRICKS_ENDPOINT_URL environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are an expert forecast analyst. Given a technical analysis of forecast issues, 
    provide a clear, human-readable explanation of what's wrong and why it matters for business planning. 
    Keep your response concise (2-3 sentences) and focus on practical implications."""
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(endpoint_url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        logger.error(f"Databricks API error: {e}")
        raise Exception(f"Failed to get Databricks explanation: {e}")


def test_llm_connection(provider: str = "claude") -> bool:
    """
    Test if LLM provider is properly configured and accessible.
    
    Args:
        provider: LLM provider to test
    
    Returns:
        True if connection successful, False otherwise
    """
    test_prompt = "Test connection: Explain what a forecast is in one sentence."
    
    try:
        response = get_explanation(test_prompt, provider=provider)
        logger.info(f"{provider.title()} connection successful")
        return True
    except Exception as e:
        logger.error(f"{provider.title()} connection failed: {e}")
        return False


def get_available_providers() -> Dict[str, bool]:
    """
    Check which LLM providers are available (have required packages and API keys).
    
    Returns:
        Dictionary mapping provider names to availability status
    """
    providers = {}
    
    # Check Claude
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        providers["claude"] = bool(api_key)
    except ImportError:
        providers["claude"] = False
    
    # Check OpenAI
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        providers["openai"] = bool(api_key)
    except ImportError:
        providers["openai"] = False
    
    # Check Databricks
    try:
        import requests
        api_key = os.getenv("DATABRICKS_API_KEY")
        endpoint = os.getenv("DATABRICKS_ENDPOINT_URL")
        providers["databricks"] = bool(api_key and endpoint)
    except ImportError:
        providers["databricks"] = False
    
    return providers