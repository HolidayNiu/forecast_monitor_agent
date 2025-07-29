"""
Simple LLM client without type annotations for older Python versions.
"""
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_explanation(prompt, provider="claude", **kwargs):
    """
    Get LLM explanation from specified provider.
    
    Args:
        prompt: The input prompt for the LLM
        provider: LLM provider ("claude", "openai", "databricks")
        **kwargs: Additional provider-specific parameters
    
    Returns:
        Generated explanation text
    """
    if provider.lower() == "claude":
        return _get_claude_explanation(prompt, **kwargs)
    elif provider.lower() == "openai":
        return _get_openai_explanation(prompt, **kwargs)
    else:
        raise ValueError("Unsupported LLM provider: " + provider)


def _get_claude_explanation(prompt, model="claude-3-5-sonnet-20240620", 
                           max_tokens=1000, temperature=0.3):
    """Get explanation from Claude API."""
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
        logger.error("Claude API error: " + str(e))
        raise Exception("Failed to get Claude explanation: " + str(e))


def _get_openai_explanation(prompt, model="gpt-3.5-turbo", 
                           max_tokens=1000, temperature=0.3):
    """Get explanation from OpenAI API."""
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
        logger.error("OpenAI API error: " + str(e))
        raise Exception("Failed to get OpenAI explanation: " + str(e))


def get_available_providers():
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
    
    return providers


def test_llm_connection(provider="claude"):
    """Test if LLM provider is properly configured and accessible."""
    test_prompt = "Test connection: Explain what a forecast is in one sentence."
    
    try:
        response = get_explanation(test_prompt, provider=provider)
        logger.info(provider.title() + " connection successful")
        return True
    except Exception as e:
        logger.error(provider.title() + " connection failed: " + str(e))
        return False