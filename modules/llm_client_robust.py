"""
Robust LLM client that automatically tries multiple model names.
"""
import os
import logging

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# List of Claude models to try (from newest to oldest)
CLAUDE_MODELS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-sonnet-20240229", 
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229"
]


def get_explanation(prompt, provider="claude", **kwargs):
    """Get LLM explanation with automatic model fallback."""
    if provider.lower() == "claude":
        return _get_claude_explanation_robust(prompt, **kwargs)
    elif provider.lower() == "openai":
        return _get_openai_explanation(prompt, **kwargs)
    elif provider.lower() == "databricks":
        return _get_databricks_explanation(prompt, **kwargs)
    else:
        raise ValueError("Unsupported LLM provider: " + provider)


def _get_claude_explanation_robust(prompt, max_tokens=1000, temperature=0.3):
    """Get Claude explanation with automatic model fallback."""
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
    
    # Try each model until one works
    last_error = None
    for model in CLAUDE_MODELS:
        try:
            logger.info("Trying Claude model: " + model)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.content[0].text.strip()
            logger.info("Success with model: " + model)
            return result
            
        except Exception as e:
            error_msg = str(e)
            last_error = e
            
            if "404" in error_msg or "not_found" in error_msg:
                logger.warning("Model " + model + " not found, trying next...")
                continue
            elif "401" in error_msg or "authentication" in error_msg.lower():
                logger.error("Authentication error - check your API key")
                break
            elif "403" in error_msg or "permission" in error_msg.lower():
                logger.error("Permission denied - check your API key permissions")
                break
            else:
                logger.warning("Error with model " + model + ": " + error_msg)
                continue
    
    # If we get here, all models failed
    raise Exception("All Claude models failed. Last error: " + str(last_error))


def _get_openai_explanation(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.3):
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


def _get_databricks_explanation(prompt, model=None, max_tokens=1000, temperature=0.3):
    """Get explanation from Databricks API."""
    try:
        import requests
    except ImportError:
        raise ImportError("requests package not installed. Run: pip install requests")
    
    token = os.getenv("DATABRICKS_TOKEN")
    if not token:
        raise ValueError("DATABRICKS_TOKEN environment variable not set")
    
    host = os.getenv("DATABRICKS_HOST")
    if not host:
        raise ValueError("DATABRICKS_HOST environment variable not set")
    
    # Get model endpoint name from environment or use default
    if model is None:
        model = os.getenv("DATABRICKS_MODEL_ENDPOINT", "databricks-meta-llama-3-1-405b-instruct")
    
    # Ensure host has proper format
    if not host.startswith("https://"):
        host = "https://" + host
    
    url = f"{host}/serving-endpoints/{model}/invocations"
    
    system_prompt = """You are an expert forecast analyst. Given a technical analysis of forecast issues, 
    provide a clear, human-readable explanation of what's wrong and why it matters for business planning. 
    Keep your response concise (2-3 sentences) and focus on practical implications."""
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        logger.info("Calling Databricks model: " + model)
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the response content
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"].strip()
            logger.info("Success with Databricks model: " + model)
            return content
        else:
            raise Exception("Unexpected response format from Databricks API")
            
    except requests.exceptions.RequestException as e:
        logger.error("Databricks API request error: " + str(e))
        raise Exception("Failed to get Databricks explanation: " + str(e))
    except Exception as e:
        logger.error("Databricks API error: " + str(e))
        raise Exception("Failed to get Databricks explanation: " + str(e))


def get_available_providers():
    """Check which LLM providers are available."""
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
        import requests  # Databricks uses requests, not a special SDK
        token = os.getenv("DATABRICKS_TOKEN")
        host = os.getenv("DATABRICKS_HOST")
        providers["databricks"] = bool(token and host)
    except ImportError:
        providers["databricks"] = False
    
    return providers


def test_llm_connection(provider="claude"):
    """Test if LLM provider is working with a simple prompt."""
    test_prompt = "Respond with exactly: 'Connection test successful'"
    
    try:
        response = get_explanation(test_prompt, provider=provider)
        logger.info("Connection test passed for " + provider)
        print("Test response: " + response)
        return True
    except Exception as e:
        logger.error("Connection test failed for " + provider + ": " + str(e))
        print("Connection test failed: " + str(e))
        return False