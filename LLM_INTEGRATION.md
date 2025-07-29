# LLM Integration Guide

This guide explains how to set up and use real LLM integration with the Forecast Monitor Agent.

## Overview

The system now supports multiple LLM providers:
- **Claude** (Anthropic) - Primary recommendation
- **OpenAI** (GPT models) - Alternative option
- **Databricks** (Foundation Models) - Enterprise option
- **Mock** - Fallback for testing/demo

## Quick Setup

### Option 1: Claude (Recommended)

1. **Install the Anthropic SDK:**
   ```bash
   pip install anthropic
   ```

2. **Get your API key:**
   - Go to [Anthropic Console](https://console.anthropic.com/)
   - Create an account and get an API key

3. **Set environment variable:**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

4. **Test the integration:**
   ```bash
   python test_simple_llm.py
   ```

### Option 2: OpenAI

1. **Install OpenAI SDK:**
   ```bash
   pip install openai
   ```

2. **Get API key from OpenAI and set:**
   ```bash
   export OPENAI_API_KEY='your-openai-key'
   ```

### Option 3: Databricks

1. **Set environment variables:**
   ```bash
   export DATABRICKS_API_KEY='your-databricks-token'
   export DATABRICKS_ENDPOINT_URL='your-endpoint-url'
   ```

## Usage

### In the Streamlit App

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **Configure LLM settings:**
   - In the sidebar, you'll see "🤖 LLM Settings"
   - If providers are available, check "Use Real LLM"
   - Select your preferred provider

3. **View explanations:**
   - The AI explanation will show which provider was used
   - If LLM fails, it automatically falls back to mock explanations

### Programmatic Usage

```python
from modules.explainer import generate_explanation

# Your analysis summary
analysis = "Analysis for part_A shows VOLATILITY MISMATCH..."

# Use real LLM
explanation = generate_explanation(analysis, use_mock=False, provider="claude")

# Use mock (fallback)
explanation = generate_explanation(analysis, use_mock=True)
```

### Advanced Usage

```python
from modules.llm_client import get_explanation

# Direct API call with custom parameters
explanation = get_explanation(
    prompt="Your forecast analysis...",
    provider="claude",
    model="claude-3-sonnet-20241022",
    temperature=0.1,  # More deterministic
    max_tokens=500    # Shorter responses
)
```

## Provider Comparison

| Provider | Speed | Quality | Cost | Best For |
|----------|-------|---------|------|----------|
| Claude | Fast | Excellent | Moderate | Technical analysis |
| OpenAI | Fast | Very Good | Low | General use |
| Databricks | Variable | Good | Enterprise | Large scale |
| Mock | Instant | Basic | Free | Testing/Demo |

## Troubleshooting

### "No LLM providers available"
- Check that you've installed the required packages
- Verify your API keys are set correctly
- Test with: `python test_simple_llm.py`

### "Failed to get Claude explanation"
- Check your API key is valid
- Ensure you have credits/quota remaining
- Try with `use_mock=True` as fallback

### Rate limiting
- The system automatically falls back to mock explanations
- Consider adding retry logic for production use

## System Prompts

The LLM receives this system prompt:
```
You are an expert forecast analyst. Given a technical analysis of forecast issues, 
provide a clear, human-readable explanation of what's wrong and why it matters for 
business planning. Keep your response concise (2-3 sentences) and focus on 
practical implications.
```

## Security Notes

- **Never commit API keys** to version control
- Use environment variables for API keys
- Consider using secrets management in production
- Monitor API usage and costs

## Example Workflow

1. **Setup:**
   ```bash
   pip install anthropic
   export ANTHROPIC_API_KEY='your-key'
   ```

2. **Test:**
   ```bash
   python test_simple_llm.py
   ```

3. **Run app:**
   ```bash
   streamlit run app.py
   ```

4. **Use:**
   - Select an item
   - Enable "Use Real LLM" in sidebar
   - View AI-generated explanations

## Cost Considerations

Approximate costs per explanation:
- **Claude:** ~$0.001-0.01 per explanation
- **OpenAI:** ~$0.0001-0.001 per explanation
- **Mock:** Free

For production use with many items, consider:
- Caching explanations
- Batch processing
- Setting usage limits

## Next Steps

1. **Test the integration** with your API key
2. **Customize system prompts** in `modules/llm_client.py`
3. **Add error handling** for your specific use case
4. **Consider caching** for repeated analyses