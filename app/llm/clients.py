"""Builds OpenAI-SDK clients for OpenAI, Azure OpenAI (AI Foundry) and Ollama.

All three speak the same API, so one code path serves every provider.
"""

from openai import AzureOpenAI, OpenAI

from app.config import Settings


def build_openai_client(settings: Settings, provider: str) -> OpenAI:
    common = {"timeout": settings.llm_timeout_s, "max_retries": settings.llm_max_retries}

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for provider=openai")
        return OpenAI(api_key=settings.openai_api_key, **common)

    if provider == "azure":
        if not settings.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for provider=azure")
        if settings.azure_openai_api_key:
            return AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                **common,
            )
        # Keyless: Managed Identity in Azure, `az login` on a laptop.
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=settings.azure_openai_api_version,
            **common,
        )

    if provider == "ollama":
        return OpenAI(base_url=settings.ollama_base_url, api_key="ollama", **common)

    raise ValueError(f"Unknown provider: {provider}")
