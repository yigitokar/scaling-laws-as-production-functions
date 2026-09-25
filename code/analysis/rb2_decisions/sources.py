"""sources.py -- primary sources for module rb2_decisions (family readings; model-level serving code).
key -> dict(url, kind in {html, pdf, text}). Downloaded once by fetch_sources.py into data/raw/rb2_decisions/."""
AR = "https://arxiv.org/pdf/"
SOURCES = {
    # ---------------- technical reports / release posts: common-D families (family readings)
    "llama2_report": dict(url=AR + "2307.09288", kind="pdf"),
    "llama3_report": dict(url=AR + "2407.21783", kind="pdf"),
    "llama3_blog": dict(url="https://ai.meta.com/blog/meta-llama-3/", kind="html"),
    "qwen25_report": dict(url=AR + "2412.15115", kind="pdf"),
    "qwen25_blog": dict(url="https://qwenlm.github.io/blog/qwen2.5/", kind="html"),
    "qwen3_report": dict(url=AR + "2505.09388", kind="pdf"),
    "qwen3_blog": dict(url="https://qwenlm.github.io/blog/qwen3/", kind="html"),
    "deepseek_llm_report": dict(url=AR + "2401.02954", kind="pdf"),
    "yi_report": dict(url=AR + "2403.04652", kind="pdf"),
    "mpt7b_blog": dict(url="https://www.databricks.com/blog/mpt-7b", kind="html"),
    "mpt30b_blog": dict(url="https://www.databricks.com/blog/mpt-30b", kind="html"),
    "stablelm_github": dict(url="https://raw.githubusercontent.com/Stability-AI/StableLM/main/README.md", kind="text"),
    "granite30_report": dict(url="https://raw.githubusercontent.com/ibm-granite/granite-3.0-language-models/main/paper.pdf", kind="pdf"),
    "apertus_report": dict(url=AR + "2509.14233", kind="pdf"),
    "olmo3_report": dict(url=AR + "2512.13961", kind="pdf"),
    # ---------------- size-specific families (context for sub-groups sharing D)
    "llama1_report": dict(url=AR + "2302.13971", kind="pdf"),
    "qwen2_report": dict(url=AR + "2407.10671", kind="pdf"),
    "qwen_report": dict(url=AR + "2309.16609", kind="pdf"),
    "olmo2_report": dict(url=AR + "2501.00656", kind="pdf"),
    "smollm2_report": dict(url=AR + "2502.02737", kind="pdf"),
    "gemma_report": dict(url=AR + "2403.08295", kind="pdf"),
    "falcon_report": dict(url=AR + "2311.16867", kind="pdf"),
    "danube3_report": dict(url=AR + "2407.09276", kind="pdf"),
    "olmo_report": dict(url=AR + "2402.00838", kind="pdf"),
}

WB = "http://web.archive.org/web/{ts}id_/{url}"
SOURCES.update({
    # ---------------- model-level serving: archived first-party catalogs and launch posts
    "wb_dashscope_qwen_20240202": dict(url=WB.format(ts="20240202212821", url="https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-api-detailes"), kind="html"),
    "wb_dashscope_qwen_20240717": dict(url=WB.format(ts="20240717113723", url="https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-api-detailes"), kind="html"),
    "wb_modelstudio_zh_20240930": dict(url=WB.format(ts="20240930220855", url="https://help.aliyun.com/zh/model-studio/getting-started/models"), kind="html"),
    "wb_modelstudio_zh_20250429": dict(url=WB.format(ts="20250429142030", url="https://help.aliyun.com/zh/model-studio/models"), kind="html"),
    "wb_modelstudio_zh_20250512": dict(url=WB.format(ts="20250512110523", url="https://help.aliyun.com/zh/model-studio/models"), kind="html"),
    "deepseek_llm_readme": dict(url="https://raw.githubusercontent.com/deepseek-ai/DeepSeek-LLM/main/README.md", kind="text"),
    "gemma_blog": dict(url="https://blog.google/technology/developers/gemma-open-models/", kind="html"),
    "gemma2_blog": dict(url="https://blog.google/technology/developers/google-gemma-2/", kind="html"),
    "meta_ai_llama3_post": dict(url="https://about.fb.com/news/2024/04/meta-ai-assistant-built-with-llama-3/", kind="html"),
    "meta_ai_llama31_post": dict(url="https://about.fb.com/news/2024/07/meta-ai-is-now-multilingual-more-creative-and-smarter/", kind="html"),
    "meta_ai_launch_2023": dict(url="https://about.fb.com/news/2023/09/introducing-ai-powered-assistants-characters-and-creative-tools/", kind="html"),
    "llama31_blog": dict(url="https://ai.meta.com/blog/meta-llama-3-1/", kind="html"),
    "granite30_announcement": dict(url="https://www.ibm.com/new/announcements/ibm-granite-3-0-open-state-of-the-art-enterprise-models", kind="html"),
    "phi2_blog": dict(url="https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/", kind="html"),
    "olmo2_blog": dict(url="https://allenai.org/blog/olmo2", kind="html"),
    "olmo2_32b_blog": dict(url="https://allenai.org/blog/olmo2-32B", kind="html"),
    "olmo3_blog": dict(url="https://allenai.org/blog/olmo3", kind="html"),
    "smollm2_card": dict(url="https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/raw/main/README.md", kind="text"),
    "smollm_blog": dict(url="https://huggingface.co/blog/smollm", kind="html"),
    "smollm3_blog": dict(url="https://huggingface.co/blog/smollm3", kind="html"),
    "falcon2_news": dict(url="https://falconllm.tii.ae/falcon-2.html", kind="html"),
    "falcon3_blog": dict(url="https://huggingface.co/blog/falcon3", kind="html"),
    "falcon180b_blog": dict(url="https://huggingface.co/blog/falcon-180b", kind="html"),
    "apertus_card": dict(url="https://huggingface.co/swiss-ai/Apertus-70B-2509/raw/main/README.md", kind="text"),
    "marin_card": dict(url="https://huggingface.co/marin-community/marin-8b-base/raw/main/README.md", kind="text"),
    "yi_readme": dict(url="https://raw.githubusercontent.com/01-ai/Yi/main/README.md", kind="text"),
    "stablelm2_report": dict(url=AR + "2402.17834", kind="pdf"),
    "tinyllama_readme": dict(url="https://raw.githubusercontent.com/jzhang38/TinyLlama/main/README.md", kind="text"),
    "neo_readme": dict(url="https://raw.githubusercontent.com/multimodal-art-projection/MAP-NEO/main/README.md", kind="text"),
    "btlm_blog": dict(url="https://www.cerebras.ai/blog/btlm-3b-8k-7b-performance-in-a-3-billion-parameter-model", kind="html"),
    "danube_card": dict(url="https://huggingface.co/h2oai/h2o-danube-1.8b-base/raw/main/README.md", kind="text"),
    "danube3_card": dict(url="https://huggingface.co/h2oai/h2o-danube3-4b-base/raw/main/README.md", kind="text"),
    "phi15_card": dict(url="https://huggingface.co/microsoft/phi-1_5/raw/main/README.md", kind="text"),
    "llama2_blog": dict(url="https://ai.meta.com/llama/", kind="html"),
    "llama1_blog": dict(url="https://ai.meta.com/blog/large-language-model-llama-meta-ai/", kind="html"),
    "qwen2_blog": dict(url="https://qwenlm.github.io/blog/qwen2/", kind="html"),
    "qwen_readme": dict(url="https://raw.githubusercontent.com/QwenLM/Qwen/main/README.md", kind="text"),
})
SOURCES.update({
    "vb_deepseek_chat_2023": dict(url="https://venturebeat.com/ai/meet-deepseek-chat-chinas-latest-chatgpt-rival-with-a-67b-model", kind="html"),
    "wb_deepseek_api_20240410": dict(url=WB.format(ts="20240410080501", url="https://platform.deepseek.com/api-docs/"), kind="html"),
})
SOURCES.update({
    "deepseek_v3_report": dict(url=AR + "2412.19437", kind="pdf"),
    "tulu3_report": dict(url=AR + "2411.15124", kind="pdf"),
})
