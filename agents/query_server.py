from agents.llm_local import get_llm, GenerationConfig
import os

TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")

def query_server(
    prompt: str | list[dict],
    system_prompt: str = "You are a helpful assistant",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    max_tokens: int = 128,
    num_completions: int = 1,
    server_port: int = 30000,
    server_address: str = "localhost",
    server_type: str = "sglang",
    model_name: str = "default",
    is_reasoning_model: bool = True,
    budget_tokens: int = 0,
    reasoning_effort: str = "medium",
):
    match server_type:
        case "local":
            llm = get_llm(model_name)  # legacy fallback
            model = model_name

        case "vllm":
            llm = get_llm(model_name, server_url=f"http://{server_address}:{server_port}/v1")
            model = model_name

        case "sglang":
            from openai import OpenAI
            url = f"http://{server_address}:{server_port}"
            client = OpenAI(api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0)
            model = "default"

        case "deepseek":
            from openai import OpenAI
            client = OpenAI(
                api_key=DEEPSEEK_KEY,
                base_url="https://api.deepseek.com",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name

        case "fireworks":
            from openai import OpenAI
            client = OpenAI(
                api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name

        case "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            model = model_name

        case "google":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            model = model_name

        case "together":
            from together import Together
            client = Together(api_key=TOGETHER_KEY)
            model = model_name

        case "sambanova":
            from openai import OpenAI
            client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")
            model = model_name

        case "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            model = model_name

        case _:
            raise NotImplementedError(f"Unsupported server_type: {server_type}")

    # ------------------ Local / vLLM --------------------
    if server_type in {"local", "vllm"}:
        assert isinstance(prompt, str), "Only string prompt supported for local/vllm model"
        cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        output = llm.chat(
            system_prompt,         
            prompt,
            cfg,
        )
        return output

    # ------------------ Cloud APIs ---------------------
    outputs = []

    if server_type == "google":
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )
        response = model.generate_content(prompt)
        return response.text

    elif server_type == "anthropic":
        assert isinstance(prompt, str)
        if is_reasoning_model:
            response = client.beta.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                betas=["output-128k-2025-02-19"],
            )
        else:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        outputs = [choice.text for choice in response.content]

    else:
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt

        if is_reasoning_model and server_type == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        outputs = [choice.message.content for choice in response.choices]

    return outputs[0] if len(outputs) == 1 else outputs
