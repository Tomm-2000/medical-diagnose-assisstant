import os
import yaml
import copy
from dotenv import load_dotenv

load_dotenv()

# ========== Hugging Face API ==========
try:
    from huggingface_hub import InferenceClient
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# =======================================

# ========== قراءة إعدادات التوليد من config.yaml ==========
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        _CFG = yaml.safe_load(f) or {}
except Exception:
    _CFG = {}

_GEN = (_CFG.get("generation") or {})
_MAX_NEW = int(_GEN.get("max_new_tokens", 256))
_TEMP = float(_GEN.get("temperature", 0.0))
_TOP_P = float(_GEN.get("top_p", 1.0))
_TOP_K = int(_GEN.get("top_k", 0))
_REP = float(_GEN.get("repetition_penalty", 1.05))

# ✅ Stops من config (مهم لـ Qwen GGUF)
_STOP = _GEN.get("stop")
if not isinstance(_STOP, list) or not _STOP:
    _STOP = ["<|im_end|>", "</s>", "<|endoftext|>"]

PROVIDER = os.getenv(
    "LLM_PROVIDER",
    os.getenv("llm_provider", (_CFG.get("models") or {}).get("llm_provider", "local")),
).lower()

# ========== OpenAI-compatible client (OpenAI / Groq) ==========
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

_oai = None


def get_openai():
    if OpenAI is None:
        raise RuntimeError("حزمة openai غير مثبتة. ثبّت openai>=1.40")
    global _oai
    if _oai is None:
        _oai = OpenAI()
    return _oai


def oai_chat(system_prompt: str, user_prompt: str, model_name: str | None = None) -> str:
    client = get_openai()
    model = model_name or os.getenv(
        "OPENAI_MODEL",
        (_CFG.get("models") or {}).get("openai_model", "gpt-4o-mini"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": (user_prompt or "").strip()},
        ],
        temperature=_TEMP,
        top_p=_TOP_P if _TEMP > 0 else None,
        max_tokens=_MAX_NEW,
    )
    return resp.choices[0].message.content


def groq_chat(system_prompt: str, user_prompt: str, model_name: str | None = None) -> str:
    """
    Groq يدعم OpenAI-compatible endpoint.
    لازم يكون GROQ_API_KEY موجود كـ env var.
    """
    if OpenAI is None:
        raise RuntimeError("حزمة openai غير مثبتة. ثبّت openai>=1.40")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY غير موجود. ضيفه للـ env.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    model = model_name or (_CFG.get("models") or {}).get("llm_model", "llama-3.1-8b-instant")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": (user_prompt or "").strip()},
        ],
        temperature=_TEMP,
        top_p=_TOP_P if _TEMP > 0 else None,
        max_tokens=_MAX_NEW,
    )
    return resp.choices[0].message.content


# ========== llama.cpp (GGUF) ==========
try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None

_llama_cache: dict[str, "Llama"] = {}


def _get_llama_cpp(model_path: str):
    if Llama is None:
        raise RuntimeError("llama-cpp-python غير مثبت أو فشل الاستيراد. ثبّت llama-cpp-python.")

    model_path = (model_path or "").strip()
    if not model_path:
        raise RuntimeError("GGUF model path فارغ.")

    if model_path in _llama_cache:
        return _llama_cache[model_path]

    n_ctx = int(os.getenv("LLAMA_N_CTX", "2048"))
    n_gpu_layers = int(os.getenv("LLAMA_N_GPU_LAYERS", "20"))
    n_threads = int(os.getenv("LLAMA_N_THREADS", "8"))

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False,
    )
    _llama_cache[model_path] = llm
    return llm


def llama_cpp_chat(system_prompt: str, user_prompt: str, model_name: str) -> str:
    llm = _get_llama_cpp(model_name)
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": (user_prompt or "").strip()},
        ],
        max_tokens=_MAX_NEW,
        temperature=_TEMP,
        top_p=_TOP_P,
        top_k=_TOP_K,
        repeat_penalty=_REP,
        stop=_STOP,
    )
    choice0 = (resp.get("choices") or [{}])[0]
    msg = choice0.get("message") or {}
    content = msg.get("content")
    if content is None:
        content = choice0.get("text", "")
    return (content or "").strip()


# ========== Hugging Face API (محسّن مع debug) ==========
def generate_answer_hf_api(system_prompt: str, user_prompt: str, model_name: str | None = None) -> str:
    """
    Hugging Face Inference Providers - chat/conversational mode.
    ?? ?????? text_generation ?? Qwen ??? featherless-ai ??? ??? ???? task mismatch.
    """
    if not _HF_AVAILABLE:
        raise RuntimeError("????? huggingface_hub ??? ?????. ?????? ????????: pip install huggingface-hub")

    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN ??? ?????. ???? ??? env ?? ??? .env")

    model = model_name or (_CFG.get("models") or {}).get(
        "hf_model",
        "Qwen/Qwen2.5-3B-Instruct",
    )

    hf_provider = (os.getenv("HF_PROVIDER") or "featherless-ai").strip()

    messages = [
        {"role": "system", "content": (system_prompt or "").strip()},
        {"role": "user", "content": (user_prompt or "").strip()},
    ]

    client_kwargs = {
        "token": token,
        "timeout": 120,
    }

    if hf_provider and hf_provider.lower() != "auto":
        client_kwargs["provider"] = hf_provider

    client = InferenceClient(**client_kwargs)

    def _clean_hf_output(content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""

        # ????? ????? ??????? ?? tool traces ?? ??? providers
        markers = [
            "\nuser\n",
            "\nassistant\n",
            "\nUser:",
            "\nAssistant:",
            "<tool_call>",
            "</tool_call>",
        ]

        cut_positions = []
        for marker in markers:
            idx = content.find(marker)
            if idx > 0:
                cut_positions.append(idx)

        if cut_positions:
            content = content[:min(cut_positions)].strip()

        # ????? ??? ??????? OK ???
        if "say ok only" in (user_prompt or "").lower() and content.lower().startswith("ok"):
            return "OK"

        return content.strip()

    def _extract_content(resp) -> str:
        choices = None

        if isinstance(resp, dict):
            choices = resp.get("choices")
        else:
            choices = getattr(resp, "choices", None)

        if not choices:
            return ""

        choice0 = choices[0]

        if isinstance(choice0, dict):
            msg = choice0.get("message") or {}
        else:
            msg = getattr(choice0, "message", None)

        if isinstance(msg, dict):
            content = msg.get("content") or msg.get("reasoning") or ""
        else:
            content = (
                getattr(msg, "content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )

        return _clean_hf_output(content)

    try:
        resp = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=_MAX_NEW,
            temperature=(_TEMP if _TEMP > 0 else None),
            top_p=(_TOP_P if _TEMP > 0 else None),
            stop=_STOP,
        )

        content = _extract_content(resp)
        if content:
            return content

        raise RuntimeError(f"HF chat_completion returned empty content. Raw response: {resp}")

    except TypeError:
        # ????? ?? providers ?? ??? huggingface_hub ?? ???? ??? ???????????
        resp = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=_MAX_NEW,
        )

        content = _extract_content(resp)
        if content:
            return content

        raise RuntimeError(f"HF chat_completion returned empty content. Raw response: {resp}")

    except Exception as e:
        raise RuntimeError(
            f"HF chat_completion failed for model={model}, provider={hf_provider}: {e}"
        )


# ========== Local (transformers) ==========
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import torch
except Exception:
    AutoModelForCausalLM = AutoTokenizer = GenerationConfig = None
    torch = None

_local_cache: dict[str, tuple] = {}


def _pick_device() -> str:
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _copy_gen_config(gen_cfg):
    try:
        cls = gen_cfg.__class__
        return cls.from_dict(gen_cfg.to_dict())
    except Exception:
        return copy.deepcopy(gen_cfg)


from pathlib import Path
import re


def _to_local_path_if_any(model_name: str) -> Path | None:
    s = (model_name or "").strip()

    if re.match(r"^[A-Za-z]:[\\/]", s):
        p = Path(s)
        return p if p.exists() else None

    if s.startswith(("/", "\\")):
        p = Path(s)
        return p if p.exists() else None

    p = Path(s)
    if p.exists():
        return p

    return None


def get_local(model_name: str):
    if (AutoModelForCausalLM is None) or (AutoTokenizer is None):
        raise RuntimeError("transformers غير متاح. ثبّت transformers + torch.")

    if model_name in _local_cache:
        return _local_cache[model_name]

    device = _pick_device()

    local_path = _to_local_path_if_any(model_name)
    if local_path is not None:
        src = local_path
        local_only = True
    else:
        src = model_name
        local_only = False

    tok = AutoTokenizer.from_pretrained(
        src,
        local_files_only=local_only,
        trust_remote_code=True,
    )

    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            src,
            local_files_only=local_only,
            trust_remote_code=True,
            dtype=(torch.float16 if (torch is not None and device == "cuda") else None),
        ).to(device)
    except Exception:
        device = "cpu"
        mdl = AutoModelForCausalLM.from_pretrained(
            src,
            local_files_only=local_only,
            trust_remote_code=True,
            dtype=None,
        ).to("cpu")

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    mdl.eval()
    _local_cache[model_name] = (tok, mdl, device)
    return tok, mdl, device


def _reload_on_cpu(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl.eval()
    return tok, mdl, "cpu"


def local_generate(system_prompt: str, user_prompt: str, model_name: str) -> str:
    if torch is None or GenerationConfig is None:
        raise RuntimeError("torch/transformers غير متاحين لتشغيل local provider.")

    tok, mdl, device = get_local(model_name)

    if hasattr(tok, "apply_chat_template"):
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt.strip()})
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (system_prompt or "").strip() + "\n\n" + (user_prompt or "").strip()

    ids = tok(prompt, return_tensors="pt")
    ids = {k: v.to(device) for k, v in ids.items()}
    input_len = ids["input_ids"].shape[1]

    do_sample = True if _TEMP > 0.0 else False

    base_cfg = mdl.generation_config if getattr(mdl, "generation_config", None) is not None else GenerationConfig()
    gen_cfg = _copy_gen_config(base_cfg)

    gen_cfg.max_new_tokens = _MAX_NEW
    gen_cfg.do_sample = do_sample
    gen_cfg.repetition_penalty = _REP
    gen_cfg.pad_token_id = tok.pad_token_id
    gen_cfg.eos_token_id = tok.eos_token_id

    if do_sample:
        gen_cfg.temperature = _TEMP
        gen_cfg.top_p = _TOP_P
        gen_cfg.top_k = _TOP_K
    else:
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None

    try:
        with torch.inference_mode():
            out = mdl.generate(**ids, generation_config=gen_cfg)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            tok2, mdl2, _ = _reload_on_cpu(model_name)
            ids2 = tok2(prompt, return_tensors="pt")
            input_len2 = ids2["input_ids"].shape[1]

            base2 = mdl2.generation_config if getattr(mdl2, "generation_config", None) is not None else GenerationConfig()
            gen_cfg2 = _copy_gen_config(base2)

            gen_cfg2.max_new_tokens = _MAX_NEW
            gen_cfg2.do_sample = do_sample
            gen_cfg2.repetition_penalty = _REP
            gen_cfg2.pad_token_id = tok2.pad_token_id
            gen_cfg2.eos_token_id = tok2.eos_token_id

            if do_sample:
                gen_cfg2.temperature = _TEMP
                gen_cfg2.top_p = _TOP_P
                gen_cfg2.top_k = _TOP_K
            else:
                gen_cfg2.temperature = None
                gen_cfg2.top_p = None
                gen_cfg2.top_k = None

            with torch.inference_mode():
                out2 = mdl2.generate(**ids2, generation_config=gen_cfg2)

            gen_tokens = out2[0][input_len2:]
            return tok2.decode(gen_tokens, skip_special_tokens=True).strip()
        raise

    gen_tokens = out[0][input_len:]
    return tok.decode(gen_tokens, skip_special_tokens=True).strip()


def generate_answer(system_prompt: str, user_prompt: str, provider: str | None = None, model_name: str | None = None) -> str:
    provider = (provider or PROVIDER).lower()

    if provider == "groq":
        return groq_chat(system_prompt, user_prompt, model_name=model_name)

    if provider == "openai":
        return oai_chat(system_prompt, user_prompt, model_name=model_name)

    if provider == "hf_api":
        return generate_answer_hf_api(system_prompt, user_prompt, model_name=model_name)

    if provider == "llama_cpp":
        gguf_path = model_name or (_CFG.get("models") or {}).get("llm_model", "")
        return llama_cpp_chat(system_prompt, user_prompt, model_name=gguf_path)

    # default: transformers local
    local_name = model_name or (_CFG.get("models") or {}).get(
        "llm_model",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    return local_generate(system_prompt, user_prompt, model_name=local_name)