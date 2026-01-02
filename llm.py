import os
import yaml
import copy
from dotenv import load_dotenv

load_dotenv()

# ========== قراءة إعدادات التوليد من config.yaml ==========
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        _CFG = yaml.safe_load(f) or {}
except Exception:
    _CFG = {}

_GEN = (_CFG.get("generation") or {})
_MAX_NEW = int(_GEN.get("max_new_tokens", 256))
_TEMP = float(_GEN.get("temperature", 0.2))
_TOP_P = float(_GEN.get("top_p", 0.9))
_TOP_K = int(_GEN.get("top_k", 50))
_REP  = float(_GEN.get("repetition_penalty", 1.05))

# ========== OpenAI (اختياري) ==========
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

PROVIDER = os.getenv(
    "LLM_PROVIDER",
    os.getenv("llm_provider", (_CFG.get("models") or {}).get("llm_provider", "local"))
).lower()

_oai = None


def get_openai():
    if OpenAI is None:
        raise RuntimeError("حزمة openai غير مثبتة. ثبّت openai>=1.40 أو اضبط provider=local.")
    global _oai
    if _oai is None:
        _oai = OpenAI()
    return _oai


def oai_chat(system_prompt: str, user_prompt: str, model_name: str | None = None) -> str:
    client = get_openai()
    model = model_name or os.getenv(
        "OPENAI_MODEL",
        (_CFG.get("models") or {}).get("openai_model", "gpt-4o-mini")
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=_TEMP,
    )
    return resp.choices[0].message.content


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
    """
    نسخ GenerationConfig بشكل متوافق مع transformers القديمة والجديدة.
    """
    try:
        cls = gen_cfg.__class__
        return cls.from_dict(gen_cfg.to_dict())
    except Exception:
        return copy.deepcopy(gen_cfg)


def get_local(model_name: str):
    """
    تحميل tokenizer + model على الجهاز المناسب (cuda/ cpu) مع كاش داخلي.
    """
    if (AutoModelForCausalLM is None) or (AutoTokenizer is None):
        raise RuntimeError("transformers غير متاح. ثبّت transformers + torch لتشغيل المزوّد المحلي.")

    if model_name in _local_cache:
        return _local_cache[model_name]

    device = _pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)

    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=(torch.float16 if (device == "cuda") else None),  # ✅ بدل torch_dtype
        ).to(device)
    except Exception:
        device = "cpu"
        mdl = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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
    """
    - chat_template إذا متاح
    - يرجّع فقط التوكنات الجديدة
    - يستخدم GenerationConfig (لتقليل تحذيرات generation flags)
    - fallback CPU عند OOM
    """
    if torch is None or GenerationConfig is None:
        raise RuntimeError("torch/transformers غير متاحين لتشغيل المزوّد المحلي.")

    tok, mdl, device = get_local(model_name)

    # Build prompt using chat template if available
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
        # greedy: ما نحتاج sampling params
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None

    # ---- generate with OOM fallback ----
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

    if provider == "openai":
        try:
            return oai_chat(system_prompt, user_prompt, model_name=model_name)
        except Exception:
            local_name = model_name or (_CFG.get("models") or {}).get(
                "llm_model",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
            return local_generate(system_prompt, user_prompt, model_name=local_name)

    local_name = model_name or (_CFG.get("models") or {}).get(
        "llm_model",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    return local_generate(system_prompt, user_prompt, model_name=local_name)
