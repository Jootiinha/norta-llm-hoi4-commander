import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict
import torch
from transformers import AutoModelForCausalLM


def remove_thinking_blocks(answer: str) -> str:
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    open_think_index = answer.lower().find("<think>")
    if open_think_index != -1:
        answer = answer[:open_think_index]

    return answer.strip()


def clean_generated_answer(answer: str) -> str:
    answer = remove_thinking_blocks(answer)

    repeated_answer_marker = "\nResposta:"
    marker_index = answer.find(repeated_answer_marker)
    if marker_index != -1:
        answer = answer[:marker_index].strip()

    repeated_sources_marker = "\nFontes:"
    first_sources_index = answer.find(repeated_sources_marker)
    if first_sources_index != -1:
        second_sources_index = answer.find(
            repeated_sources_marker,
            first_sources_index + len(repeated_sources_marker),
        )
        if second_sources_index != -1:
            answer = answer[:second_sources_index].strip()

    return answer


def load_model(model_path: str):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )


def generate_from_loaded_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=6,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return clean_generated_answer(answer)


def generate_answer(
    model_path: str,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    model = load_model(model_path)
    return generate_from_loaded_model(model, tokenizer, prompt, max_new_tokens)


class TransformersLocalLLM(LLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    tokenizer: Any
    max_new_tokens: int

    @property
    def _llm_type(self) -> str:
        return "local_transformers_causal_lm"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        answer = generate_from_loaded_model(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=int(kwargs.get("max_new_tokens") or self.max_new_tokens),
        )

        if stop:
            answer = self._apply_stop_tokens(answer, stop)

        return answer

    def _apply_stop_tokens(self, answer: str, stop: list[str]) -> str:
        stop_indexes = [
            answer.find(stop_token)
            for stop_token in stop
            if stop_token and answer.find(stop_token) != -1
        ]
        if not stop_indexes:
            return answer

        return answer[: min(stop_indexes)].rstrip()
