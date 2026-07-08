import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"Usuário: {user_prompt}\nAssistente:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Caminho local do modelo")
    parser.add_argument("--prompt", required=True, help="Pergunta para o modelo")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    print(f"Carregando modelo: {args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    text = build_prompt(tokenizer, args.prompt)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print("\n--- RESPOSTA ---\n", flush=True)
    print(answer, flush=True)


if __name__ == "__main__":
    main()
