from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = (
    "Voce e um assistente especialista em Hearts of Iron IV. "
    "Responda sempre em portugues brasileiro. "
    "Nao use ingles, exceto nomes oficiais do jogo, focos, paises, "
    "tecnologias, recursos, efeitos ou paginas."
)
USER_PROMPT_TEMPLATE = """
Regras obrigatorias:
- Responda em portugues brasileiro usando apenas o contexto fornecido.
- Nao invente nomes de focos, bonus, predicados ou efeitos.
- Se a resposta nao estiver clara no contexto, diga explicitamente que nao encontrou informacao suficiente.
- Ao mencionar um foco, use o nome exato que aparece no contexto.
- Nao escreva raciocinio interno, analise passo a passo, tags <think> ou comentarios sobre as fontes.
- Nao repita o enunciado, nao escreva "Fonte 1", "Answer:" ou "Resposta final:".
- Responda uma unica vez.
- Use exatamente duas secoes: "Resposta:" e "Fontes:".
- Em "Fontes:", liste somente titulos de paginas presentes no contexto.

Contexto:
{context}

Pergunta:
{question}

Resposta:
""".strip()
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ]
)


def format_contexts(contexts: list[dict]) -> str:
    return "\n\n".join(
        [
            (
                f"[Fonte {i + 1}] {ctx.get('title') or ''}\n"
                f"URL: {ctx.get('url') or ''}\n"
                f"Secao: {ctx.get('section') or '-'}\n"
                f"Subsecao: {ctx.get('subsection') or '-'}\n"
                f"Trecho:\n{ctx['text']}"
            )
            for i, ctx in enumerate(contexts)
        ]
    )


def langchain_message_to_tokenizer_message(message: BaseMessage) -> dict[str, str]:
    role = "user"
    if message.type == "system":
        role = "system"
    elif message.type == "ai":
        role = "assistant"

    return {"role": role, "content": str(message.content)}


def build_langchain_messages(question: str, contexts: list[dict]) -> list[BaseMessage]:
    context_text = format_contexts(contexts)
    return RAG_PROMPT.format_messages(
        context=context_text,
        question=question,
    )


def build_prompt(
    tokenizer,
    question: str,
    contexts: list[dict],
    enable_thinking: bool,
) -> str:
    langchain_messages = build_langchain_messages(question, contexts)
    messages = [
        langchain_message_to_tokenizer_message(message)
        for message in langchain_messages
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    return "\n\n".join(str(message.content) for message in langchain_messages)
