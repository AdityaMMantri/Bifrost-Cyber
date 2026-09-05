def format_chat_prompt(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer does not have a chat template."
        )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )