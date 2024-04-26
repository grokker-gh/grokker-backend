import tiktoken


# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.get_encoding("cl100k_base")


def count():
    total_tokens = 0
    with open("grok-training.jsonl", "r") as f:
        log_line = f.read()
        total_tokens += len(enc.encode(log_line))
    print(f"Total tokens: {total_tokens}")
    cost = (total_tokens * 0.0080/1000) * 3
    print(f"Cost: {cost}")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    count()
