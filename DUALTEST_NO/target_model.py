
from openai import OpenAI


def build_completion_prompt(prefix):
    return f"""Complete the following passage continuing naturally from the text.
Return only the continuation, without explanations.

Passage:
{prefix}
"""


def query_openai_target(prefix, api_key, model="gpt-4o-mini", max_tokens=80):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": build_completion_prompt(prefix)}
        ],
        temperature=0,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()
