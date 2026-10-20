import json


PROMPT_TEMPLATE = """Select the functions needed to fulfill the user request using the available definitions.
Return only a JSON array of function calls. Each call must contain exactly two keys:
"name" (a function name string) and "arguments" (an object of argument names and values).
Include every requested call. Use only the supplied functions and their defined parameters.
Do not include explanations or Markdown.

Available functions:
{tools}

User request:
{query}
"""


def format_prompt(query, tools):
    return PROMPT_TEMPLATE.format(query=query, tools=json.dumps(tools, ensure_ascii=False))
