import json

SYSTEM_PROMPT = """You select API calls from supplied function definitions.
Respond only with a valid JSON array of call objects.
Each object must contain exactly two keys: "name" and "arguments".
Output call data only.
Do not write Python programs, function implementations, explanations, or Markdown.
"""

PROMPT_TEMPLATE = """Generate the function calls needed to fulfill the user request using the available function definitions.

Function selection:
- Use only available functions that are needed for the request.
- Include every requested operation.
- Include separate calls when the same function must be invoked multiple times.
- Use function and parameter names exactly as defined.

Arguments:
- Include all explicitly required parameters and all parameters needed to fulfill the request.
- Use values stated in or clearly implied by the request and function definitions.
- Match the declared parameter types and constraints.
- A default value is a fallback, not a restriction on the values a parameter can accept.
- Represent lists and tuples as JSON arrays.
- Represent sets as JSON arrays with unique elements.
- Represent callable arguments as strings containing the function representation.
- Use JSON objects for dictionaries and JSON literals true, false, and null where appropriate.

Output:
- Return exactly one valid JSON array.
- Each array element must contain exactly two keys: "name" (a string) and "arguments" (an object).
- Use double quotes for JSON keys and strings.
- Return function calls without executing them or providing their results.
- Do not include explanations, Markdown fences, or text outside the JSON array.

Available functions:
{tools}

User request:
{query}
"""


def format_prompt(query, tools):
    return PROMPT_TEMPLATE.format(
        query=query,
        tools=json.dumps(tools, ensure_ascii=False),
    )

def build_messages(query, tools):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(query, tools)},
    ]