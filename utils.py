"""Observe that this file can't contain local functions to avoid circular imports"""

from typing import Any, Dict
import json
import re


def generate_sources(result: Dict[str, Any]) -> str:
    """
    Generates a formatted list of unique document sources.

    Args:
        result (Dict[str, Any]): The result containing document metadata.

    Returns:
        str: A formatted list of document sources.
    """

    assert 'docs' in result.keys()

    sources = list()
    for doc in result['docs']:
        sources.append(doc.metadata)

    seen_items = set()
    unique_list = []

    for item in sources:
        if (item['title'], item['page']) not in seen_items:
            unique_list.append(item)
            seen_items.add((item['title'], item['page']))

    output = ''

    for d in unique_list:
        output += '-' + ' ' + ', '.join([f"{k}: {v}" for k, v in d.items()]) + '\n'

    output = '\n''\n''\n''\n' + '**References:**' + '\n' + output

    return output


# TODO: CASES TO ADD: if next not in string and Researcher or others , build the final string
def fix_json_string(input_string, keys=['Coder', 'Researcher', 'Searcher']): # ['Coder', 'Researcher', 'Searcher']
    """
s0 = '// Coder should act next to write the code for printing "hello world" to the terminal{ next: "Coder" }'
s1 = '{next: "FINISH"}'
s2 = "next('Coder')"
s3 = '{"next": "Researcher"}'
s4 = 'next: "Coder"'
s5 = 'next: Temper'
s6 = '{"next": "FINISH",}'
s7 = 'Coder'

cases = [s0, s1, s2, s3, s4, s5, s6, s7]

for i in cases:
    print(f'Case: {i}')
    print(fix_json_string(i))

    :param input_string:
    :param keys:
    :return:
    """
    # Regular expression pattern to match and extract the JSON object or "next('Coder')" format
    pattern = r'{\s*([^}]+)\s*}|next\s*[:=]\s*[\'"]([^\'"]+)[\'"]|next\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
    if any(key in input_string for key in keys) and 'next' not in input_string:
    #if all(key in input_string and 'next' not in input_string for key in keys):
        # next not in string and one of the roles
        for key in keys:
            if key in input_string:
                return f'{{"next": "{key}"}}'

    # Find the JSON object or "next('Coder')" format in the input string
    if ',' in input_string:
        input_string = input_string.replace(',', '')

    match = re.search(pattern, input_string)

    if match:
        # Extract the matched string
        matched_str = match.group(0)
        try:
            json.loads(matched_str)
            return matched_str
        except json.JSONDecodeError:
            pass
        # Check if the matched string is a JSON object
        if matched_str.startswith('{'):
            # Replace single quotes with double quotes to ensure valid JSON format
            matched_str = matched_str.replace("'", '"')
            matched_str = matched_str.replace('next', '"next"')
            return matched_str
        else:
            # If the matched string is "next('Coder')" or 'next: "Coder"', convert it to JSON format
            func_str = match.group(2) or match.group(3)
            json_str = '{{"next": "{}"}}'.format(func_str)
            return json_str
    else:
        if 'next' in input_string and ':' in input_string:
            chars = input_string.split(':')
            first, second = chars[0], chars[1]

            original_string = f'"{first}":"{second}"'
            result_string = '{' + original_string + '}'

            return result_string

        print("No JSON object or 'next' format found in the input string.")
        return None


def extract_json_data(text):
    import re
    # Use regular expressions to find the JSON data in the text
    match = re.search(r'\{.*\}', text)

    # pattern = r'<pre><code.*?>\s*({.*?})\s*</code></pre>'
    # match = re.search(pattern, text, re.DOTALL)

    if match:
        json_str = match.group()
        # json_str = match.group(1)
        try:
            # Try to parse the extracted JSON string
            json_data = json.loads(json_str)
            remaining_text = text.replace(json_str, "")
            return remaining_text, json_data
        except json.JSONDecodeError:
            # If parsing fails, return None or handle the error as needed
            raise ValueError('Raise error')
    else:
        # If no JSON data is found, return None
        return text, None
