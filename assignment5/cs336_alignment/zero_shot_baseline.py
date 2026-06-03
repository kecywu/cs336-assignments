import re 

def parse_mmlu_response(model_output):

    prefix = "The correct answer is"
    pattern = rf"{re.escape(prefix)}\s*[:\s]*\s*\(?([A-D])\)?(?:\b|\.|\s|$)"

    match = re.search(pattern, model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None

def parse_gsm8k_response(model_output):

    if not model_output or not isinstance(model_output, str):
        return None
    
    pattern = r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?"  # allow decimals
    numbers = re.findall(pattern, model_output)
    if numbers:
        final_number_str = numbers[-1]
        return final_number_str.replace(",", "")

    return None
