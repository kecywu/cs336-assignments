import re 

def mask_emails(text):

    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    new_text, count = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)

    return new_text, count


def mask_phone_numbers(text):

    # optional country code + area code + 3 digits + 4 digits
    pattern = r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    return re.subn(pattern, "|||PHONE_NUMBER|||", text)


def mask_ips(text):

    pattern = r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    return re.subn(pattern, "|||IP_ADDRESS|||", text)