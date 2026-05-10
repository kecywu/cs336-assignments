from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text

def extract_text(html_bytes):

    encoding_type = detect_encoding(html_bytes)
    decoded_string = html_bytes.decode(encoding_type, errors="replace")

    return extract_plain_text(decoded_string)