def average_rgb(rgb_colors):
  """Calculates the average RGB color from a list of RGB tuples.

  Args:
    rgb_colors: A list of RGB tuples, where each tuple contains three integers
      representing the red, green, and blue values (0-255).

  Returns:
    A tuple representing the average RGB color, or None if the input list is empty.
  """
  if not rgb_colors:
    return None

  r_sum = 0
  g_sum = 0
  b_sum = 0

  for r, g, b in rgb_colors:
    r_sum += r
    g_sum += g
    b_sum += b

  num_colors = len(rgb_colors)
  r_avg = r_sum / num_colors
  g_avg = g_sum / num_colors
  b_avg = b_sum / num_colors

  return (int(r_avg), int(g_avg), int(b_avg))

def beautify_text(text: str) -> str:
    """
    Turn things_likeThis_andThat into:
      'Things Like This And That'
    Preserves ALL-CAPS words (acronyms) as-is.
    """
    # Identify word boundaries

    breaks = [False] * len(text)
    for i in range(1, len(text)):
        if text[i] == "_":
            breaks[i] = True
        elif text[i].isupper() and text[i-1].islower():
            breaks[i] = True

    # Extract words
    words = []
    current = []
    for i, ch in enumerate(text):
        if ch == "_":
            if current:
                words.append("".join(current))
                current = []
        elif breaks[i]:
            if current:
                words.append("".join(current))
            current = [ch]
        else:
            current.append(ch)
    if current:
        words.append("".join(current))

    # Transform: preserve all-caps, otherwise title-case
    return " ".join(w if w.isupper() else w.capitalize() for w in words)

def is_numeric(text):
    """Validate if text can be safely converted to a number without exceptions."""
    if not isinstance(text, str) or not text:
        return False

    # Handle commas in number format
    text = text.replace(",", "")

    # Check for decimal numbers
    if text.count(".") <= 1:
        # Remove one decimal point if it exists
        text = text.replace(".", "", 1)

    # Check for sign character at beginning
    if text and text[0] in "+-":
        text = text[1:]

    # If we're left with only digits, it's numeric
    return text.isdigit()

def format_percent(x: float, decimals: int = 2, max_value: float = 1000.0) -> str:
    """
    Format a fraction x (e.g. 0.9999) as a percentage string:
      • two decimal places normally → "99.99%"
      • no trailing .00 → "100%"
      • gradient explosions → "OVERFLOW"
    """
    # Check for gradient explosion (anything beyond max_value)
    if abs(x) > max_value:
        return "OVERFLOW" if x > 0 else "-OVERFLOW"

    # Normal formatting
    s = f"{x:.{decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return s + "%"


def wrap_text(text, max_len):
    """Wrap text to max_len characters per line, breaking at word boundaries."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + 1 + len(word) <= max_len:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def smart_format(num):
    try:
        num = float(num)  # Ensure input is a number
    except (ValueError, TypeError):

        return str(num)  # If conversion fails, return as is

    if num == 0:
        return "0"
    #elif abs(num) < 1e-6:  # Use scientific notation for very small numbers
    #    return f"{num:.2e}"

    elif abs(num) >= 1e8:  # Very large → scientific
        return f"{num:.1e}"
    elif abs(num) < 0.001:  # Use 6 decimal places for small numbers
        #formatted = f"{num:,.6f}"
        return f"{num:.1e}"
    elif abs(num) < 1:  # Use 3 decimal places for numbers less than 1
        formatted = f"{num:,.3f}"
#    elif abs(num) > 1e5:  # Use 6 decimal places for small numbers
#        return f"{num:.1e}"
    elif abs(num) > 1000:  # Use no decimal places for large numbers
        formatted = f"{num:,.0f}"

    else:  # Default to 2 decimal places
        formatted = f"{num:,.2f}"

    # Remove trailing zeros and trailing decimal point if necessary
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

def store_num(number):
    formatted = f"{number:,.6f}"
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted