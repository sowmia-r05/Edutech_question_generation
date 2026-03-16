"""
Generate contextual, question-complementary images for NAPLAN questions using Gemini.

Key improvements over v1:
- Extracts numbers, shapes, objects and data directly from question text
- Per-topic prompt templates (Division, Multiplication, Fractions, Place Value,
  Patterns, Geometry 2D/3D, Measurement, Money, Time, Data/Graphs,
  Spelling, Grammar, Punctuation)
- Year-level aware (Year 3/4/5 vs Year 7/9) for age-appropriate diagrams
- Word-problem illustrator generates countable objects, not generic scenes

STRICT RULES enforced in every prompt:
  1. EXACT COUNT — if the question says 7 bananas, draw exactly 7 bananas.
     Count each object before finalising. Never approximate.
  2. NO QUESTION TEXT — do not write the question or any part of it in the image.
  3. NO ANSWER — never reveal, circle, highlight, or hint at the correct answer.
  4. OBJECT–QUESTION MATCH — every object, number, and shape in the image must
     come directly from the question. Do not add or remove anything.
"""

import logging
import os
import re
import time

from src.core.models import Question
from src.utils.s3_uploader import S3Uploader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STRICT RULES FOOTER — appended to EVERY prompt
# ─────────────────────────────────────────────────────────────────────────────

_STRICT_RULES = (
    "\n\n"
    "══════════════════════════════════════════════════════\n"
    "MANDATORY RULES — follow all of these without exception:\n"
    "══════════════════════════════════════════════════════\n"
    "1. EXACT COUNT: Count every object you draw. "
    "If the question specifies a number (e.g. 7 bananas, 12 apples, 3 groups of 4), "
    "draw EXACTLY that number — not one more, not one less. "
    "Lay objects out in a neat grid or rows so they are easy to count and verify.\n"
    "2. NO QUESTION TEXT: Do NOT write the question sentence, any part of it, "
    "or any explanatory sentence in the image. No captions, no question wording.\n"
    "3. NO ANSWER: Do NOT write, circle, highlight, underline, arrow-point, "
    "or in any way indicate the correct answer. Leave any answer blank or as '?'.\n"
    "4. MATCH THE QUESTION: Only draw objects, shapes, and numbers explicitly "
    "mentioned in the question. Do not add extra objects for decoration.\n"
    "5. WHITE BACKGROUND: Plain white background only.\n"
    "══════════════════════════════════════════════════════"
)


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

_DATA_KW = re.compile(
    r"\b(tally|tally marks?|table|chart|graph|bar graph|column graph|pie chart|"
    r"pictograph|picture graph|dot plot|line graph|votes?|voted|survey|results|"
    r"frequency|data|shows|recorded|listed|most popular|least popular)\b",
    re.IGNORECASE,
)
_GEOMETRY_2D_KW = re.compile(
    r"\b(triangle|square|rectangle|circle|pentagon|hexagon|octagon|rhombus|"
    r"trapezium|parallelogram|polygon|quadrilateral|2d shape|two.dimensional|"
    r"perimeter|area|symmetry|line of symmetry|angle|diagonal|parallel|"
    r"perpendicular|sides|vertices|vertex|corners|edges|face)\b",
    re.IGNORECASE,
)
_GEOMETRY_3D_KW = re.compile(
    r"\b(cube|rectangular prism|sphere|cone|cylinder|pyramid|prism|3d|"
    r"three.dimensional|solid|net of|faces|edges|vertices)\b",
    re.IGNORECASE,
)
_MEASUREMENT_KW = re.compile(
    r"\b(ruler|measure|length|width|height|mass|weight|thermometer|scale|"
    r"temperature|capacity|volume|litre|millilitre|kilogram|gram|kg|cm|mm|km|"
    r"metre|meter|how long|how tall|how wide|how heavy|how much does|"
    r"balance|beaker|graduated|fill|pour)\b",
    re.IGNORECASE,
)
_TIME_KW = re.compile(
    r"\b(clock|time|o'clock|half past|quarter past|quarter to|analog|digital|"
    r"minute|hour|am|pm|calendar|day|week|month|year|morning|afternoon|"
    r"how many hours|how many minutes|what time|duration|interval)\b",
    re.IGNORECASE,
)
_MONEY_KW = re.compile(
    r"\b(coin|coins?|note|dollar|\$|cent|cents|buy|pay|cost|price|change|"
    r"shop|bought|spend|receipt|wallet|afford|total cost)\b",
    re.IGNORECASE,
)
_FRACTION_KW = re.compile(
    r"\b(fraction|half|halves|quarter|quarters|third|thirds|fifths?|sixths?|"
    r"eighths?|shaded|divided|equal parts?|numerator|denominator|number line|"
    r"what fraction|how many parts)\b",
    re.IGNORECASE,
)
_PATTERN_KW = re.compile(
    r"\b(pattern|sequence|next|continue|what comes next|rule|skip count|"
    r"arrange|array|grid|term|increasing|decreasing|repeating)\b",
    re.IGNORECASE,
)
_DIVISION_KW = re.compile(
    r"\b(divid|÷|shared? (equally|between|among)|split into|how many groups|"
    r"each person gets|each row has|rows of|equal groups?)\b",
    re.IGNORECASE,
)
_MULTIPLICATION_KW = re.compile(
    r"\b(multipl|times|×|\*|groups? of|rows? of|array|repeated addition|"
    r"how many (altogether|in total)|lots? of)\b",
    re.IGNORECASE,
)
_PLACE_VALUE_KW = re.compile(
    r"\b(place value|ones|tens|hundreds|thousands|digit|expanded form|"
    r"partition|regroup|standard form|base.?10|block|abacus)\b",
    re.IGNORECASE,
)
_ALGEBRA_KW = re.compile(
    r"\b(equation|expression|variable|unknown|solve for|__ =|= __|"
    r"missing number|nth term|linear|substitute|formula|input output|"
    r"function machine|number machine)\b",
    re.IGNORECASE,
)
_SPELLING_KW = re.compile(
    r"\b(spell|spelling|correct spelling|which word|word that is correct|"
    r"misspell|missing letters?|fill in the blank)\b",
    re.IGNORECASE,
)
_GRAMMAR_KW = re.compile(
    r"\b(noun|verb|adjective|adverb|pronoun|conjunction|preposition|"
    r"subject|predicate|tense|sentence|part of speech|grammar|"
    r"describing word|doing word|naming word)\b",
    re.IGNORECASE,
)
_PUNCTUATION_KW = re.compile(
    r"\b(punctuation|capital letter|full stop|comma|question mark|"
    r"exclamation mark|apostrophe|inverted commas|speech marks|"
    r"colon|semicolon|brackets|dash|hyphen)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION-TYPE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _detect_question_type(question_text: str, sub_subject: str) -> str:
    text = f"{question_text} {sub_subject}".lower()
    if _DATA_KW.search(text):           return "data_chart"
    if _TIME_KW.search(text):           return "time"
    if _MONEY_KW.search(text):          return "money"
    if _FRACTION_KW.search(text):       return "fraction"
    if _DIVISION_KW.search(text):       return "division"
    if _MULTIPLICATION_KW.search(text): return "multiplication"
    if _PATTERN_KW.search(text):        return "pattern"
    if _PLACE_VALUE_KW.search(text):    return "place_value"
    if _ALGEBRA_KW.search(text):        return "algebra"
    if _GEOMETRY_3D_KW.search(text):    return "geometry_3d"
    if _GEOMETRY_2D_KW.search(text):    return "geometry_2d"
    if _MEASUREMENT_KW.search(text):    return "measurement"
    if _SPELLING_KW.search(text):       return "spelling"
    if _GRAMMAR_KW.search(text):        return "grammar"
    if _PUNCTUATION_KW.search(text):    return "punctuation"
    return "word_problem"


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[./]\d+)?", text)


def _extract_shapes(text: str) -> list[str]:
    shapes = re.findall(
        r"\b(triangle|square|rectangle|circle|pentagon|hexagon|octagon|"
        r"rhombus|trapezium|cube|sphere|cone|cylinder|pyramid|prism)\b",
        text, re.IGNORECASE
    )
    return list(dict.fromkeys(s.lower() for s in shapes))


def _is_year_7_9(grade: str) -> bool:
    num = re.sub(r"[^0-9]", "", grade or "")
    return num in ("7", "8", "9", "10")


def _extract_object_count(question_text: str) -> tuple[str, str]:
    """
    Extract (exact_count, object_name) from question text.
    e.g. "There are 7 bananas" -> ("7", "banana")
    Returns ("", "") if not found.
    """
    match = re.search(
        r"\b(\d+)\s+(?:\w+\s+)?"
        r"(apple|orange|banana|cake|cookie|biscuit|ball|book|bag|box|"
        r"pencil|pen|bottle|cup|glass|chair|table|tree|flower|bird|fish|"
        r"car|bus|train|dog|cat|student|child|person|people|boy|girl|"
        r"ticket|token|marble|block|cube|sticker|card|coin|toy|lolly|"
        r"grape|mango|star|heart|dot|counter|chip|bead|button|shell|"
        r"leaf|leaves|egg|sweet|lemon|strawberr|cherry|cherries)s?\b",
        question_text, re.IGNORECASE
    )
    if match:
        return match.group(1), match.group(2).lower().rstrip("s")
    return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class ImagePromptBuilder:
    """Builds tailored Gemini image prompts for each NAPLAN question type."""

    BASE_STYLE_PRIMARY = (
        "Clean, colourful educational diagram for a primary school student. "
        "White background. Simple flat-design cartoon style. "
        "Bright primary colours (red, blue, yellow, green). "
        "Large clear labels using a rounded sans-serif font. "
        "No shadows, no 3D effects. "
    )

    BASE_STYLE_SECONDARY = (
        "Clean, accurate educational diagram for a secondary school student. "
        "White background. Neat, precise technical illustration. "
        "Minimal colour palette — use blues, greens, and greys. "
        "Clearly labelled axes, tick marks, and units. "
        "Professional textbook style. "
    )

    def build(self, question: Question) -> str:
        q_text    = question.question_text
        sub       = question.sub_subject or question.subject
        grade     = question.grade or "grade3"
        q_type    = _detect_question_type(q_text, sub)
        numbers   = _extract_numbers(q_text)
        shapes    = _extract_shapes(q_text)
        secondary = _is_year_7_9(grade)
        style     = self.BASE_STYLE_SECONDARY if secondary else self.BASE_STYLE_PRIMARY

        method = getattr(self, f"_prompt_{q_type}", self._prompt_word_problem)
        # Strict rules are appended to EVERY prompt
        return method(q_text, sub, numbers, shapes, style, secondary) + _STRICT_RULES

    # ── MATHS TYPES ──────────────────────────────────────────────────────────

    def _prompt_division(self, q, sub, nums, shapes, style, secondary):
        total   = nums[0] if nums else "12"
        divisor = nums[1] if len(nums) > 1 else "3"
        try:
            each = str(int(total) // int(divisor))
        except (ValueError, ZeroDivisionError):
            each = "?"
        return (
            f"{style}"
            f"Draw a division diagram (do NOT copy or include any question text in the image):\n"
            f"Total objects: {total}. Number of groups: {divisor}. Objects per group: {each}.\n\n"
            f"Visual model:\n"
            f"- Draw exactly {total} identical simple objects (circles or stars).\n"
            f"- Arrange them into exactly {divisor} clearly separated groups "
            f"(use dashed oval borders around each group).\n"
            f"- Each group must contain exactly {each} objects. "
            f"Count every object carefully before finishing.\n"
            f"- Label each group 'Group 1', 'Group 2', etc. — nothing else.\n"
            f"- Do NOT write the division equation or the answer anywhere.\n"
            f"- Space objects so every single one is clearly visible and countable."
        )

    def _prompt_multiplication(self, q, sub, nums, shapes, style, secondary):
        rows  = nums[0] if nums else "3"
        cols  = nums[1] if len(nums) > 1 else "4"
        try:
            total = str(int(rows) * int(cols))
        except ValueError:
            total = "?"
        return (
            f"{style}"
            f"Draw a multiplication array diagram (do NOT copy or include any question text in the image):\n"
            f"Rows: {rows}. Columns per row: {cols}. Total objects: {total}.\n\n"
            f"Visual model:\n"
            f"- Draw a rectangular grid: exactly {rows} rows, each containing exactly {cols} objects.\n"
            f"- Total object count must be exactly {total}. Count every object before finishing.\n"
            f"- Use identical simple objects (stars or filled circles) — same shape in every cell.\n"
            f"- Draw a dotted brace on the left labelled '{rows} rows' "
            f"and a brace on top labelled '{cols} per row'.\n"
            f"- Do NOT write the multiplication sentence, product, or any equation.\n"
            f"- Keep the grid evenly spaced so each object is individually visible."
        )

    def _prompt_fraction(self, q, sub, nums, shapes, style, secondary):
        fraction_match = re.search(r"(\d+)\s*/\s*(\d+)", q)
        if fraction_match:
            frac_str = fraction_match.group(0)
            num_part = fraction_match.group(1)
            den_part = fraction_match.group(2)
        elif len(nums) >= 2:
            num_part, den_part = nums[0], nums[1]
            frac_str = f"{num_part}/{den_part}"
        else:
            num_part, den_part, frac_str = "1", "4", "1/4"

        shape_hint = shapes[0] if shapes else "rectangle"
        try:
            unshaded = str(int(den_part) - int(num_part))
        except ValueError:
            unshaded = "the rest"

        return (
            f"{style}"
            f"Draw a fraction diagram (do NOT copy or include any question text in the image):\n"
            f"Fraction to show: {frac_str}. Shape: {shape_hint}.\n\n"
            f"Visual model:\n"
            f"- Draw one {shape_hint} divided into exactly {den_part} EQUAL parts "
            f"(all parts must be identical in size).\n"
            f"- Shade exactly {num_part} part(s) in a bright solid colour. "
            f"Leave exactly {unshaded} part(s) unshaded (white).\n"
            f"- Write only the fraction label '{frac_str}' centred below the shape.\n"
            f"- If a number line is more appropriate: draw a horizontal line from 0 to 1 "
            f"with exactly {den_part} equal tick divisions; mark the fraction point with a filled dot.\n"
            f"- Do NOT write the question text or the answer anywhere."
        )

    def _prompt_pattern(self, q, sub, nums, shapes, style, secondary):
        sequence = re.findall(r"\d+", q)
        seq_shown = sequence[:6] if sequence else ["2", "4", "6", "8"]
        seq_str   = ", ".join(seq_shown)
        count     = len(seq_shown)
        return (
            f"{style}"
            f"Draw a number pattern diagram (do NOT copy or include any question text in the image):\n"
            f"Sequence: {seq_str}, ?\n\n"
            f"Visual model:\n"
            f"- Draw exactly {count} coloured boxes or circles in a horizontal row, evenly spaced.\n"
            f"- Write one number from the sequence inside each box: {seq_str}.\n"
            f"- After the last known term, draw one empty box containing only '?'.\n"
            f"- Draw a small arrow between each consecutive box.\n"
            f"- Label each box '1st', '2nd', '3rd', etc. below.\n"
            f"- Do NOT write the pattern rule, the answer, or the question text."
        )

    def _prompt_place_value(self, q, sub, nums, shapes, style, secondary):
        number = nums[0] if nums else "456"
        digits = list(str(number).replace(",", "").replace(".", ""))
        place_names = ["Thousands", "Hundreds", "Tens", "Ones"]
        aligned = place_names[-(len(digits)):] if len(digits) <= 4 else place_names
        return (
            f"{style}"
            f"Draw a place value table (do NOT copy or include any question text in the image):\n"
            f"Number: {number}. Digits: {', '.join(digits)}.\n\n"
            f"Visual model:\n"
            f"- Draw a 2-row table with {len(digits)} columns.\n"
            f"- Header row (coloured background): {' | '.join(aligned)}.\n"
            f"- Data row: {' | '.join(digits)} — one digit per column, large font.\n"
            f"- Use a distinct background colour for each column header.\n"
            f"- Do NOT write the number in full outside the table or show the answer."
        )

    def _prompt_algebra(self, q, sub, nums, shapes, style, secondary):
        known = nums[0] if nums else "7"
        total = nums[1] if len(nums) > 1 else "15"
        return (
            f"{style}"
            f"Draw an algebra balance-scale diagram (do NOT copy or include any question text in the image):\n"
            f"Known value: {known}. Total/result: {total}. Unknown: ?.\n\n"
            f"Visual model:\n"
            f"- Draw a balance scale (triangle fulcrum, horizontal beam, two pans).\n"
            f"- Left pan: show {known} as stacked blocks labelled '{known}'.\n"
            f"- Right pan: show {total} as stacked blocks labelled '{total}'.\n"
            f"- Place a box labelled '?' above or beside the left pan for the unknown.\n"
            f"- Do NOT write the equation, the answer, or the question text."
        )

    def _prompt_geometry_2d(self, q, sub, nums, shapes, style, secondary):
        shape_list = shapes if shapes else ["rectangle"]
        dims = nums[:4] if nums else []
        side_labels = (
            f"Label the sides with these measurements: {', '.join(d + ' cm' for d in dims[:4])}."
            if dims else "Label sides only if measurements appear in the question."
        )
        return (
            f"{style}"
            f"Draw a geometry diagram (do NOT copy or include any question text in the image):\n"
            f"Shape(s): {', '.join(shape_list)}.\n\n"
            f"Visual model:\n"
            f"- Draw the shape(s) with clean, precise straight lines (ruler-style, not freehand).\n"
            f"- {side_labels}\n"
            f"- If symmetry: draw dashed line(s) of symmetry only.\n"
            f"- If angles: mark the angle with an arc and its degree value.\n"
            f"- If shading is required: shade the described portion only.\n"
            f"- Do NOT write the area, perimeter answer, or the question text."
        )

    def _prompt_geometry_3d(self, q, sub, nums, shapes, style, secondary):
        shape_list = shapes if shapes else ["rectangular prism"]
        return (
            f"{style}"
            f"Draw a 3D solid diagram (do NOT copy or include any question text in the image):\n"
            f"Solid(s): {', '.join(shape_list)}.\n\n"
            f"Visual model:\n"
            f"- Draw the solid in slight isometric perspective so all key faces are visible.\n"
            f"- Use different shades of the same colour for each visible face to show depth.\n"
            f"- If faces/edges/vertices are asked about: label counts on the diagram.\n"
            f"- If a net is required: draw the unfolded flat net with each face outlined.\n"
            f"- Do NOT write the answer or the question text."
        )

    def _prompt_measurement(self, q, sub, nums, shapes, style, secondary):
        instrument = "ruler"
        if re.search(r"\b(mass|weight|kg|gram|balance|scale)\b", q, re.I):
            instrument = "weighing scale"
        elif re.search(r"\b(thermometer|temperature|degrees)\b", q, re.I):
            instrument = "thermometer"
        elif re.search(r"\b(capacity|litre|ml|beaker|jug|container)\b", q, re.I):
            instrument = "measuring jug"
        elif re.search(r"\b(area|square|grid|tile)\b", q, re.I):
            instrument = "square grid"

        reading = nums[0] if nums else "the value in the question"
        return (
            f"{style}"
            f"Draw a measurement diagram (do NOT copy or include any question text in the image):\n"
            f"Instrument: {instrument}. Reading: {reading}.\n\n"
            f"Visual model:\n"
            f"- Draw the {instrument} with clearly visible, evenly spaced scale markings and units.\n"
            f"- Mark the reading '{reading}' with an arrow or pointer landing EXACTLY on the "
            f"correct mark — not between marks.\n"
            f"- Do NOT write the answer or the question text."
        )

    def _prompt_time(self, q, sub, nums, shapes, style, secondary):
        is_calendar = bool(re.search(
            r"\b(calendar|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b", q, re.I
        ))
        is_digital = bool(re.search(r"\b(digital|display|screen|shows)\b", q, re.I))

        if is_calendar:
            month_match = re.search(
                r"(january|february|march|april|may|june|july|august|"
                r"september|october|november|december)", q, re.I
            )
            month = month_match.group(0).capitalize() if month_match else "March"
            return (
                f"{style}"
                f"Draw a monthly calendar (do NOT copy or include any question text in the image):\n"
                f"Month: {month}.\n\n"
                f"Visual model:\n"
                f"- Draw a calendar grid for {month}: 7 columns labelled Mon Tue Wed Thu Fri Sat Sun.\n"
                f"- Fill in all correct dates using a standard layout.\n"
                f"- Circle only the specific date(s) mentioned in the question.\n"
                f"- Do NOT write the answer or the question text."
            )

        time_vals = re.findall(r"\d{1,2}:\d{2}|\d{1,2}\s*(?:o'clock|am|pm)", q, re.I)
        time_str  = time_vals[0].strip() if time_vals else (nums[0] + ":00" if nums else "3:00")
        hour_match = re.search(r"(\d{1,2})(?::(\d{2}))?", time_str)
        hour   = hour_match.group(1) if hour_match else "3"
        minute = hour_match.group(2) if (hour_match and hour_match.group(2)) else "00"
        clock_type = "digital display" if is_digital else "analog clock face"

        return (
            f"{style}"
            f"Draw a {clock_type} (do NOT copy or include any question text in the image):\n"
            f"Time: {hour}:{minute}.\n\n"
            f"Visual model:\n"
            f"- ANALOG: round face, all 12 numbers 1–12, 60 minute tick marks. "
            f"Hour hand (short thick) at {hour}. Minute hand (long thin) at minute {minute}. "
            f"Place hands at the EXACT angle for {hour}:{minute}.\n"
            f"- DIGITAL: rectangular LCD display showing {hour}:{minute} in large 7-segment digits.\n"
            f"- Do NOT write the time as text elsewhere in the image."
        )

    def _prompt_money(self, q, sub, nums, shapes, style, secondary):
        amounts = re.findall(r"\$\s*\d+(?:\.\d{2})?|\d+\s*cents?|\d+c\b", q, re.I)
        amount_str = ", ".join(amounts[:4]) if amounts else (
            f"${nums[0]}" if nums else "the amounts in the question"
        )
        return (
            f"{style}"
            f"Draw an Australian money illustration (do NOT copy or include any question text in the image):\n"
            f"Amounts: {amount_str}.\n\n"
            f"Visual model:\n"
            f"- Draw ONLY these specific coins/notes: {amount_str}.\n"
            f"- Coins: gold $2 (large), gold $1, silver 50c (12-sided), silver 20c, "
            f"silver 10c, small silver 5c.\n"
            f"- Notes: simple rectangle — $5 pink, $10 blue, $20 red, $50 yellow, $100 green — "
            f"denomination in large text on the note.\n"
            f"- Label each coin/note with its value. Space them clearly.\n"
            f"- Do NOT draw the total, change, or the answer."
        )

    def _prompt_data_chart(self, q, sub, nums, shapes, style, secondary):
        is_tally   = bool(re.search(r"\b(tally|tally marks?)\b", q, re.I))
        is_bar     = bool(re.search(r"\b(bar graph|column graph|bar chart)\b", q, re.I))
        is_picture = bool(re.search(r"\b(picture graph|pictograph|symbol)\b", q, re.I))

        cats = re.findall(r"'([^']+)'|\"([^\"]+)\"", q)
        cat_list = [c[0] or c[1] for c in cats[:5]]
        cat_str  = ", ".join(cat_list) if cat_list else "the categories in the question"
        vals     = nums[:len(cat_list)] if nums else []
        val_str  = ", ".join(f"{c}={v}" for c, v in zip(cat_list, vals)) if vals else \
                   "values as given in the question"

        if is_tally:
            return (
                f"{style}"
                f"Draw a tally table (do NOT copy or include any question text in the image):\n"
                f"Categories: {cat_str}. Values: {val_str}.\n\n"
                f"Visual model:\n"
                f"- Draw a 3-column table: Category | Tally Marks | Total.\n"
                f"- For each category draw the EXACT tally mark count "
                f"(groups of 4 vertical + 1 diagonal for every 5).\n"
                f"- Write the numeric total in the Total column.\n"
                f"- Bold column headers, clean grid lines.\n"
                f"- Do NOT highlight the answer row."
            )

        if is_bar:
            return (
                f"{style}"
                f"Draw a bar/column graph (do NOT copy or include any question text in the image):\n"
                f"Categories: {cat_str}. Bar values: {val_str}.\n\n"
                f"Visual model:\n"
                f"- Draw vertical bars for: {cat_str}.\n"
                f"- Bar heights must exactly match: {val_str}.\n"
                f"- x-axis: category names. y-axis: numeric scale starting at 0.\n"
                f"- Different solid colour per bar. Horizontal grid lines at each interval.\n"
                f"- Do NOT indicate the answer bar."
            )

        if is_picture:
            return (
                f"{style}"
                f"Draw a pictograph (do NOT copy or include any question text in the image):\n"
                f"Categories: {cat_str}. Values: {val_str}.\n\n"
                f"Visual model:\n"
                f"- Rows for each category. Draw EXACTLY the correct number of symbols per row.\n"
                f"- Include a KEY box: 'Symbol = X units'.\n"
                f"- Symbols identical and neatly aligned in each row.\n"
                f"- Do NOT indicate the answer row."
            )

        return (
            f"{style}"
            f"Draw a data table (do NOT copy or include any question text in the image):\n"
            f"Categories: {cat_str}. Values: {val_str}.\n\n"
            f"Visual model:\n"
            f"- Table with correct column/row headers and exact values from the question.\n"
            f"- Bold coloured header row. Alternate row shading.\n"
            f"- Do NOT indicate the answer cell."
        )

    # ── LANGUAGE TYPES ───────────────────────────────────────────────────────

    def _prompt_spelling(self, q, sub, nums, shapes, style, secondary):
        return (
            f"Clean educational spelling worksheet illustration. "
            f"White background. Large bold sans-serif font.\n\n"
            f"Draw a spelling activity visual (do NOT copy or include any question text in the image):\n\n"
            f"Visual model:\n"
            f"- Fill-in-the-blank: draw the word large with missing letters as blank boxes. "
            f"No other text.\n"
            f"- Pick correct spelling: draw 3–4 word cards in rounded rectangles — "
            f"all cards identical in style.\n"
            f"- Add one small icon illustrating the word's meaning.\n"
            f"- Do NOT highlight or circle the correct spelling."
        )

    def _prompt_grammar(self, q, sub, nums, shapes, style, secondary):
        return (
            f"Clean educational grammar worksheet illustration. "
            f"White background. Large bold sans-serif font.\n\n"
            f"Draw a grammar visual (do NOT copy or include any question text in the image):\n\n"
            f"Visual model:\n"
            f"- Draw the key sentence inside a rounded speech bubble.\n"
            f"- Underline or highlight in yellow the word being tested.\n"
            f"- Add a small colour-coded tag below (e.g. blue = VERB, orange = NOUN).\n"
            f"- Verb tense: draw a Past → Present → Future arrow with a marker.\n"
            f"- Do NOT indicate which answer option is correct."
        )

    def _prompt_punctuation(self, q, sub, nums, shapes, style, secondary):
        return (
            f"Clean educational punctuation worksheet illustration. "
            f"White background. Large bold sans-serif font.\n\n"
            f"Draw a punctuation visual (do NOT copy or include any question text in the image):\n\n"
            f"Visual model:\n"
            f"- Show the key sentence in a text box with a gap symbol (▢) "
            f"where the punctuation belongs.\n"
            f"- Capital letters: lowercase with red strikethrough, correct form in green beside it.\n"
            f"- Apostrophes: two word cards (full form → contracted form) "
            f"with arrow and apostrophe in red.\n"
            f"- Do NOT reveal the correct punctuation in the gap."
        )

    def _prompt_word_problem(self, q, sub, nums, shapes, style, secondary):
        """Fallback for generic word problems — draw exact countable objects."""
        count, obj = _extract_object_count(q)

        if not obj:
            obj_match = re.search(
                r"\b(apple|orange|banana|cake|cookie|biscuit|ball|book|bag|box|"
                r"pencil|pen|bottle|cup|glass|chair|tree|flower|bird|fish|"
                r"car|bus|dog|cat|student|child|person|boy|girl|ticket|"
                r"marble|block|sticker|card|coin|toy|lolly|grape|mango|"
                r"star|heart|dot|counter|bead|button|shell|leaf|egg|sweet)\b",
                q, re.IGNORECASE
            )
            obj = obj_match.group(0).lower() if obj_match else "circle"

        if not count:
            count = nums[0] if nums else "8"

        is_split  = bool(re.search(r"\b(share|split|divide|give each|equal group)\b", q, re.I))
        is_add    = bool(re.search(r"\b(add|together|altogether|total|combined|join)\b", q, re.I))
        is_remove = bool(re.search(r"\b(take away|remove|eat|lost|left|gave away|used)\b", q, re.I))

        if is_split:
            groups = nums[1] if len(nums) > 1 else "3"
            op = (
                f"- Arrange all {count} {obj}s into exactly {groups} equally separated groups "
                f"(dashed oval border around each group). Each group must have the same count.\n"
            )
        elif is_add:
            second = nums[1] if len(nums) > 1 else "4"
            op = (
                f"- Show two separate groups side by side: "
                f"one group of exactly {count} {obj}s and a second group of exactly {second} {obj}s, "
                f"with a '+' symbol between them.\n"
            )
        elif is_remove:
            remove = nums[1] if len(nums) > 1 else "3"
            op = (
                f"- Draw exactly {count} {obj}s. "
                f"Cross out exactly {remove} of them with a clear red X.\n"
            )
        else:
            op = (
                f"- Arrange all {count} {obj}s in neat rows of 5 "
                f"so every one is clearly visible and countable.\n"
            )

        return (
            f"{style}"
            f"Draw a simple countable scene (do NOT copy or include any question text in the image):\n"
            f"Object: {obj}. Exact total count: {count}.\n\n"
            f"Visual model:\n"
            f"- Draw exactly {count} {obj}s. "
            f"Count every one before finishing — not {int(count)-1 if count.isdigit() else '?'}, "
            f"not {int(count)+1 if count.isdigit() else '?'}, exactly {count}.\n"
            f"{op}"
            f"- Each {obj} must be clearly drawn, identically styled, not overlapping, "
            f"and individually distinguishable.\n"
            f"- Keep objects large enough to count easily (max 5 per row).\n"
            f"- Do NOT write any numbers, the question text, or the answer."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN IMAGE GENERATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

IMAGEN_MODELS = {
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "imagen-3.0-fast-generate-001",
    "imagen-3.0-generate-001",
    "imagen-3.0-capability-001",
}


class ImageGenerator:
    """
    Generate contextual images using either:
      - Imagen 4  (imagen-4.0-fast-generate-001) via client.models.generate_images()
      - Gemini    (gemini-2.5-flash-image)        via client.models.generate_content()

    Auto-detected from model name — no config change needed.
    """

    DEFAULT_MODEL = "gemini-2.5-flash-image"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        s3_uploader: "S3Uploader | None" = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in .env file.")

        try:
            from google import genai
            self.genai  = genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        self.model_name      = model_name
        self.max_retries     = max_retries
        self.retry_delay     = retry_delay
        self.s3_uploader     = s3_uploader
        self._prompt_builder = ImagePromptBuilder()

        self._use_imagen = any(
            self.model_name.startswith(m.split("-generate")[0])
            for m in IMAGEN_MODELS
        ) or "imagen" in self.model_name.lower()

        logger.info(
            f"ImageGenerator ready — model: {self.model_name}  "
            f"API: {'Imagen generate_images()' if self._use_imagen else 'Gemini generate_content()'}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_question_image(
        self,
        question: "Question",
        image_style: str = "educational illustration",
    ) -> "str | None":
        try:
            prompt = self._prompt_builder.build(question)
            q_type = _detect_question_type(
                question.question_text, question.sub_subject or ""
            )
            logger.info(
                f"Generating image for Q{question.question_number} "
                f"[{q_type}]: {question.sub_subject or question.subject}"
            )
            logger.debug(f"Prompt (first 120 chars): {prompt[:120]}…")

            image_bytes = self._generate_with_retry(prompt)
            if not image_bytes:
                logger.warning(f"No image bytes returned for Q{question.question_number}")
                return None

            if self.s3_uploader:
                s3_url = self.s3_uploader.upload_image(image_bytes, question)
                logger.info(f"Uploaded to S3: {s3_url}")
                return s3_url

            logger.warning("No S3 uploader — bytes generated but not stored.")
            return None

        except Exception as e:
            logger.error(f"Error generating image for Q{question.question_number}: {e}")
            return None

    def generate_images_batch(
        self,
        questions: "list[Question]",
        image_style: str = "educational illustration",
    ) -> "dict[int, str]":
        image_urls: dict[int, str] = {}
        for question in questions:
            try:
                time.sleep(1.5)
                s3_url = self.generate_question_image(question, image_style)
                if s3_url:
                    image_urls[question.question_number] = s3_url
            except Exception as e:
                logger.warning(f"Skipping image Q{question.question_number}: {e}")

        logger.info(f"Generated {len(image_urls)}/{len(questions)} images")
        return image_urls

    # ── Internal API dispatch ─────────────────────────────────────────────────

    def _generate_with_retry(self, prompt: str) -> "bytes | None":
        for attempt in range(1, self.max_retries + 1):
            try:
                image_bytes = (
                    self._call_imagen(prompt)
                    if self._use_imagen
                    else self._call_gemini_image(prompt)
                )
                if image_bytes:
                    logger.info(f"Image generated: {len(image_bytes)} bytes")
                    return image_bytes
                logger.warning(f"No image bytes on attempt {attempt}")
                return None

            except Exception as e:
                is_last = attempt == self.max_retries
                wait    = self.retry_delay * (2 ** (attempt - 1))
                if is_last:
                    logger.error(f"Image generation failed after {self.max_retries} attempts: {e}")
                    return None
                logger.warning(f"Attempt {attempt} failed — retrying in {wait:.0f}s: {e}")
                time.sleep(wait)
        return None

    def _call_imagen(self, prompt: str) -> "bytes | None":
        from google.genai import types
        response = self.client.models.generate_images(
            model=self.model_name,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        if not response.generated_images:
            logger.warning("Imagen returned no images — prompt may have been blocked")
            return None
        try:
            image_bytes = response.generated_images[0].image.image_bytes
            if image_bytes:
                logger.info(f"Imagen bytes extracted: {len(image_bytes)}")
                return image_bytes
            logger.warning("image_bytes is empty/None")
            return None
        except AttributeError as e:
            gen_img = response.generated_images[0]
            logger.error(
                f"Unexpected Imagen response structure: {e}\n"
                f"  attrs: {[a for a in dir(gen_img) if not a.startswith('_')]}"
            )
            return None

    def _call_gemini_image(self, prompt: str) -> "bytes | None":
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        logger.warning("Gemini image response contained no IMAGE part")
        return None