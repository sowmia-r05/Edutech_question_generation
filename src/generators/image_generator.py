"""
Generate contextual, question-complementary images for NAPLAN questions using Gemini.

Key improvements over v1:
- Extracts numbers, shapes, objects and data directly from question text
- Per-topic prompt templates (Division, Multiplication, Fractions, Place Value,
  Patterns, Geometry 2D/3D, Measurement, Money, Time, Data/Graphs,
  Spelling, Grammar, Punctuation)
- Year-level aware (Year 3/4/5 vs Year 7/9) for age-appropriate diagrams
- Word-problem illustrator generates countable objects, not generic scenes
"""

import logging
import os
import re
import time

from src.core.models import Question
from src.utils.s3_uploader import S3Uploader

logger = logging.getLogger(__name__)


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
# Language
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
    """
    Classify into a fine-grained visual type.
    Priority order ensures the most specific match wins.
    """
    text = f"{question_text} {sub_subject}".lower()

    if _DATA_KW.search(text):
        return "data_chart"
    if _TIME_KW.search(text):
        return "time"
    if _MONEY_KW.search(text):
        return "money"
    if _FRACTION_KW.search(text):
        return "fraction"
    if _DIVISION_KW.search(text):
        return "division"
    if _MULTIPLICATION_KW.search(text):
        return "multiplication"
    if _PATTERN_KW.search(text):
        return "pattern"
    if _PLACE_VALUE_KW.search(text):
        return "place_value"
    if _ALGEBRA_KW.search(text):
        return "algebra"
    if _GEOMETRY_3D_KW.search(text):
        return "geometry_3d"
    if _GEOMETRY_2D_KW.search(text):
        return "geometry_2d"
    if _MEASUREMENT_KW.search(text):
        return "measurement"
    if _SPELLING_KW.search(text):
        return "spelling"
    if _GRAMMAR_KW.search(text):
        return "grammar"
    if _PUNCTUATION_KW.search(text):
        return "punctuation"
    return "word_problem"


def _extract_numbers(text: str) -> list[str]:
    """Pull all numeric values and simple fractions from question text."""
    return re.findall(r"\d+(?:[./]\d+)?", text)


def _extract_shapes(text: str) -> list[str]:
    """Pull shape names from question text."""
    shapes = re.findall(
        r"\b(triangle|square|rectangle|circle|pentagon|hexagon|octagon|"
        r"rhombus|trapezium|cube|sphere|cone|cylinder|pyramid|prism)\b",
        text, re.IGNORECASE
    )
    return list(dict.fromkeys(s.lower() for s in shapes))


def _is_year_7_9(grade: str) -> bool:
    num = re.sub(r"[^0-9]", "", grade or "")
    return num in ("7", "8", "9", "10")


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
        q_text = question.question_text
        sub = question.sub_subject or question.subject
        grade = question.grade or "grade3"
        q_type = _detect_question_type(q_text, sub)
        numbers = _extract_numbers(q_text)
        shapes = _extract_shapes(q_text)
        secondary = _is_year_7_9(grade)
        style = self.BASE_STYLE_SECONDARY if secondary else self.BASE_STYLE_PRIMARY

        method = getattr(self, f"_prompt_{q_type}", self._prompt_word_problem)
        return method(q_text, sub, numbers, shapes, style, secondary)

    # ── MATHS TYPES ──────────────────────────────────────────────────────────

    def _prompt_division(self, q, sub, nums, shapes, style, secondary):
        return (
            f"{style}"
            f"Draw a division diagram that directly matches this question:\n\"{q}\"\n\n"
            f"Choose ONE of the following visual models that best fits the question:\n"
            f"  (A) SHARING MODEL: draw a set of objects (e.g. {nums[0] if nums else 12} "
            f"apples, stars, or counters) arranged in a row, then show them being "
            f"distributed equally into {nums[1] if len(nums) > 1 else '3'} groups/plates. "
            f"Each group must show the same number of objects.\n"
            f"  (B) GROUPING MODEL: draw the total objects in one pool, then circle "
            f"equal-sized groups of the divisor.\n"
            f"  (C) ARRAY MODEL: draw rows and columns (e.g. 3 rows of 4 = 12).\n\n"
            f"Rules:\n"
            f"- Choose whichever model makes the question most visual and self-explanatory.\n"
            f"- Show the exact numbers from the question as object counts.\n"
            f"- Label each group with the group size OR leave blank if asking 'how many in each?'.\n"
            f"- Do NOT write the final answer or the division symbol in the image.\n"
            f"- Use cute simple objects (stars ★, circles, apples, dots) not abstract squares.\n"
            f"- Large, clear, colourful, suitable for a student worksheet."
        )

    def _prompt_multiplication(self, q, sub, nums, shapes, style, secondary):
        rows = nums[0] if nums else "3"
        cols = nums[1] if len(nums) > 1 else "4"
        return (
            f"{style}"
            f"Draw a multiplication diagram for this question:\n\"{q}\"\n\n"
            f"Visual model to draw:\n"
            f"  ARRAY MODEL: draw a rectangular grid of objects — "
            f"{rows} rows of {cols} objects each.\n"
            f"  Use cute identical objects (stars, circles, smiley faces, dots).\n"
            f"  Label each row and each column clearly.\n"
            f"  Write the multiplication sentence ABOVE the array: "
            f"\"{rows} × {cols} = ?\"\n"
            f"  Show a dotted brace/arrow labelling the rows and columns.\n\n"
            f"Rules:\n"
            f"- Do NOT write the product/answer next to '= ?'.\n"
            f"- Keep the grid clean and evenly spaced.\n"
            f"- Use alternating colours for each row to make counting easy.\n"
            f"- Large, bright, easy to count."
        )

    def _prompt_fraction(self, q, sub, nums, shapes, style, secondary):
        # Try to detect the fraction from the question text
        fraction_match = re.search(r"(\d+)\s*/\s*(\d+)", q)
        frac_str = fraction_match.group(0) if fraction_match else (nums[0] + "/" + nums[1] if len(nums) >= 2 else "1/4")
        num = fraction_match.group(1) if fraction_match else "1"
        den = fraction_match.group(2) if fraction_match else "4"
        shape_hint = "rectangle" if not shapes else shapes[0]
        return (
            f"{style}"
            f"Draw a fraction diagram for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Draw a {shape_hint} (or circle if more appropriate) divided into "
            f"{den} EQUAL parts.\n"
            f"- Shade exactly {num} part(s) in a bright colour to represent the fraction {frac_str}.\n"
            f"- Label each part (e.g. '1/{den}') inside the sections.\n"
            f"- Write the fraction '{frac_str}' as a large label beneath the shape.\n"
            f"- If the question uses a number line, draw a number line from 0 to 1 "
            f"with {den} equal divisions and mark the fraction point.\n"
            f"- Do NOT show the correct answer to the question.\n"
            f"- Large, clear, easy for a primary student to read."
        )

    def _prompt_pattern(self, q, sub, nums, shapes, style, secondary):
        # Extract the sequence from the question if possible
        sequence = re.findall(r"\d+", q)
        seq_str = ", ".join(sequence[:5]) if sequence else "2, 4, 6, 8"
        return (
            f"{style}"
            f"Draw the number or shape pattern from this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- If it is a NUMBER PATTERN ({seq_str}…): write each number inside a "
            f"coloured circle or box, evenly spaced in a row. "
            f"Draw a blank box (with a '?') at the position asking for the missing term.\n"
            f"- If it is a SHAPE PATTERN: draw the shapes in sequence, use a blank box "
            f"with '?' at the position the student must find.\n"
            f"- Draw arrows between each term showing the direction of the pattern.\n"
            f"- Label each term '1st', '2nd', '3rd', etc. below the shapes/boxes.\n"
            f"- Use alternating bright colours for each term.\n"
            f"- Do NOT fill in the '?' box.\n"
            f"- Clear, bold, easy to read from left to right."
        )

    def _prompt_place_value(self, q, sub, nums, shapes, style, secondary):
        number = nums[0] if nums else "456"
        digits = list(str(number).replace(",", "").replace(".", ""))
        place_names = ["Thousands", "Hundreds", "Tens", "Ones"]
        # align from right
        aligned = place_names[-(len(digits)):] if len(digits) <= 4 else place_names
        return (
            f"{style}"
            f"Draw a place value diagram for this question:\n\"{q}\"\n\n"
            f"Choose the best visual:\n"
            f"  (A) PLACE VALUE TABLE: draw a 2-row table. "
            f"Top row = column headers: {' | '.join(aligned)}. "
            f"Bottom row = the digits {' | '.join(digits)}. "
            f"Use a distinct colour for each place column.\n"
            f"  (B) BASE-10 BLOCKS: draw the correct number of "
            f"large squares (hundreds), long rods (tens), and small cubes (ones) "
            f"to represent the number {number}.\n\n"
            f"Rules:\n"
            f"- Choose whichever model best matches the wording of the question.\n"
            f"- Label every column or block clearly.\n"
            f"- Do NOT show the answer to the question.\n"
            f"- Clean, colourful grid lines."
        )

    def _prompt_algebra(self, q, sub, nums, shapes, style, secondary):
        return (
            f"{style}"
            f"Draw a visual algebra or function machine diagram for:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- If it is a MISSING NUMBER equation (e.g. '7 + __ = 15'): "
            f"draw a balance scale with the left side showing the known values "
            f"and the right side showing the total, with a question mark box for the unknown.\n"
            f"- If it is a FUNCTION MACHINE: draw an input box → machine box → output box. "
            f"Label the operation inside the machine (+, -, ×, ÷ with the number). "
            f"Show the known input/output values.\n"
            f"- If it is a NUMBER PATTERN or SEQUENCE: show boxes for each term, "
            f"with arrows labelled with the rule (e.g. '+3').\n"
            f"- For Year 7/9: draw a Cartesian coordinate plane if the question "
            f"involves graphing or substitution.\n"
            f"- Do NOT show the answer.\n"
            f"- Use bold labels, bright colours, clean lines."
        )

    def _prompt_geometry_2d(self, q, sub, nums, shapes, style, secondary):
        shape_list = shapes if shapes else ["shape"]
        dims = nums[:4] if nums else []
        dim_hint = (
            f"Label the sides with the measurements {' cm, '.join(dims[:4])} cm"
            if dims else "Label any side lengths if mentioned"
        )
        return (
            f"{style}"
            f"Draw a precise geometric diagram for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Draw the shape(s): {', '.join(shape_list)}.\n"
            f"- {dim_hint}.\n"
            f"- If the question mentions SYMMETRY: draw the line(s) of symmetry "
            f"as a dashed line through the shape.\n"
            f"- If the question mentions ANGLES: mark the angle clearly with an arc "
            f"and the degree value.\n"
            f"- If the question mentions PERIMETER or AREA: label all sides, "
            f"but do NOT calculate or write the answer.\n"
            f"- If SHADING is described: shade the correct portion.\n"
            f"- Clean geometric lines using a ruler-like precision, no freehand wobbly lines.\n"
            f"- Large, clear, black outlines with coloured fills or highlights."
        )

    def _prompt_geometry_3d(self, q, sub, nums, shapes, style, secondary):
        shape_list = shapes if shapes else ["3D object"]
        return (
            f"{style}"
            f"Draw a clear 3D object diagram for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Draw the 3D solid(s): {', '.join(shape_list)} in a slight isometric "
            f"perspective so all key faces are visible.\n"
            f"- Label the number of: FACES, EDGES, VERTICES if the question asks about them.\n"
            f"- If the question is about a NET: draw the unfolded net of the solid "
            f"with each face clearly outlined and labelled.\n"
            f"- Use different shades for each visible face to show depth.\n"
            f"- Do NOT show the answer to the question.\n"
            f"- Clean, accurate, large enough to count faces/edges easily."
        )

    def _prompt_measurement(self, q, sub, nums, shapes, style, secondary):
        instrument = "ruler"
        if re.search(r"\b(mass|weight|kg|gram|balance|scale)\b", q, re.I):
            instrument = "weighing scale or balance"
        elif re.search(r"\b(thermometer|temperature|degrees)\b", q, re.I):
            instrument = "thermometer"
        elif re.search(r"\b(capacity|litre|ml|beaker|jug|container)\b", q, re.I):
            instrument = "measuring jug or beaker"
        elif re.search(r"\b(area|square|grid|tile)\b", q, re.I):
            instrument = "grid of squares"

        return (
            f"{style}"
            f"Draw a measurement diagram for this question:\n\"{q}\"\n\n"
            f"Instrument to draw: {instrument}\n\n"
            f"Instructions:\n"
            f"- Draw the {instrument} with clearly visible scale markings and units.\n"
            f"- Show the measurement value indicated in the question "
            f"({', '.join(nums[:3]) if nums else 'as described'}).\n"
            f"- Mark the reading point with a clear indicator (arrow, line, or pointer).\n"
            f"- Do NOT write the answer to the question.\n"
            f"- If it is an AREA question: draw the shape on a square grid and shade it, "
            f"but do NOT write the area total.\n"
            f"- Large readable scale markings, suitable for a worksheet."
        )

    def _prompt_time(self, q, sub, nums, shapes, style, secondary):
        # Detect if it's an analog clock, digital, or calendar question
        is_calendar = bool(re.search(r"\b(calendar|day|week|month|year|monday|tuesday|"
                                     r"wednesday|thursday|friday|saturday|sunday|january|"
                                     r"february|march|april|may|june|july|august|"
                                     r"september|october|november|december)\b", q, re.I))
        is_digital = bool(re.search(r"\b(digital|display|screen|shows)\b", q, re.I))

        if is_calendar:
            month_match = re.search(
                r"(january|february|march|april|may|june|july|august|"
                r"september|october|november|december)", q, re.I
            )
            month = month_match.group(0).capitalize() if month_match else "March"
            return (
                f"{style}"
                f"Draw a simple monthly calendar page for this question:\n\"{q}\"\n\n"
                f"Instructions:\n"
                f"- Draw a calendar grid for {month}.\n"
                f"- Show 7 columns labelled Mon, Tue, Wed, Thu, Fri, Sat, Sun.\n"
                f"- Fill in the correct dates (use a standard month layout).\n"
                f"- Highlight or circle any specific date mentioned in the question.\n"
                f"- Clean grid lines, bold day headers.\n"
                f"- Do NOT show the answer to the question."
            )

        time_vals = re.findall(r"\d{1,2}(?::\d{2})?(?:\s*(?:am|pm|o'clock))?", q, re.I)
        time_str = time_vals[0] if time_vals else "3:00"
        clock_type = "digital display" if is_digital else "analog clock face"

        return (
            f"{style}"
            f"Draw a {clock_type} showing the time for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Draw a clear {clock_type}.\n"
            f"- If ANALOG: draw a round clock face with all 12 numbers, "
            f"hour and minute hands pointing to {time_str}. "
            f"Use a thick short hand for hours and a long thin hand for minutes. "
            f"Include minute tick marks.\n"
            f"- If DIGITAL: draw a rectangular digital display showing {time_str} "
            f"in large 7-segment LCD style digits.\n"
            f"- Do NOT write the time as text below the clock.\n"
            f"- Large, clear, easy to read."
        )

    def _prompt_money(self, q, sub, nums, shapes, style, secondary):
        amounts = re.findall(r"\$\s*\d+(?:\.\d{2})?|\d+\s*cents?|\d+c\b", q, re.I)
        amount_str = ", ".join(amounts[:4]) if amounts else "the amounts shown in the question"
        return (
            f"{style}"
            f"Draw an Australian money illustration for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Draw the specific Australian coins and/or notes: {amount_str}.\n"
            f"- Australian coins: gold $2 (large), gold $1, silver 50c (dodecagon), "
            f"silver 20c, silver 10c, small silver 5c.\n"
            f"- Australian notes: $5 polymer pink/mauve, $10 blue, $20 red/orange, "
            f"$50 yellow/gold, $100 green. Show a simplified rectangular note "
            f"with the denomination printed large.\n"
            f"- Arrange coins/notes clearly, evenly spaced.\n"
            f"- Label each coin/note with its value.\n"
            f"- Do NOT draw the total or the answer.\n"
            f"- Bright, clear, realistic-cartoon style."
        )

    def _prompt_data_chart(self, q, sub, nums, shapes, style, secondary):
        # Detect if it's a tally, bar graph, table, or picture graph
        is_tally = bool(re.search(r"\b(tally|tally marks?)\b", q, re.I))
        is_bar = bool(re.search(r"\b(bar graph|column graph|bar chart)\b", q, re.I))
        is_picture = bool(re.search(r"\b(picture graph|pictograph|symbol)\b", q, re.I))
        is_table = bool(re.search(r"\b(table|list|frequency table)\b", q, re.I))

        # Extract categories if present
        cats = re.findall(r"'([^']+)'|\"([^\"]+)\"", q)
        cat_list = [c[0] or c[1] for c in cats[:5]]

        if is_tally:
            return (
                f"{style}"
                f"Draw a TALLY TABLE for this question:\n\"{q}\"\n\n"
                f"Instructions:\n"
                f"- Draw a 3-column table: Category | Tally | Total.\n"
                f"- Use the categories mentioned in the question "
                f"({', '.join(cat_list) if cat_list else 'e.g. Football, Cricket, Swimming'}).\n"
                f"- Draw the correct tally marks (groups of 5 with a diagonal cross stroke) "
                f"as described in the question.\n"
                f"- Write the numeric totals in the Total column.\n"
                f"- Bold column headers, clean grid lines, readable tally marks.\n"
                f"- Do NOT highlight or circle the correct answer to the question."
            )
        if is_bar:
            return (
                f"{style}"
                f"Draw a COLUMN / BAR GRAPH for this question:\n\"{q}\"\n\n"
                f"Instructions:\n"
                f"- Draw vertical bars for each category mentioned "
                f"({', '.join(cat_list) if cat_list else 'use the categories in the question'}).\n"
                f"- The bars must reflect the values given in the question.\n"
                f"- Label the x-axis with category names and the y-axis with a numeric scale.\n"
                f"- Title the graph based on what is being measured.\n"
                f"- Use a different bright colour for each bar.\n"
                f"- Do NOT indicate which bar is the 'correct answer'.\n"
                f"- Neat grid lines, clear axis labels."
            )
        if is_picture:
            return (
                f"{style}"
                f"Draw a PICTURE GRAPH (pictograph) for this question:\n\"{q}\"\n\n"
                f"Instructions:\n"
                f"- Draw rows for each category "
                f"({', '.join(cat_list) if cat_list else 'use categories from the question'}).\n"
                f"- Use a simple emoji-style symbol (e.g. ★ or ☺) for each unit.\n"
                f"- Include a KEY showing what each symbol represents (e.g. ★ = 2 votes).\n"
                f"- Align symbols neatly in rows.\n"
                f"- Do NOT indicate the answer.\n"
                f"- Colourful, clearly labelled."
            )
        # Default: data table
        return (
            f"{style}"
            f"Draw a neat DATA TABLE for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Draw a table with the correct columns and rows as described.\n"
            f"- Use the category names and values from the question.\n"
            f"- Bold the header row with a coloured background.\n"
            f"- Alternate row shading (white and light grey) for readability.\n"
            f"- Do NOT indicate which value is the answer.\n"
            f"- Clear borders, readable font."
        )

    # ── LANGUAGE TYPES ───────────────────────────────────────────────────────

    def _prompt_spelling(self, q, sub, nums, shapes, style, secondary):
        # Extract any blank/underline pattern
        blank_match = re.search(r"(\w+)[\s_]{2,}(\w+)|_+", q)
        return (
            f"Clean educational spelling worksheet illustration. "
            f"White background. Simple, friendly design. "
            f"Large bold sans-serif font. Bright primary colours.\n\n"
            f"Draw a spelling activity visual for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- If the question has a FILL-IN-THE-BLANK word (e.g. 'happ___'): "
            f"draw the word large with the missing letters shown as underscores or boxes. "
            f"Use a bright dashed underline for the blank section.\n"
            f"- If the question asks to IDENTIFY the correct spelling: "
            f"draw 3-4 word cards in a row, each inside a rounded rectangle. "
            f"Do NOT highlight the correct one.\n"
            f"- Add a simple, friendly icon next to the word that illustrates its meaning "
            f"(e.g. 'running' → small stick figure running).\n"
            f"- Large, legible text. No sentences, just the word(s).\n"
            f"- Do NOT circle or indicate the correct answer."
        )

    def _prompt_grammar(self, q, sub, nums, shapes, style, secondary):
        return (
            f"Clean educational grammar worksheet illustration. "
            f"White background. Simple, friendly design. "
            f"Large bold sans-serif font.\n\n"
            f"Draw a grammar visual for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Write the sentence from the question in large, clear text inside "
            f"a rounded speech-bubble or text box.\n"
            f"- Underline or highlight (in yellow) the word that is being tested "
            f"(noun/verb/adjective/adverb as relevant).\n"
            f"- Add a small colour-coded label below the highlighted word showing "
            f"its part of speech (e.g. a blue tag 'VERB', an orange tag 'NOUN').\n"
            f"- If the question is about VERB TENSE: show a small timeline arrow "
            f"labelled Past → Present → Future, with a marker at the relevant tense.\n"
            f"- If the question is about SUBJECT-VERB AGREEMENT: "
            f"draw two side-by-side boxes showing the singular vs plural form.\n"
            f"- Do NOT indicate which answer option is correct.\n"
            f"- Large font, clean and simple."
        )

    def _prompt_punctuation(self, q, sub, nums, shapes, style, secondary):
        return (
            f"Clean educational punctuation worksheet illustration. "
            f"White background. Simple, friendly design. "
            f"Large bold sans-serif font.\n\n"
            f"Draw a punctuation visual for this question:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Write the sentence from the question in large text inside a "
            f"speech bubble or clean text box.\n"
            f"- If the question is about a MISSING punctuation mark: "
            f"show the sentence with a bright red question-mark-shaped gap "
            f"(▢) where the punctuation mark belongs.\n"
            f"- If it is about CAPITAL LETTERS: highlight the relevant word in yellow, "
            f"showing the lowercase version with a red strikethrough and the "
            f"correct capitalised version in green beside it.\n"
            f"- If it is about APOSTROPHES: draw two word cards — "
            f"the full form (e.g. 'do not') and the contracted form (e.g. \"don't\") "
            f"with an arrow between them and the apostrophe highlighted in red.\n"
            f"- Do NOT show which answer is correct.\n"
            f"- Clear, well-spaced layout."
        )

    def _prompt_word_problem(self, q, sub, nums, shapes, style, secondary):
        """Fallback for generic word problems — create a countable object scene."""
        # Try to identify what objects are in the problem
        objects = re.findall(
            r"\b(apple|orange|banana|cake|cookie|biscuit|ball|book|bag|box|"
            r"pencil|pen|bottle|cup|glass|chair|table|tree|flower|bird|fish|"
            r"car|bus|train|dog|cat|student|child|person|people|boy|girl|"
            r"ticket|token|marble|block|cube|sticker|card|coin|toy|lolly|"
            r"grape|mango|star|heart|dot)\b", q, re.I
        )
        obj_str = objects[0].lower() if objects else "object"
        total = nums[0] if nums else "some"
        return (
            f"{style}"
            f"Draw a simple, countable scene for this word problem:\n\"{q}\"\n\n"
            f"Instructions:\n"
            f"- Draw {total} {obj_str}s arranged in a clear, countable layout "
            f"(rows, groups, or a spread).\n"
            f"- If the problem involves SPLITTING or SHARING: show the objects "
            f"being distributed into groups with a dividing line.\n"
            f"- If the problem involves COMBINING: show two groups with a '+' "
            f"symbol between them.\n"
            f"- If the problem involves TAKING AWAY: show the original group with "
            f"some items crossed out or faded.\n"
            f"- Keep objects large enough to count (max 24 items on screen).\n"
            f"- Do NOT write the answer or any numbers.\n"
            f"- Colourful, friendly cartoon style. White background."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN IMAGE GENERATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

# Imagen models use client.models.generate_images() — NOT generate_content().
# Gemini image models use client.models.generate_content() with IMAGE modality.
# This class auto-detects which API to use based on the model name.

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
            self.genai = genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.s3_uploader = s3_uploader
        self._prompt_builder = ImagePromptBuilder()

        # Detect API type once at init
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
        """Generate a contextual image for a question and upload to S3."""
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
        """Generate images for a batch of questions → {question_number: s3_url}."""
        image_urls: dict[int, str] = {}
        for question in questions:
            try:
                time.sleep(1.5)   # stay within rate limits
                s3_url = self.generate_question_image(question, image_style)
                if s3_url:
                    image_urls[question.question_number] = s3_url
            except Exception as e:
                logger.warning(f"Skipping image Q{question.question_number}: {e}")

        logger.info(f"Generated {len(image_urls)}/{len(questions)} images")
        return image_urls

    # ── Internal API dispatch ─────────────────────────────────────────────────

    def _generate_with_retry(self, prompt: str) -> "bytes | None":
        """Call the correct Gemini API and return raw PNG bytes, with retries."""
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
                wait = self.retry_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s
                if is_last:
                    logger.error(
                        f"Image generation failed after {self.max_retries} attempts: {e}"
                    )
                    return None
                logger.warning(f"Attempt {attempt} failed — retrying in {wait:.0f}s: {e}")
                time.sleep(wait)

        return None

    def _call_imagen(self, prompt: str) -> "bytes | None":
        """
        Call Imagen 4 via client.models.generate_images().
        Returns raw PNG bytes.

        Confirmed working path (matches Colab test):
            response.generated_images[i].image.image_bytes
        """
        from google.genai import types

        # IMPORTANT: Keep config minimal for imagen-4.0-fast-generate-001.
        # Adding aspect_ratio, safety_filter_level, or person_generation
        # causes a ClientError on this model — they are not supported.
        response = self.client.models.generate_images(
            model=self.model_name,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
            ),
        )

        if not response.generated_images:
            logger.warning("Imagen returned no images — prompt may have been blocked")
            return None

        # Exact path confirmed in Colab:  img.image.image_bytes
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
                f"  generated_images[0] attrs: {[a for a in dir(gen_img) if not a.startswith('_')]}\n"
                f"  .image attrs: {[a for a in dir(gen_img.image) if not a.startswith('_')]}"
            )
            return None

    def _call_gemini_image(self, prompt: str) -> "bytes | None":
        """
        Call Gemini image models (e.g. gemini-2.5-flash-image)
        via client.models.generate_content() with IMAGE modality.
        Returns raw PNG bytes.
        """
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