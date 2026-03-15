"""Excel exporter — writes questions into the admin dashboard import template.
Image URL is in its own separate column, NOT mixed into Question Feedback.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile

logger = logging.getLogger(__name__)


class CSVExporter:
    """Export questions into the admin dashboard ImportQuestionsTemplate.xlsx format."""

    INSTRUCTION_ROW = 1
    HEADER_ROW = 2
    DATA_START_ROW = 3

    # Core columns (matching template exactly)
    CORE_HEADERS = [
        "Question Text",            # A
        "Question Type",            # B
        "Points Type",              # C
        "Question Points",          # D
        "Page Number",              # E
        "Required",                 # F
        "Question Feedback",        # G  ← explanation only, NO image tag
        "Question Categories\n(separate each category with a comma)",  # H
        "Randomize Options",        # I
        "Option 1 Text",            # J
        "Option 1\nCorrect",        # K
        "Option 1\nPoints",         # L
        "Option 2 Text",            # M
        "Option 2\nCorrect",        # N
        "Option 2\nPoints",         # O
        "Option 3 Text",            # P
        "Option 3\nCorrect",        # Q
        "Option 3\nPoints",         # R
        "Option 4 Text",            # S
        "Option 4\nCorrect",        # T
        "Option 4\nPoints",         # U
        "Question Image",           # V  ← separate image column
    ]

    # ── Sub-subject → Major Topic lookup sets ────────────────────────────────

    _NUMBER_ALGEBRA_SUBS = {
        "counting objects", "reading numbers", "ordering numbers",
        "tens and ones", "hundreds", "value of digits",
        "simple addition", "addition with carrying",
        "simple subtraction", "subtraction with regrouping",
        "repeated addition", "groups of objects", "simple multiplication facts",
        "sharing equally", "grouping objects",
        "increasing patterns", "skip counting", "missing numbers",
        "odd and even numbers", "half", "quarter", "equal parts",
        "adding money", "subtracting money", "counting coins",
        # broader labels Gemini sometimes returns
        "place value", "addition", "subtraction", "multiplication",
        "division", "number patterns", "fractions", "money",
        "counting and number recognition", "multiplication groups and arrays",
        "division sharing", "simple fractions", "number and place value",
        "patterns and algebra", "money and financial mathematics",
        "number", "algebra", "arithmetic",
    }

    _MEASUREMENT_GEOMETRY_SUBS = {
        "length", "mass", "capacity", "time", "area",
        "2d shapes", "3d objects", "position and direction",
        "measurement", "geometry", "shapes", "angles", "symmetry",
        "perimeter", "volume", "2d shape", "3d object",
        "position", "direction",
    }

    _STATISTICS_PROBABILITY_SUBS = {
        "reading graphs", "picture graphs", "bar graphs",
        "data interpretation", "chance likely unlikely",
        "data", "statistics", "graphs", "tables",
        "probability", "chance", "data and statistics",
        "tally", "pictograph", "survey", "column graph",
    }

    _DIFFICULTY_LABELS = {
        0: "Easy", 1: "Easy", 2: "Medium", 3: "Medium", 4: "Hard", 5: "Hard"
    }

    # ─────────────────────────────────────────────────────────────────────────

    def export_to_xlsx(
        self,
        questions,
        output_path: str | None = None,
        grade: str = "grade5",
        template_path: str | None = None,
    ) -> str:
        """Write questions into the Excel template starting at row 3."""
        try:
            import openpyxl
            from openpyxl.styles import Alignment, Font, PatternFill
        except ImportError:
            raise ImportError("Run: pip install openpyxl")

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("output", exist_ok=True)
            output_path = f"output/questions_{grade}_{timestamp}.xlsx"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try to find template
        template_found = None
        if template_path and Path(template_path).exists():
            template_found = template_path
        else:
            for candidate in [
                "ImportQuestionsTemplate.xlsx",
                "ImportQuestionsTemplate__5_.xlsx",
                "ImportQuestionsTemplate__8_.xlsx",
            ]:
                if Path(candidate).exists():
                    template_found = candidate
                    break

        if template_found:
            copyfile(template_found, output_path)
            wb = openpyxl.load_workbook(output_path)
            ws = wb["Template - v1.1"]
            # Clear data rows only (keep rows 1 and 2)
            for row in ws.iter_rows(min_row=self.DATA_START_ROW, max_row=ws.max_row):
                for cell in row:
                    cell.value = None

            # Add "Question Image" header in column V (col 22) if not already there
            image_col = 22
            if ws.cell(row=self.HEADER_ROW, column=image_col).value is None:
                cell = ws.cell(row=self.HEADER_ROW, column=image_col, value="Question Image")
                cell.font = Font(bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")

            logger.info(f"Using template: {template_found}")
        else:
            logger.warning("Template not found - creating workbook from scratch.")
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Template - v1.1"

            # Write header row
            for col_idx, h in enumerate(self.CORE_HEADERS, start=1):
                cell = ws.cell(row=self.HEADER_ROW, column=col_idx, value=h)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4",
                                        fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF")
                cell.alignment = Alignment(wrap_text=True, vertical="center",
                                           horizontal="center")

        # Write question data starting at row 3
        for q_idx, q in enumerate(questions):
            row_num = self.DATA_START_ROW + q_idx
            row_data = self._build_row(q)
            for col_idx, value in enumerate(row_data, start=1):
                cell = ws.cell(row=row_num, column=col_idx, value=value)
                cell.alignment = Alignment(wrap_text=True, vertical="top")

        wb.save(output_path)
        logger.info(f"Exported {len(questions)} questions to: {output_path}")
        return output_path

    def export_to_csv(self, questions, output_path=None, grade="grade5") -> str:
        import csv
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("output", exist_ok=True)
            output_path = f"output/questions_{grade}_{timestamp}.csv"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        headers = [
            "Question Text", "Question Type", "Points Type", "Question Points",
            "Page Number", "Required", "Question Feedback", "Question Categories",
            "Randomize Options",
            "Option 1 Text", "Option 1 Correct", "Option 1 Points",
            "Option 2 Text", "Option 2 Correct", "Option 2 Points",
            "Option 3 Text", "Option 3 Correct", "Option 3 Points",
            "Option 4 Text", "Option 4 Correct", "Option 4 Points",
            "Question Image",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for q in questions:
                writer.writerow(self._build_row(q))

        logger.info(f"Exported {len(questions)} questions to CSV: {output_path}")
        return output_path

    @classmethod
    def export_with_summary(cls, questions, output_dir, base_filename) -> dict:
        exporter = cls()
        output_path = str(Path(output_dir) / f"{base_filename}.xlsx")
        out = exporter.export_to_xlsx(questions=questions, output_path=output_path)
        return {"csv": out, "summary": out}

    def _build_row(self, q) -> list:
        """
        Build one data row.
        Column G = explanation text ONLY
        Column H = clean readable categories: "Major Topic, Sub Topic, Difficulty"
        Column V = image URL ONLY (separate column)
        """
        options = [self._clean_option(opt) for opt in q.options]
        while len(options) < 4:
            options.append("")

        correct_idx = self._get_correct_index(q)

        # Feedback = explanation ONLY, no image
        feedback = getattr(q, "explanation", "") or ""

        # Image URL from artifacts or question_image field
        image_url = ""
        artifacts = getattr(q, "artifacts", None)
        if artifacts and isinstance(artifacts, list) and artifacts:
            image_url = artifacts[0]
        elif not image_url:
            image_url = getattr(q, "question_image", "") or ""

        # Categories — clean human-readable
        categories = self._build_categories(q)

        row = [
            q.question_text,    # A: Question Text
            "Multiple Choice",  # B: Question Type
            "Points",           # C: Points Type
            1,                  # D: Question Points
            1,                  # E: Page Number
            "Yes",              # F: Required
            feedback,           # G: Question Feedback (explanation ONLY)
            categories,         # H: Question Categories
            "Yes",              # I: Randomize Options
        ]

        # Options 1-4 (J to U)
        for i in range(4):
            opt_text = options[i] if i < len(options) else ""
            is_correct = "Yes" if i == correct_idx else "No"
            points = 1 if i == correct_idx else 0
            row.extend([opt_text, is_correct, points])

        # V: Question Image (separate column, URL only)
        row.append(image_url)

        return row

    def _build_categories(self, q) -> str:
        """
        Build clean human-readable categories string.
        Format: "Major Topic, Sub Topic, Difficulty"

        Examples:
          "Number & Algebra, Place Value, Easy"
          "Measurement & Geometry, Length, Medium"
          "Statistics & Probability, Bar Graphs, Hard"
        """
        sub = (getattr(q, "sub_subject", "") or "").strip()
        sub_lower = sub.lower()

        # ── Determine Major Topic ────────────────────────────────────────────
        if sub_lower in self._NUMBER_ALGEBRA_SUBS:
            major = "Number & Algebra"
        elif sub_lower in self._MEASUREMENT_GEOMETRY_SUBS:
            major = "Measurement & Geometry"
        elif sub_lower in self._STATISTICS_PROBABILITY_SUBS:
            major = "Statistics & Probability"
        else:
            # Keyword fallback for sub_subjects not in the sets
            if any(k in sub_lower for k in [
                "count", "number", "add", "subtract", "multipl", "divis",
                "fraction", "money", "pattern", "place", "odd", "even",
                "coin", "skip", "algebra", "arith",
            ]):
                major = "Number & Algebra"
            elif any(k in sub_lower for k in [
                "length", "mass", "time", "area", "shape", "object",
                "position", "capacity", "geometry", "measure", "perimeter",
                "volume", "angle", "symmetr",
            ]):
                major = "Measurement & Geometry"
            elif any(k in sub_lower for k in [
                "graph", "data", "chance", "probab", "statistic",
                "tally", "survey", "pictograph",
            ]):
                major = "Statistics & Probability"
            else:
                major = "Numeracy"

        # ── Sub Topic display — title case, clean ────────────────────────────
        sub_display = sub.title() if sub else "General"

        # ── Difficulty label ─────────────────────────────────────────────────
        diff_int = getattr(q, "difficulty", 0)
        try:
            diff_int = int(diff_int)
        except (TypeError, ValueError):
            diff_int = 0
        diff_label = self._DIFFICULTY_LABELS.get(diff_int, "Easy")

        return f"{major}, {sub_display}, {diff_label}"

    def _get_correct_index(self, q) -> int:
        val = getattr(q, "correct_option_index", None)
        if val is not None:
            return int(val)
        val = getattr(q, "answer_index", None)
        if val is not None:
            return int(val)
        val = getattr(q, "correct_answer", None)
        if val is not None:
            return {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(val).upper(), 0)
        return 0

    @staticmethod
    def _clean_option(option_text: str) -> str:
        if not option_text:
            return ""
        text = str(option_text).strip()
        if len(text) > 2 and text[1] in (".", ")") and text[0].upper() in "ABCD1234":
            return text[2:].strip()
        return text