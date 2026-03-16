"""
hybrid_image_generator.py
==========================
Reuses ImageGenerator's detection + number extraction logic,
but draws with matplotlib instead of calling Gemini/Imagen.

Result: pixel-perfect diagrams, no hallucination, no API cost.
"""
from __future__ import annotations
import logging
from src.generators.image_generator import (
    _detect_question_type,
    _extract_numbers,
    _extract_shapes,
    _is_year_7_9,
    _extract_object_count,
)
from src.generators.programmatic_image_generator import (
    draw_place_value, draw_money, draw_fraction, draw_pattern,
    draw_data_chart, draw_clock, draw_division, draw_multiplication,
    draw_geometry_2d, draw_measurement, draw_text_card, draw_word_problem,
    _fig_bytes,
)
import re

logger = logging.getLogger(__name__)


class HybridImageGenerator:
    """
    Drop-in replacement for ImageGenerator.
    - Uses your existing smart detection (regex + number extraction)
    - Draws with matplotlib — exact, accurate, no hallucination
    - No Gemini image API calls needed
    """

    def __init__(self, s3_uploader=None, **kwargs):
        # Accept same kwargs as ImageGenerator so it's a true drop-in
        self.s3_uploader = s3_uploader
        logger.info("HybridImageGenerator ready — matplotlib drawing, no image API")

    # ── Public API (identical to ImageGenerator) ──────────────────────────────

    def generate_question_image(self, question, image_style: str = "") -> str | None:
        try:
            image_bytes = self._draw(question)
            if not image_bytes:
                return None
            if self.s3_uploader:
                url = self.s3_uploader.upload_image(image_bytes, question)
                logger.info(f"Q{question.question_number} uploaded: {url}")
                return url
            logger.warning("No S3 uploader configured")
            return None
        except Exception as e:
            logger.error(f"HybridImageGenerator error Q{question.question_number}: {e}")
            return None

    def generate_images_batch(self, questions, image_style: str = "") -> dict:
        import time
        urls = {}
        for q in questions:
            url = self.generate_question_image(q)
            if url:
                urls[q.question_number] = url
        logger.info(f"Generated {len(urls)}/{len(questions)} images")
        return urls

    # ── Drawing dispatcher ────────────────────────────────────────────────────

    def _draw(self, question) -> bytes | None:
        q_text  = question.question_text
        sub     = question.sub_subject or question.subject or ""
        nums    = _extract_numbers(q_text)
        shapes  = _extract_shapes(q_text)
        q_type  = _detect_question_type(q_text, sub)

        logger.info(f"Q{question.question_number} → type={q_type}, nums={nums[:4]}")

        try:
            if q_type == "place_value":
                return draw_place_value(q_text)

            elif q_type == "money":
                return draw_money(q_text)

            elif q_type == "fraction":
                return self._draw_fraction(q_text, nums, shapes)

            elif q_type == "pattern":
                return draw_pattern(q_text)

            elif q_type == "data_chart":
                return draw_data_chart(q_text)

            elif q_type == "time":
                return self._draw_time(q_text, nums)

            elif q_type == "division":
                return self._draw_division(q_text, nums)

            elif q_type == "multiplication":
                return self._draw_multiplication(q_text, nums)

            elif q_type in ("geometry_2d", "geometry_3d"):
                return draw_geometry_2d(q_text)

            elif q_type == "measurement":
                return draw_measurement(q_text)

            elif q_type in ("spelling", "grammar", "punctuation"):
                return draw_text_card(q_text, q_type)

            else:
                return self._draw_word_problem(q_text, nums)

        except Exception as e:
            logger.error(f"Draw failed [{q_type}]: {e}")
            return draw_word_problem(q_text)

    # ── Type-specific drawers with your existing extraction logic ─────────────

    def _draw_fraction(self, q_text, nums, shapes):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Reuse your fraction extraction logic from ImagePromptBuilder
        fraction_match = re.search(r"(\d+)\s*/\s*(\d+)", q_text)
        if fraction_match:
            n, d = int(fraction_match.group(1)), int(fraction_match.group(2))
        elif len(nums) >= 2:
            n, d = int(nums[0]), int(nums[1])
        else:
            n, d = 1, 4
        d = max(d, 1)

        fig, ax = plt.subplots(figsize=(max(4, d * 0.9), 3))
        ax.set_xlim(0, d); ax.set_ylim(0, 1.8)
        ax.axis("off"); fig.patch.set_facecolor("white")

        for i in range(d):
            color = "#4A90D9" if i < n else "#ECF0F1"
            edge  = "#2C3E50"
            ax.add_patch(mpatches.Rectangle(
                (i + 0.05, 0.3), 0.88, 0.9,
                facecolor=color, edgecolor=edge, linewidth=2.5
            ))

        ax.text(d / 2, 1.55, f"{n}/{d}", ha="center", va="center",
                fontsize=16, fontweight="bold", color="#2C3E50")
        ax.text(d / 2, 0.05, f"{n} out of {d} equal parts shaded",
                ha="center", fontsize=10, color="#555")

        return _fig_bytes(fig)

    def _draw_time(self, q_text, nums):
        import matplotlib.pyplot as plt
        import numpy as np

        # Use your time extraction from ImagePromptBuilder._prompt_time
        time_vals = re.findall(r"\d{1,2}:\d{2}|\d{1,2}\s*o'clock", q_text, re.I)
        if time_vals:
            match = re.search(r"(\d{1,2})(?::(\d{2}))?", time_vals[0])
            h = int(match.group(1)) % 12 if match else 3
            m = int(match.group(2)) if (match and match.group(2)) else 0
        else:
            h = int(nums[0]) % 12 if nums else 3
            m = int(nums[1]) if len(nums) > 1 else 0

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.axis("off"); fig.patch.set_facecolor("white")

        ax.add_patch(plt.Circle((0, 0), 1.2, fill=False,
                                linewidth=3, color="#2C3E50"))

        for i in range(1, 13):
            angle = np.radians(90 - i * 30)
            ax.text(1.05 * np.cos(angle), 1.05 * np.sin(angle), str(i),
                    ha="center", va="center", fontsize=11, fontweight="bold")

        # Minute marks
        for i in range(60):
            a = np.radians(90 - i * 6)
            r1, r2 = (1.12, 1.2) if i % 5 == 0 else (1.16, 1.2)
            ax.plot([r1*np.cos(a), r2*np.cos(a)], [r1*np.sin(a), r2*np.sin(a)],
                    color="#2C3E50", linewidth=1.5 if i % 5 == 0 else 0.8)

        # Hour hand — exact angle
        h_angle = np.radians(90 - (h + m / 60) * 30)
        ax.plot([0, 0.55 * np.cos(h_angle)], [0, 0.55 * np.sin(h_angle)],
                color="#2C3E50", linewidth=6, solid_capstyle="round")

        # Minute hand — exact angle
        m_angle = np.radians(90 - m * 6)
        ax.plot([0, 0.85 * np.cos(m_angle)], [0, 0.85 * np.sin(m_angle)],
                color="#4A90D9", linewidth=3, solid_capstyle="round")

        ax.add_patch(plt.Circle((0, 0), 0.05, color="#2C3E50"))
        ax.set_title(f"Time: {h}:{m:02d}", fontsize=13, fontweight="bold")
        return _fig_bytes(fig)

    def _draw_division(self, q_text, nums):
        import matplotlib.pyplot as plt

        total  = int(nums[0]) if nums else 12
        groups = int(nums[1]) if len(nums) > 1 else 3
        total  = min(total, 50)   # safety cap
        groups = max(groups, 1)
        per    = total // groups
        colors = ["#4A90D9", "#27AE60", "#E67E22", "#E74C3C", "#8E44AD"]

        fig, ax = plt.subplots(figsize=(max(6, groups * 2.5), 3.5))
        ax.set_xlim(0, groups * 2.8); ax.set_ylim(0, 3.5)
        ax.axis("off"); fig.patch.set_facecolor("white")

        for g in range(groups):
            x = g * 2.8 + 0.1
            c = colors[g % len(colors)]
            ax.add_patch(plt.Rectangle((x, 0.2), 2.4, 2.6,
                         facecolor=c + "22", edgecolor=c,
                         linewidth=2, linestyle="--"))
            ax.text(x + 1.2, 3.1, f"Group {g+1}", ha="center",
                    fontsize=9, fontweight="bold", color=c)

            cols_g = min(per, 5)
            for idx in range(per):
                cx = x + 0.4 + (idx % cols_g) * 0.42
                cy = 0.5 + (idx // cols_g) * 0.5
                ax.add_patch(plt.Circle((cx, cy), 0.17,
                             facecolor=c, edgecolor="white", linewidth=1.5))

        ax.set_title(f"{total} ÷ {groups} = {per} in each group",
                     fontsize=13, fontweight="bold", pad=8)
        return _fig_bytes(fig)

    def _draw_multiplication(self, q_text, nums):
        import matplotlib.pyplot as plt

        rows = min(int(nums[0]), 10) if nums else 3
        cols = min(int(nums[1]), 10) if len(nums) > 1 else 4
        colors = ["#4A90D9", "#E74C3C", "#27AE60", "#E67E22", "#8E44AD"]

        fig, ax = plt.subplots(figsize=(max(4, cols), max(3, rows + 1.2)))
        ax.set_xlim(-0.5, cols); ax.set_ylim(-0.9, rows + 0.4)
        ax.axis("off"); fig.patch.set_facecolor("white")

        for r in range(rows):
            for c in range(cols):
                ax.add_patch(plt.Circle((c, r), 0.35,
                             facecolor=colors[r % len(colors)],
                             edgecolor="white", linewidth=1.5))

        # Braces
        ax.annotate("", xy=(-0.45, rows-1), xytext=(-0.45, 0),
                    arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=1.5))
        ax.text(-0.48, rows/2 - 0.5, f"{rows}\nrows",
                ha="right", va="center", fontsize=9)
        ax.annotate("", xy=(cols-1, -0.55), xytext=(0, -0.55),
                    arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=1.5))
        ax.text(cols/2 - 0.5, -0.75, f"{cols} per row",
                ha="center", fontsize=9)

        ax.set_title(f"{rows} × {cols} = ?", fontsize=14, fontweight="bold", pad=8)
        return _fig_bytes(fig)

    def _draw_word_problem(self, q_text, nums):
        # Reuse your _extract_object_count logic exactly
        count, obj = _extract_object_count(q_text)
        if not count:
            count = nums[0] if nums else "8"
        return draw_word_problem(q_text)  # existing programmatic fallback