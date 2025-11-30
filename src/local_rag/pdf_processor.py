"""PDF processing with layout and bounding box awareness."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pdfplumber


@dataclass
class BoundingBox:
    """Represents a bounding box in a PDF."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float

    @property
    def area(self) -> float:
        """Calculate the area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Get the center point of the bounding box."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


@dataclass
class TextElement:
    """Represents a text element with layout information."""

    text: str
    bbox: BoundingBox
    page_number: int
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False

    @property
    def position_context(self) -> str:
        """Generate a description of the element's position."""
        # Determine vertical position
        if self.bbox.y0 < 100:
            v_pos = "top"
        elif self.bbox.y0 > 600:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        # Determine horizontal position
        if self.bbox.x0 < 200:
            h_pos = "left"
        elif self.bbox.x0 > 400:
            h_pos = "right"
        else:
            h_pos = "center"

        return f"{v_pos}-{h_pos}"

    @property
    def is_likely_heading(self) -> bool:
        """Heuristic to determine if this is likely a heading."""
        if self.font_size and self.font_size > 12:
            return True
        if self.is_bold and len(self.text.split()) < 10:
            return True
        return False


class PDFLayoutProcessor:
    """Processes PDFs with layout and bounding box awareness."""

    def __init__(self, preserve_layout: bool = True):
        """Initialize the PDF layout processor.

        Args:
            preserve_layout: Whether to preserve layout information in extracted text
        """
        self.preserve_layout = preserve_layout

    def extract_text_elements(self, pdf_path: Path) -> List[TextElement]:
        """Extract text elements with bounding box information.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of TextElement objects with layout information
        """
        elements = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract words with their bounding boxes
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=True,
                    extra_attrs=["fontname", "size"],
                )

                # Group words into text elements (e.g., by line or proximity)
                current_line_words = []
                current_line_bbox = None

                for word in words:
                    bbox = BoundingBox(
                        x0=word["x0"],
                        y0=word["top"],
                        x1=word["x1"],
                        y1=word["bottom"],
                        width=word["x1"] - word["x0"],
                        height=word["bottom"] - word["top"],
                    )

                    font_name = word.get("fontname", "")
                    is_bold = "bold" in font_name.lower()
                    is_italic = "italic" in font_name.lower()

                    # Check if this word belongs to the current line
                    if current_line_bbox and abs(bbox.y0 - current_line_bbox.y0) < 5:
                        # Same line - add to current line
                        current_line_words.append(word)
                        # Expand bounding box
                        current_line_bbox = BoundingBox(
                            x0=min(current_line_bbox.x0, bbox.x0),
                            y0=min(current_line_bbox.y0, bbox.y0),
                            x1=max(current_line_bbox.x1, bbox.x1),
                            y1=max(current_line_bbox.y1, bbox.y1),
                            width=max(current_line_bbox.x1, bbox.x1)
                            - min(current_line_bbox.x0, bbox.x0),
                            height=max(current_line_bbox.y1, bbox.y1)
                            - min(current_line_bbox.y0, bbox.y0),
                        )
                    else:
                        # New line - save previous line if exists
                        if current_line_words:
                            line_text = " ".join(w["text"] for w in current_line_words)
                            avg_size = sum(w.get("size", 0) for w in current_line_words) / len(
                                current_line_words
                            )
                            first_font = current_line_words[0].get("fontname", "")

                            elements.append(
                                TextElement(
                                    text=line_text,
                                    bbox=current_line_bbox,
                                    page_number=page_num,
                                    font_size=avg_size,
                                    font_name=first_font,
                                    is_bold="bold" in first_font.lower(),
                                    is_italic="italic" in first_font.lower(),
                                )
                            )

                        # Start new line
                        current_line_words = [word]
                        current_line_bbox = bbox

                # Don't forget the last line
                if current_line_words:
                    line_text = " ".join(w["text"] for w in current_line_words)
                    avg_size = sum(w.get("size", 0) for w in current_line_words) / len(
                        current_line_words
                    )
                    first_font = current_line_words[0].get("fontname", "")

                    elements.append(
                        TextElement(
                            text=line_text,
                            bbox=current_line_bbox,
                            page_number=page_num,
                            font_size=avg_size,
                            font_name=first_font,
                            is_bold="bold" in first_font.lower(),
                            is_italic="italic" in first_font.lower(),
                        )
                    )

        return elements

    def format_with_layout_context(self, elements: List[TextElement]) -> str:
        """Format text elements with layout context annotations.

        Args:
            elements: List of TextElement objects

        Returns:
            Formatted text with layout annotations
        """
        formatted_parts = []
        current_page = None

        for element in elements:
            # Add page separator
            if element.page_number != current_page:
                if current_page is not None:
                    formatted_parts.append("\n" + "=" * 80 + "\n")
                formatted_parts.append(f"[PAGE {element.page_number}]\n\n")
                current_page = element.page_number

            # Add layout context
            if self.preserve_layout:
                annotations = []

                # Position
                annotations.append(f"position:{element.position_context}")

                # Heading detection
                if element.is_likely_heading:
                    annotations.append("type:heading")

                # Font size
                if element.font_size:
                    annotations.append(f"size:{element.font_size:.1f}")

                # Styling
                if element.is_bold:
                    annotations.append("style:bold")
                if element.is_italic:
                    annotations.append("style:italic")

                # Bounding box
                annotations.append(
                    f"bbox:[{element.bbox.x0:.0f},{element.bbox.y0:.0f},"
                    f"{element.bbox.x1:.0f},{element.bbox.y1:.0f}]"
                )

                context = " | ".join(annotations)
                formatted_parts.append(f"[{context}]\n{element.text}\n\n")
            else:
                formatted_parts.append(f"{element.text}\n")

        return "".join(formatted_parts)

    def process_pdf(self, pdf_path: Path) -> tuple[str, dict]:
        """Process a PDF file with layout awareness.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (formatted text, metadata)
        """
        elements = self.extract_text_elements(pdf_path)

        # Generate formatted text
        formatted_text = self.format_with_layout_context(elements)

        # Calculate statistics
        num_headings = sum(1 for e in elements if e.is_likely_heading)
        avg_font_size = (
            sum(e.font_size for e in elements if e.font_size) / len(elements)
            if elements
            else 0
        )

        # Get unique pages
        pages = set(e.page_number for e in elements)

        metadata = {
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "num_pages": len(pages),
            "file_type": "pdf",
            "num_text_elements": len(elements),
            "num_headings": num_headings,
            "avg_font_size": avg_font_size,
            "layout_preserved": self.preserve_layout,
        }

        return formatted_text, metadata
