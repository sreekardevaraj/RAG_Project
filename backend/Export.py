from fpdf import FPDF
import os
from datetime import datetime


class ChatExporter(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(79, 70, 229)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "RAG Chatbot V2 - Chat Export", align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")


def export_chat_to_pdf(messages: list, output_path: str) -> str:
    """
    Export chat history to a PDF file.
    messages: list of dicts with role, content, source, details
    output_path: where to save the PDF
    Returns the output path.
    """
    pdf = ChatExporter()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Subtitle
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Exported on {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        source  = msg.get("source", "")
        details = msg.get("details", [])

        if role == "user":
            # User bubble header
            pdf.set_fill_color(237, 233, 254)
            pdf.set_text_color(79, 70, 229)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 8, "  You", fill=True, new_x="LMARGIN", new_y="NEXT")

            # User message content
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 46)
            pdf.set_fill_color(245, 243, 255)
            safe_content = content.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 7, f"  {safe_content}", fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        elif role == "assistant":
            # Source badge
            source_label = ""
            if source == "PDF":
                pdf.set_fill_color(5, 150, 105)
                source_label = "  PDF Source"
            elif source == "Web":
                pdf.set_fill_color(37, 99, 235)
                source_label = "  Web Source"
            elif source == "Chat":
                pdf.set_fill_color(217, 119, 6)
                source_label = "  Chat"
            elif source == "Analyst":
                pdf.set_fill_color(124, 58, 237)
                source_label = "  Analyst"
            else:
                pdf.set_fill_color(100, 100, 100)
                source_label = "  Assistant"

            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(40, 7, source_label, fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

            # Assistant answer
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 46)
            pdf.set_fill_color(248, 250, 252)
            safe_content = content.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 7, f"  {safe_content}", fill=True, new_x="LMARGIN", new_y="NEXT")

            # References
            if details:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 6, "  References:", new_x="LMARGIN", new_y="NEXT")
                for item in details:
                    if source == "PDF":
                        ref = f"    - Page {item.get('page','?')} | {os.path.basename(str(item.get('source','unknown')))}"
                    else:
                        ref = f"    - {item.get('title','')} -> {item.get('url','')}"
                    safe_ref = ref.encode("latin-1", "replace").decode("latin-1")
                    pdf.multi_cell(0, 6, safe_ref, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(5)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    print(f"[Export] Chat exported to: {output_path}")
    return output_path