"""Bill & WhatsApp Khata Statement Generator for LedgerGone."""
import urllib.parse
from io import BytesIO
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_whatsapp_message(vendor_name: str, totals: Dict[str, Any], entries: List[Dict[str, Any]], store_name: str = "Shree Kirana & General Store") -> str:
    """Format an audited, polite WhatsApp Khata statement for Kirana suppliers or customers."""
    total_billed = float(totals.get("total_billed", 0.0) or 0.0)
    total_paid = float(totals.get("total_paid", 0.0) or 0.0)
    total_pending = float(totals.get("total_pending", 0.0) or 0.0)

    lines = [
        f"🙏 *NAMASTE FROM {store_name.upper()}* 🙏",
        f"📅 *Audited Khata Statement*",
        f"👤 *Party:* {vendor_name}",
        "────────────────────────",
        "📊 *Account Summary:*",
        f"• Total Billed: ₹{total_billed:,.2f}",
        f"• Total Paid (Jama): ₹{total_paid:,.2f}",
        f"• *Current Balance (Baaki/Udhaar):* *₹{total_pending:,.2f}*",
        "────────────────────────",
        "📝 *Recent Transactions:*"
    ]

    for idx, e in enumerate(entries[-5:], 1):
        dt = e.get("created_at")
        date_str = dt.strftime("%d-%b-%Y") if hasattr(dt, "strftime") else str(dt)[:10]
        inv = e.get("invoice_no") or "Bill"
        tot = float(e.get("total_amount", 0.0) or 0.0)
        bal = float(e.get("pending_balance", 0.0) or 0.0)
        lines.append(f"{idx}. {date_str} | {inv} | Total: ₹{tot:,.2f} (Bal: ₹{bal:,.2f})")

    lines.extend([
        "────────────────────────",
        "✨ *Audited with LedgerGone (Local-First Kirana AI)*",
        "Please settle via UPI or visit shop. Thank you for your continued partnership!"
    ])

    return "\n".join(lines)


def get_whatsapp_share_url(phone_number: str, message: str) -> str:
    """Generate a 1-click WhatsApp web/mobile click-to-chat URL."""
    clean_phone = "".join(filter(str.isdigit, phone_number))
    encoded_text = urllib.parse.quote(message)
    if clean_phone:
        return f"https://wa.me/{clean_phone}?text={encoded_text}"
    return f"https://wa.me/?text={encoded_text}"


def generate_pdf_statement(vendor_name: str, totals: Dict[str, Any], entries: List[Dict[str, Any]], store_name: str = "Shree Kirana & General Store") -> bytes:
    """Render a clean, printable PDF statement using ReportLab."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1b365d"),
        alignment=1  # Center
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555"),
        alignment=1
    )
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1b365d")
    )
    
    story = []
    
    # Header
    story.append(Paragraph(store_name, title_style))
    story.append(Paragraph("Official WhatsApp Khata & Audited Ledger Statement", subtitle_style))
    story.append(Spacer(1, 15))
    
    # Party details table
    party_data = [
        [Paragraph(f"<b>Party Name:</b> {vendor_name}", styles['Normal']),
         Paragraph(f"<b>Total Billed:</b> ₹{float(totals.get('total_billed', 0.0)):,.2f}", styles['Normal'])],
        [Paragraph(f"<b>Statement Type:</b> Vendor / Supplier Khata", styles['Normal']),
         Paragraph(f"<b>Total Paid:</b> ₹{float(totals.get('total_paid', 0.0)):,.2f}", styles['Normal'])],
        [Paragraph(f"<b>Generated Via:</b> LedgerGone Local AI", styles['Normal']),
         Paragraph(f"<b><font color='red'>Pending Balance: ₹{float(totals.get('total_pending', 0.0)):,.2f}</font></b>", styles['Normal'])],
    ]
    t_party = Table(party_data, colWidths=[270, 270])
    t_party.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_party)
    story.append(Spacer(1, 15))
    
    # Transactions table
    story.append(Paragraph("Transaction History", section_style))
    story.append(Spacer(1, 6))
    
    table_rows = [["Date", "Ref / Invoice", "Type", "Mode", "Total (₹)", "Paid (₹)", "Balance (₹)"]]
    for e in entries:
        dt = e.get("created_at")
        date_str = dt.strftime("%d-%m-%Y") if hasattr(dt, "strftime") else str(dt)[:10]
        table_rows.append([
            date_str,
            e.get("invoice_no") or "-",
            e.get("entry_type", "PURCHASE"),
            e.get("payment_mode", "CASH"),
            f"{float(e.get('total_amount', 0.0)):,.2f}",
            f"{float(e.get('paid_amount', 0.0)):,.2f}",
            f"{float(e.get('pending_balance', 0.0)):,.2f}"
        ])
    
    t_entries = Table(table_rows, colWidths=[70, 95, 75, 60, 80, 80, 80])
    t_entries.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (4,1), (-1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t_entries)
    
    # Footer note
    story.append(Spacer(1, 25))
    story.append(Paragraph("<i>This is a computer-audited statement generated locally with LedgerGone. No financial data was uploaded to public clouds.</i>", subtitle_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
