"""Sample Kirana Presets and Demonstration Data for LedgerGone.

Located in /sample_data outside src for clean modularity.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List
from src.embeddings import generate_embeddings
from src.db import insert_ledger_entry

SAMPLE_PRESETS = [
    {
        "title": "🎙️ WhatsApp Voice Note (Ramesh Hardware)",
        "source_type": "VOICE_NOTE",
        "description": "Kirana owner speaking after delivery truck arrives.",
        "text": "Ramesh hardware sent 5000 rupees stock today for cement and binding wire, paid 2000 via UPI on the spot, remaining 3000 rupees pending in udhaar khata.",
        "mock_ledger": {
            "vendor_name": "Ramesh Hardware & Tools",
            "entry_type": "PURCHASE",
            "invoice_no": "RH-VOICE-801",
            "line_items": [
                {"item_name": "ACC Grade 53 Cement Bags", "quantity": 10.0, "unit_price": 380.0, "total_price": 3800.0},
                {"item_name": "Galvanized Binding Wire Roll", "quantity": 2.0, "unit_price": 600.0, "total_price": 1200.0}
            ],
            "subtotal": 4237.28,
            "gst_amount": 762.72,
            "total_amount": 5000.0,
            "paid_amount": 2000.0,
            "pending_balance": 3000.0,
            "payment_mode": "UPI",
            "notes": "Voice note logged by shopkeeper. Delivery received at 10 AM.",
            "confidence_score": 0.98
        }
    },
    {
        "title": "📝 Handwritten Parchi (Gupta Provisions)",
        "source_type": "HANDWRITTEN_PARCHI",
        "description": "Daily loose grocery paper slip delivered with wholesale sacks.",
        "text": "Gupta Provisions Parchi: 10kg Aashirvaad Chakki Atta (4200), 5L Fortune Mustard Oil (750), 20 Pcs Lifebuoy Soap (700). Total: 5650. Jama Cash: 2000. Baaki (Udhaar): 3650.",
        "mock_ledger": {
            "vendor_name": "Gupta Wholesale Provisions",
            "entry_type": "PURCHASE",
            "invoice_no": "PARCHI-104",
            "line_items": [
                {"item_name": "Aashirvaad Chakki Fresh Atta 10kg", "quantity": 10.0, "unit_price": 420.0, "total_price": 4200.0},
                {"item_name": "Fortune Kachi Ghani Mustard Oil 1L", "quantity": 5.0, "unit_price": 150.0, "total_price": 750.0},
                {"item_name": "Lifebuoy Total Germ Protection Soap", "quantity": 20.0, "unit_price": 35.0, "total_price": 700.0}
            ],
            "subtotal": 5650.0,
            "gst_amount": 0.0,
            "total_amount": 5650.0,
            "paid_amount": 2000.0,
            "pending_balance": 3650.0,
            "payment_mode": "CASH",
            "notes": "Handwritten paper slip verified. Baaki ₹3,650 added to Gupta ji ledger.",
            "confidence_score": 0.94
        }
    },
    {
        "title": "📱 PhonePe / GPay UPI Screenshot (Krishna Dairy)",
        "source_type": "UPI_SCREENSHOT",
        "description": "UPI screenshot from supplier account settlement.",
        "text": "Paid to Krishna Dairy & Ice Cream Supplies: ₹1,850. Transaction ID: T261018112233. Debited from SBI. Status: SUCCESSFUL.",
        "mock_ledger": {
            "vendor_name": "Krishna Dairy & Ice Cream Supplies",
            "entry_type": "PAYMENT",
            "invoice_no": "UPI-TXN-112233",
            "line_items": [
                {"item_name": "Amul Taaza Milk Crates Settlement", "quantity": 1.0, "unit_price": 1850.0, "total_price": 1850.0}
            ],
            "subtotal": 1850.0,
            "gst_amount": 0.0,
            "total_amount": 1850.0,
            "paid_amount": 1850.0,
            "pending_balance": 0.0,
            "payment_mode": "UPI",
            "notes": "Direct UPI transfer verified via PhonePe screenshot.",
            "confidence_score": 0.99
        }
    },
    {
        "title": "📦 Supplier GST Tax Invoice (Om Sai Spices)",
        "source_type": "RECEIPT_IMAGE",
        "description": "Standard B2B tax invoice with GST breakdown.",
        "text": "Om Sai Spices Wholesalers (GSTIN: 07AAFCO1234M1Z2). Invoice #OSS-8921. 20 Packets MDH Deggi Mirch 500g @ 160 (3200), 10 Packets Everest Turmeric 500g @ 120 (1200). Subtotal: 4400, GST 5%: 220, Total: 4620. Paid: 1620, Pending: 3000.",
        "mock_ledger": {
            "vendor_name": "Om Sai Spices Wholesalers",
            "entry_type": "PURCHASE",
            "invoice_no": "OSS-8921",
            "line_items": [
                {"item_name": "MDH Deggi Mirch 500g Packets", "quantity": 20.0, "unit_price": 160.0, "total_price": 3200.0},
                {"item_name": "Everest Turmeric Powder 500g", "quantity": 10.0, "unit_price": 120.0, "total_price": 1200.0}
            ],
            "subtotal": 4400.0,
            "gst_amount": 220.0,
            "total_amount": 4620.0,
            "paid_amount": 1620.0,
            "pending_balance": 3000.0,
            "payment_mode": "CREDIT",
            "notes": "GST input tax credit (ITC) eligible: ₹220 recorded.",
            "confidence_score": 0.97
        }
    }
]


def seed_demo_entries() -> int:
    """Seed sample Kirana ledger entries into TigerData TimescaleDB with time offsets."""
    count = 0
    now = datetime.now()
    
    for idx, preset in enumerate(SAMPLE_PRESETS):
        mock = preset["mock_ledger"]
        days_ago = (len(SAMPLE_PRESETS) - idx) * 2
        entry_time = now - timedelta(days=days_ago, hours=idx * 3)

        entry_dict = {
            "created_at": entry_time,
            "vendor_name": mock["vendor_name"],
            "entry_type": mock["entry_type"],
            "invoice_no": mock["invoice_no"],
            "subtotal": mock["subtotal"],
            "gst_amount": mock["gst_amount"],
            "total_amount": mock["total_amount"],
            "paid_amount": mock["paid_amount"],
            "pending_balance": mock["pending_balance"],
            "payment_mode": mock["payment_mode"],
            "source_type": preset["source_type"],
            "notes": mock["notes"],
            "raw_text": preset["text"]
        }

        items = mock["line_items"]
        item_names = [it["item_name"] for it in items]
        embeddings = generate_embeddings(item_names)

        insert_ledger_entry(entry_dict, items, embeddings)
        count += 1

    return count
