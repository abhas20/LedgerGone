"""LedgerGone - Local WhatsApp Khata & Receipt Auditor for Kirana/Small Shops.

A local-first, privacy-focused audit dashboard powered by Multimodal Gemma 3 and TigerData (TimescaleDB + pgvector).
"""
import os
import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime

from src.config import DATABASE_URL, DEFAULT_GEMMA_MODEL, OLLAMA_BASE_URL
from src.db import (
    init_db,
    insert_ledger_entry,
    get_ledger_entries,
    get_summary_metrics,
    get_time_series_cashflow,
    semantic_search_items,
    get_vendor_statement
)
from src.embeddings import generate_embedding, generate_embeddings
from src.gemma_extractor import extract_with_gemma_multimodal, ExtractedLedger, ExtractedItem
from src.generate_bill import (
    generate_whatsapp_message,
    get_whatsapp_share_url,
    generate_pdf_statement
)
from sample_data.presets import SAMPLE_PRESETS, seed_demo_entries

# Page Configuration
st.set_page_config(
    page_title="LedgerGone | Kirana Khata Auditor",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database on Startup
try:
    init_db()
    db_connected = True
except Exception as e:
    db_connected = False
    db_error = str(e)

# Sidebar
with st.sidebar:
    st.title("🏪 LedgerGone")
    st.markdown("**Local WhatsApp Khata & Receipt Auditor**")
    st.caption("AI belongs to everyone • Hacktoberfest 2026")
    
    st.markdown("---")
    st.subheader("⚙️ System Status")
    
    if db_connected:
        st.success("🟢 **TigerData TimescaleDB**: Connected")
        st.caption("Hypertables & pgvector enabled")
    else:
        st.error(f"🔴 **DB Connection Error**: {db_error}")

    st.info(f"🧠 **Model**: Multimodal Gemma 3 (`{DEFAULT_GEMMA_MODEL}`)")
    st.caption("🔒 100% Local Inference • CPU Optimized")

    st.markdown("---")
    st.subheader("🏪 Store Profile")
    store_name = st.text_input("Store Name", value="Shree Ganesh Kirana & Hardware")
    owner_phone = st.text_input("Owner WhatsApp Number", value="919876543210")
    
    st.markdown("---")
    st.subheader("⚡ Quick Demonstration")
    if st.button("🌱 Seed Sample Kirana Ledger", use_container_width=True):
        with st.spinner("Seeding time-series entries into TigerData..."):
            count = seed_demo_entries()
            st.success(f"Added {count} realistic Kirana transactions!")
            st.rerun()

# Main Header
st.title("🏪 LedgerGone — Local Kirana Khata & Receipt Auditor")
st.markdown(
    "*Audits messy handwritten supplier parchis, GST bills, UPI payment screenshots, and WhatsApp voice notes locally.* "
    "**Zero tax data leaks to third-party clouds.**"
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Drop & Audit (Vision & Voice)",
    "📊 Live Khata & Audit Ledger",
    "🔍 Semantic Item Search (pgvector)",
    "🧾 WhatsApp Bill Generator",
    "🧪 Sample Kirana Data"
])

# -----------------------------------------------------------------------------
# TAB 1: DROP & AUDIT
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("📥 Audit Bills, Handwritten Parchis & WhatsApp Voice Notes")
    st.markdown(
        "Drop receipt images or paste voice notes. **Multimodal Gemma 3** extracts supplier details, line items, and pending balances."
    )

    col_input_type, col_model = st.columns([2, 1])
    with col_input_type:
        input_mode = st.radio(
            "Select Audit Input Source:",
            ["📷 Receipt / Parchi Image (Multimodal)", "🎙️ WhatsApp Voice Note Transcript"],
            horizontal=True
        )
    with col_model:
        selected_model = st.text_input("Gemma 3 Tag in Ollama", value=DEFAULT_GEMMA_MODEL)

    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None

    if input_mode == "📷 Receipt / Parchi Image (Multimodal)":
        col_img_up, col_img_prev = st.columns([1, 1])
        with col_img_up:
            uploaded_file = st.file_uploader(
                "Upload Supplier Bill, Handwritten Parchi, or UPI Screenshot",
                type=["png", "jpg", "jpeg", "webp"]
            )
            
            # Preset selector
            st.markdown("**Or pick a pre-loaded sample receipt:**")
            sample_choice = st.selectbox(
                "Choose sample receipt:",
                ["None", "sample_data/receipts/ramesh_hardware_bill.png", "sample_data/receipts/gupta_kirana_parchi.png"]
            )
            
            target_image = None
            if uploaded_file is not None:
                target_image = Image.open(uploaded_file)
            elif sample_choice != "None" and os.path.exists(sample_choice):
                target_image = Image.open(sample_choice)

            run_btn = st.button("🚀 Audit with Gemma 3 Multimodal", type="primary", use_container_width=True)

        with col_img_prev:
            if target_image:
                st.image(target_image, caption="Input Receipt Preview", use_container_width=True)
            else:
                st.info("Upload an image or pick a sample from the left.")

        if run_btn and target_image:
            with st.spinner("Gemma 3 analyzing image end-to-end (OCR + accounting extraction)..."):
                result = extract_with_gemma_multimodal(image=target_image, model_name=selected_model)
                st.session_state.extracted_data = result
                st.success("Audit complete!")

    else:
        # WhatsApp Voice Note Mode
        st.markdown("**Enter or paste WhatsApp voice note message / transcript:**")
        voice_prompt = st.text_area(
            "Voice Note Text (Hindi / Hinglish / English)",
            value="Ramesh hardware sent 5000 rupees stock today for cement and binding wire, paid 2000 via UPI on the spot, remaining 3000 rupees pending in udhaar khata.",
            height=120
        )
        if st.button("🎙️ Audit Voice Note with Gemma 3", type="primary"):
            with st.spinner("Gemma 3 extracting financial entities from voice note..."):
                result = extract_with_gemma_multimodal(text_prompt=voice_prompt, model_name=selected_model)
                st.session_state.extracted_data = result
                st.success("Voice note audited!")

    # Display Extracted Result & Human Verification
    if st.session_state.extracted_data:
        data = st.session_state.extracted_data
        st.markdown("---")
        st.subheader("📋 Audited Khata Verification & Review")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            vendor = st.text_input("Party / Vendor Name", value=data.vendor_name)
            inv_no = st.text_input("Invoice / Parchi Ref", value=data.invoice_no or "PARCHI-NEW")
        with col2:
            entry_type = st.selectbox("Transaction Type", ["PURCHASE", "SALE", "PAYMENT", "CREDIT_KHATA"], index=0)
            pay_mode = st.selectbox("Payment Mode", ["UPI", "CASH", "CREDIT", "CHEQUE"], index=["UPI", "CASH", "CREDIT", "CHEQUE"].index(data.payment_mode if data.payment_mode in ["UPI", "CASH", "CREDIT", "CHEQUE"] else "CASH"))
        with col3:
            total_amt = st.number_input("Total Amount (₹)", value=float(data.total_amount), step=100.0)
            paid_amt = st.number_input("Paid / Jama (₹)", value=float(data.paid_amount), step=100.0)
        with col4:
            pending_bal = st.number_input("Pending Udhaar / Baaki (₹)", value=float(data.pending_balance), step=100.0)
            conf = st.progress(data.confidence_score, text=f"AI Confidence: {int(data.confidence_score*100)}%")

        # Line items review
        st.markdown("**Itemized Inventory Items:**")
        items_list = [it.dict() for it in data.line_items]
        df_items = pd.DataFrame(items_list) if items_list else pd.DataFrame(columns=["item_name", "quantity", "unit_price", "total_price"])
        edited_items = st.data_editor(df_items, num_rows="dynamic", use_container_width=True)

        notes_input = st.text_input("Audit Notes / Remarks", value=data.notes or "")

        if st.button("💾 Commit to TigerData Ledger", type="primary", use_container_width=True):
            with st.spinner("Storing time-series entry & generating pgvector item embeddings..."):
                entry_dict = {
                    "created_at": datetime.now(),
                    "vendor_name": vendor,
                    "entry_type": entry_type,
                    "invoice_no": inv_no,
                    "subtotal": total_amt,
                    "gst_amount": data.gst_amount,
                    "total_amount": total_amt,
                    "paid_amount": paid_amt,
                    "pending_balance": pending_bal,
                    "payment_mode": pay_mode,
                    "source_type": "RECEIPT_AUDIT",
                    "notes": notes_input,
                    "raw_text": f"Vendor: {vendor}, Items: {len(edited_items)}"
                }
                
                # Generate embeddings for all line items
                items_to_save = edited_items.to_dict(orient="records")
                item_names = [str(r.get("item_name", "")) for r in items_to_save]
                embeddings = generate_embeddings(item_names)
                
                new_id = insert_ledger_entry(entry_dict, items_to_save, embeddings)
                st.success(f"🎉 Successfully committed Entry #{new_id} to TigerData TimescaleDB with pgvector embeddings!")
                st.session_state.extracted_data = None
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 2: LIVE KHATA & AUDIT LEDGER
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 TimescaleDB Time-Series Khata Ledger")
    
    metrics = get_summary_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Billed Turnover", f"₹{metrics.get('total_turnover', 0):,.2f}")
    c2.metric("Total Paid", f"₹{metrics.get('total_paid', 0):,.2f}")
    c3.metric("🔴 Total Pending (Udhaar)", f"₹{metrics.get('total_pending_udhaar', 0):,.2f}")
    c4.metric("Audited Entries", f"{metrics.get('total_entries', 0)}")

    st.markdown("---")
    st.subheader("📈 Time-Series Cashflow Breakdown (Timescale time_bucket)")
    cashflow_data = get_time_series_cashflow(days=30)
    if cashflow_data:
        df_cf = pd.DataFrame(cashflow_data)
        df_cf["Day"] = pd.to_datetime(df_cf["bucket_day"]).dt.strftime("%d-%b")
        df_cf["Total Billed"] = df_cf["daily_total"].astype(float)
        df_cf["Paid Amount"] = df_cf["daily_paid"].astype(float)
        df_cf["Pending Udhaar"] = df_cf["daily_udhaar"].astype(float)
        st.bar_chart(df_cf.set_index("Day")[["Total Billed", "Paid Amount", "Pending Udhaar"]])
    else:
        st.info("No time-series entries recorded yet. Click 'Seed Sample Kirana Ledger' in the sidebar to populate.")

    st.markdown("---")
    st.subheader("📑 All Ledger Entries")
    filter_col, _ = st.columns([1, 2])
    with filter_col:
        vendor_search = st.text_input("Filter by Vendor/Party Name", placeholder="e.g. Ramesh Hardware")

    entries = get_ledger_entries(vendor_filter=vendor_search, limit=50)
    if entries:
        df_entries = pd.DataFrame(entries)
        display_cols = ["id", "created_at", "vendor_name", "entry_type", "invoice_no", "total_amount", "paid_amount", "pending_balance", "payment_mode", "notes"]
        st.dataframe(df_entries[display_cols], use_container_width=True)
    else:
        st.info("No ledger entries found matching criteria.")

# -----------------------------------------------------------------------------
# TAB 3: SEMANTIC SEARCH (pgvector)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🔍 Semantic Inventory & Historical Bill Search (pgvector)")
    st.markdown(
        "Search your store's historical line items and supplier bills using natural language. "
        "Powered by **TigerData pgvector** and local ONNX embeddings."
    )

    search_col, limit_col = st.columns([3, 1])
    with search_col:
        query = st.text_input("Search Line Items & Stock:", placeholder="e.g. 'cement bags', 'cooking oil', 'soap', 'hardware wire'")
    with limit_col:
        top_k = st.slider("Max Results", 3, 20, 5)

    if query:
        with st.spinner("Running pgvector cosine similarity search..."):
            q_emb = generate_embedding(query)
            matches = semantic_search_items(q_emb, limit=top_k)

        if matches:
            st.success(f"Found {len(matches)} matching line items in historical bills:")
            for m in matches:
                sim = float(m.get("similarity_score", 0.0))
                with st.container():
                    col_res1, col_res2, col_res3 = st.columns([2, 1, 1])
                    with col_res1:
                        st.markdown(f"**📦 {m['item_name']}**")
                        st.caption(f"Supplier: **{m['vendor_name']}** • Invoice: {m['invoice_no']} • Date: {str(m['created_at'])[:10]}")
                    with col_res2:
                        st.markdown(f"**Qty:** {m['quantity']} | **Rate:** ₹{m['unit_price']:,.2f}")
                        st.caption(f"Total: ₹{m['total_price']:,.2f} ({m['payment_mode']})")
                    with col_res3:
                        st.metric("Vector Match", f"{int(sim * 100)}%")
                    st.divider()
        else:
            st.warning("No matching items found. Try another query or seed sample data.")

# -----------------------------------------------------------------------------
# TAB 4: WHATSAPP BILL GENERATOR
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🧾 WhatsApp Khata Bill & Statement Generator")
    st.markdown(
        "Generate a structured, polite WhatsApp reminder or printable PDF statement for any vendor or customer."
    )

    all_entries = get_ledger_entries(limit=100)
    vendor_names = sorted(list(set(e["vendor_name"] for e in all_entries if e.get("vendor_name"))))

    if vendor_names:
        sel_vendor = st.selectbox("Select Party / Vendor:", vendor_names)
        cust_phone = st.text_input("Party WhatsApp Mobile Number (with country code):", value="919876543210")

        stmt = get_vendor_statement(sel_vendor)
        msg_text = generate_whatsapp_message(
            vendor_name=sel_vendor,
            totals=stmt["totals"],
            entries=stmt["entries"],
            store_name=store_name
        )

        col_msg, col_actions = st.columns([3, 2])
        with col_msg:
            st.markdown("**Message Preview (Ready for WhatsApp):**")
            st.text_area("WhatsApp Text", value=msg_text, height=260)

        with col_actions:
            st.markdown("**Actions:**")
            wa_url = get_whatsapp_share_url(cust_phone, msg_text)
            st.link_button("📲 Send via WhatsApp Web / App", wa_url, type="primary", use_container_width=True)

            pdf_bytes = generate_pdf_statement(
                vendor_name=sel_vendor,
                totals=stmt["totals"],
                entries=stmt["entries"],
                store_name=store_name
            )
            st.download_button(
                label="📄 Download Official PDF Khata Statement",
                data=pdf_bytes,
                file_name=f"Khata_Statement_{sel_vendor.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("No parties recorded in the ledger yet. Audit a receipt or seed demo entries to start.")

# -----------------------------------------------------------------------------
# TAB 5: SAMPLE KIRANA DATA
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🧪 Built-in Kirana Sample Data & Test Scenarios")
    st.markdown(
        "Test realistic scenarios small business owners face daily without needing physical documents."
    )

    for idx, preset in enumerate(SAMPLE_PRESETS):
        with st.expander(f"{preset['title']} — {preset['mock_ledger']['vendor_name']}"):
            st.markdown(f"**Scenario:** {preset['description']}")
            st.info(f"**Input Text:** \"{preset['text']}\"")
            st.json(preset["mock_ledger"])
