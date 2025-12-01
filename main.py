import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from src import utils, ocr_engine, preprocessing, postprocessing
import json

# Streamlit App
st.set_page_config(page_title="OCR Text Extractor", page_icon="📦", layout="wide")

st.title("📦 OCR Text Extractor")
st.markdown("Upload a shipping label image to extract barcode text with automatic underscore correction")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    # show_preprocessed = st.checkbox("Show preprocessed image", value=False)

    st.markdown("---")
    st.markdown("### 📋 Pattern Info")
    st.markdown("""
    **Expected Format:**
    - `XXXXXXX_X_XXX`
    - Example: `156381A26414724544_1_whl`

    **Fixes Applied:**
    - Replaces spaces with underscores
    - Corrects missing underscores
    - Validates pattern structure
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Read and display original image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

        # # Show preprocessed image if enabled
        # if show_preprocessed:
        #     height = image.shape[0]
        #     bottom_region = image[int(height * 0.75):, :]
        #     # cv2.imshow("Preprocessed Image", preprocess)
        #     st.image(preprocess, caption="Preprocessed Bottom Region", use_container_width=True)

with col2:
    st.subheader("🔍 OCR Results")

    if uploaded_file is not None and st.button("Search Pattern"):
        with st.spinner("Processing OCR..."):
            preprocess = preprocessing.preprocess(image)
            result = ocr_engine.predict(img = preprocess)
            output = postprocessing.patten_match(result)

            if output:
                st.success(f"✅ Found {len(output)} barcode text(s)")

                # Display target results with highlighting
                for idx, result in enumerate(output):
                    with st.container():
                        st.markdown(f"### 🎯 Target {idx + 1}")

                        col_a, col_b, col_c = st.columns([2, 2, 1])

                        with col_a:
                            st.markdown("**Original OCR:**")
                            st.code(result['original'])

                        with col_b:
                            st.markdown("**Fixed Text:**")
                            st.code(result['fixed'], language=None)

                        with col_c:
                            st.metric("Confidence", f"{result['confidence']:.2%}")

                        st.markdown("---")


                st.subheader("🖼️ Detection Visualization")
                img_draw = image.copy()
                # Extract fields from your result
                text = result["fixed"]
                confidence = result["confidence"]

                # Convert string bbox → numpy array
                bbox = result["bbox"]  # [x_min, y_min, x_max, y_max]

                # Extract coords
                x1, y1, x2, y2 = map(int, bbox)

                # Create 4-point polygon
                pts = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.int32)

                # Draw polygon
                cv2.polylines(img_draw, [pts], True, (0, 255, 0), 2)

                # Put text
                cv2.putText(img_draw, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)

                img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                col_vis1, col_vis2 = st.columns(2)

                with col_vis1:
                    st.image(img_rgb, caption="OCR Detection", use_container_width=True)

                with col_vis2:
                    if output:
                        st.markdown("### 📊 Summary")
                        st.metric("Total Detected", len(result))
                        st.metric("Target Matches", len(output))
                        st.metric("Avg Confidence", f"{np.mean([r['confidence'] for r in output]):.2%}")

                        # Download results
                        st.markdown("### 💾 Download Results")

                        # Prepare JSON download
                        download_data = {
                            'target_texts': [r['fixed'] for r in output],
                            'all_results': [
                                {
                                    'original': r['original'],
                                    'fixed': r['fixed'],
                                    'confidence': float(r['confidence'])
                                }
                                for r in output
                            ]
                        }

                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(download_data, indent=2),
                            file_name="barcode_extraction_results.json",
                            mime="application/json"
                        )
            else:
                st.warning("⚠️ No pattern found in the image")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit 🎈 | PaddleOCR 🔍 | OpenCV 📷</p>
</div>
""", unsafe_allow_html=True)