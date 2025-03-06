import streamlit as st
import os
import glob
from categorizer import ImageCategorizer  # <-- Import the class here

def main():
    st.title("Image Categorization App")

    # Instantiate the ImageCategorizer class
    cat = ImageCategorizer()

    # Let users upload multiple images
    uploaded_files = st.file_uploader(
        "Upload images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("temp_uploads", exist_ok=True)

        for uploaded_file in uploaded_files:
            temp_path = os.path.join("temp_uploads", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Categorize the image
            result = cat.categorize_screenshot(temp_path)

            st.subheader(f"Results for: {uploaded_file.name}")
            st.image(temp_path, caption=uploaded_file.name, use_column_width=True)
            st.write(f"**Predicted Domain:** {result['domain']} (Probability: {result['probability']:.2f})")
            st.write("**Top Predictions:**")
            for domain, prob in zip(result["top_domains"], result["probabilities"]):
                st.write(f"- {domain}: {prob:.2f}")

            # Save image if above threshold
            if result["domain"] != "Error" and result["probability"] >= cat.confidence_threshold:
                saved_path = cat.save_categorized_image(temp_path, result["domain"])
                if saved_path:
                    st.write(f"Image saved to: `{saved_path}`")
                else:
                    st.write("**Error saving image**.")
            else:
                st.write("Probability below threshold or error. Not saved.")

        # After all images are processed, generate analytics
        analytics_data = cat.generate_analytics()
        st.write("**Analytics:**", analytics_data)
        

        # Check if confusion matrix is generated
        confusion_files = glob.glob(os.path.join("analytics", "confusion_matrix_*.png"))
        if confusion_files:
            latest_confusion = max(confusion_files, key=os.path.getctime)
            st.subheader("Confusion Matrix")
            st.image(latest_confusion, caption="Confusion Matrix from User Feedback")
        else:
            st.write("No confusion matrix available yet. (Requires enough feedback data.)")

if __name__ == "__main__":
    main()
