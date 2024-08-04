import streamlit as st
import io
import os
import tempfile
import google.generativeai as genai
from api_key import api_key

# Configure the API key
genai.configure(api_key=api_key)

# Define the generation configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # You can adjust safety settings here if needed
)

# Streamlit app configuration
st.set_page_config(page_title="Image to Text Converter", page_icon=':robot:', layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: #2D2D2D;
        margin: 0;
    }
    .container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 1.5em;
        color: #606060;
        text-align: center;
    }
    .header {
        text-align: center;
        padding: 20px;
        background-color: #F4F4F4;
        border-radius: 10px;
    }
    .description {
        text-align: center;
        font-size: 1.2em;
        color: #404040;
        padding: 20px;
    }
    .image-container img {
        width: 150px; /* Set the width of the logo image */
    }
    </style>
""", unsafe_allow_html=True)

# Header with logo and title
st.markdown("""
    <div class="header">
        <div class="container">
            <div class="title">Image to Text Converter</div>
          
        </div>
    </div>
""", unsafe_allow_html=True)

# Main Subtitle
st.markdown('<div class="subtitle">An application that can extract textual data from the image</div>', unsafe_allow_html=True)

# Add a description for more context
st.markdown('<div class="description">Upload an image containing text, and this application will extract and display the textual data for you. Simply use the uploader below to start the process.</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])

# Process the image when uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Display the uploaded image
    st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)
    st.write("Processing your image...")

    # Upload the temporary file to Gemini
    file = genai.upload_file(temp_file_path, mime_type="image/jpeg")

    # Define the system prompt
    system_prompt = "Convert this image to text."

    # Start the chat session
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    file,
                    system_prompt,
                ],
            },
        ]
    )

    # Send message and get response
    response = chat_session.send_message("Please analyze the image and give me text out from it.")

    # Display the response
    st.write("Model Response:")
    st.write(response.text)

    # Clean up the temporary file
    os.remove(temp_file_path)
