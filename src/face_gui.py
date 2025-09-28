import gradio as gr
import os

from embedder import FaceEmbedder
from recogniser import FaceRecognizer

embedder = FaceEmbedder()
recognizer = FaceRecognizer()


def register_single_image(img_file):
    try:
        embedder.process_single_image(img_file.name)  # .name gives the full path
        return "✅ Face registered successfully from image!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def register_from_folder(folder_path):
    try:
        embedder.new_capture(folder_path)
        return "✅ Faces registered successfully from folder!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def recognize_from_image(img_or_folder, output_folder):
    try:
        if not output_folder:
            output_folder = os.path.join(os.path.dirname(img_or_folder), "recognized_output")
        os.makedirs(output_folder, exist_ok=True)

        recognizer.process_images(img_or_folder, output_folder)
        return f"✅ Recognition completed. Output saved to: {output_folder}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def recognize_from_video(video_file, output_path=None):
    try:
        video_path = video_file.name  # Get actual path from UploadedFile object
        if not output_path:
            output_path = os.path.dirname(video_path)
        os.makedirs(output_path, exist_ok=True)

        output_video_path = recognizer.process_video(video_path, output_path)
        return output_video_path, f"✅ Recognition complete. Saved to {output_video_path}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


with gr.Blocks(title="Face Recognition System") as demo:
    gr.Markdown("## 🎭 Face Recognition System")

    with gr.Tab("1️⃣ Register Face"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖼️ Register from Single Image")
                single_image_input = gr.File(label="Choose Image", file_types=[".jpg", ".png", ".jpeg", ".webp"])
                reg_img_btn = gr.Button("Register")
                reg_img_output = gr.Textbox(label="Result")

            with gr.Column():
                gr.Markdown("### 📁 Register from Folder")
                folder_input = gr.Textbox(placeholder="Path to folder with images", label="Folder Path")  # Gradio lacks folder picker
                reg_folder_btn = gr.Button("Register")
                reg_folder_output = gr.Textbox(label="Result")

    with gr.Tab("2️⃣ Recognize Face"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖼️ From Image or Folder")
                image_input = gr.Textbox(placeholder="Path to image or folder", label="Input Image/Folder")
                output_folder_input = gr.Textbox(placeholder="Path to output folder", label="Output Folder")
                recog_img_btn = gr.Button("Recognize")
                recog_img_output = gr.Textbox(label="Result")

            with gr.Column():
                gr.Markdown("### 🎞️ From Video")
                video_input = gr.File(label="Select Video File", file_types=[".mp4", ".avi"])
                video_output = gr.Textbox(placeholder="Path to output folder", label="Output Folder")
                recog_vid_btn = gr.Button("Recognize")
                recog_vid_output = gr.Video(label="Output Video")



    # Event bindings
    reg_img_btn.click(fn=register_single_image, inputs=single_image_input, outputs=reg_img_output)
    reg_folder_btn.click(fn=register_from_folder, inputs=folder_input, outputs=reg_folder_output)
    recog_img_btn.click(fn=recognize_from_image, inputs=[image_input, output_folder_input], outputs=recog_img_output)
    recog_vid_output_status = gr.Textbox(label="Status")
    recog_vid_btn.click(fn=recognize_from_video, inputs=[video_input, video_output], outputs=[recog_vid_output, recog_vid_output_status])

demo.launch()
