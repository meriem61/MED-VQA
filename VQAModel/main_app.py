import gradio as gr
from PIL import Image
import sys 
import pickle
sys.path.insert(1, "ClassificationModel")
from torchvision import transforms
from  FeatureExtractor import extract_image_features_inst, extract_text_features
import torch
import torch.nn as nn
from torchvision import models
from model import VQAClassifier 


mymodel = VQAClassifier(num_classes=1742,dropout=0.2,num_heads=8)
mymodel.eval()
state_dict = torch.load("model_imgClef_8h_0.2d.pth", map_location=torch.device('cpu'))
mymodel.load_state_dict(state_dict)

# Set the model to evaluation mode
mymodel.eval()
import gradio as gr

theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="blue",
    neutral_hue="zinc",
).set(
    background_fill_primary='*neutral_100',
    background_fill_primary_dark='*neutral_900',
    background_fill_secondary='*neutral_200',
    border_color_accent='*secondary_500',
    border_color_accent_dark='*secondary_500',
    border_color_accent_subdued_dark='*secondary_500',
    border_color_primary='*neutral_400',
    border_color_primary_dark='*primary_950',
    color_accent_soft='*neutral_200',
    block_label_background_fill='*primary_200',
    block_label_border_color='*primary_300',
    block_label_radius='*radius_lg',
    block_title_text_color='*secondary_950',
    block_title_text_color_dark='*neutral_50',
    button_large_radius='*radius_xl',
    button_small_radius='*radius_xl',
    button_primary_background_fill='*secondary_400',
    button_primary_background_fill_dark='*secondary_800',
    button_primary_background_fill_hover='*secondary_500',
    button_secondary_background_fill='*secondary_400',
    button_secondary_background_fill_dark='*secondary_800',
    button_secondary_background_fill_hover='*secondary_500',
    button_secondary_background_fill_hover_dark='*secondary_400'
)

def numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),         
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])
def process_image(image):
    image = preprocess(image).unsqueeze(0)
    image = extract_image_features_inst(image)
    return image.unsqueeze(0)
def process_question(question):
    question = extract_text_features(question)
    return question.unsqueeze(0)
def make_prediction(image, question):
    image = process_image(image)
    question = process_question(question)
    output = mymodel(image, question)
    _, predicted = torch.max(output, 1)
    print(predicted)
    with open("label_to_answer_mapping.pkl", 'rb') as f:
        label_to_answers = pickle.load(f)
    predicted = label_to_answers[predicted.item()]
    return predicted


def predict(image, question):
    if image is None:
        gr.Warning("Please upload an image.")
        return
    elif question is None or question == "":
        gr.Warning("Please what's your question?")
        return
    if (image is None) and (question is None or question == ""):
        gr.Warning("Please upload an image and enter a question.")
        return
    elif numeric(question) is True:
        gr.Warning("Please enter a valid question.")
        return
    else:
        gr.Info("Processing... Please Note that this may take a while. Thank you for your patience.")
        answer = make_prediction(image, question)
        return answer
blocks = gr.Blocks(theme=theme,
                   css="""
        .header {
            text-align: center;
            padding: 0px;
            background-color: #f2f2f2;
        }
        .header img {
            max-width: 100%;
            max-height: 100%;
            margin-bottom: 0px;
        }
    """, title="MED-VQA")
with blocks as demo:
    gr.HTML("""<div class='header'><img src='http://localhost:8000/header.png' alt='Header Image'>     
            </div>""")
    with gr.Row():
        image = gr.Image(type="pil", label="Upload Image")
        question = gr.Textbox(lines=2, label="Question")
    answer = gr.Textbox(label="Answer")
    with gr.Row():
        submit_button = gr.Button("Submit")
        details_button = gr.Button("Clear")
    submit_button.click(fn=predict, inputs=[image, question], outputs=[answer])


# Launch the interface
demo.launch()

