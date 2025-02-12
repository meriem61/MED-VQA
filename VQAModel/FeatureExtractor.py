import torch
from torchvision import models
import torch
import torchvision.models as models
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms


#device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# load ViT extractor weights
pretrained_vit_weights = models.ViT_B_16_Weights.DEFAULT
pretrained_vit = models.vit_b_16(weights=pretrained_vit_weights).to(device)
# feature extraction using ViT
def image_features_extraction(model, input_tensor):
    cls_token_output = None

    def hook(module, input, output):
        nonlocal cls_token_output
        cls_token_output = output[:, 0] 

    handle = model.encoder.layers[-1].register_forward_hook(hook)
    with torch.no_grad():
        model(input_tensor)
    handle.remove()
    return cls_token_output
#   BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
text_encoder = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)

for p in text_encoder.parameters():
    p.requires_grad = False
# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def preprocess_image(img_id,type):
    if type == "train":
        img_path = "../datasets/clef2019/train/Train_images/" + img_id + ".jpg"
    elif type == "test":
        img_path = "../datasets/clef2019/test/Test_images/" + img_id + ".jpg"
    else:
        img_path = "../datasets/clef2019/valid/Val_images/" + img_id + ".jpg"
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)
# extraction image features (Preprocessing + ViT extractor)
def extract_image_features(img_id,type):
    pixel_values = preprocess_image(img_id,type).to(device)
    outputs = image_features_extraction(pretrained_vit,pixel_values)
    outputs = outputs.squeeze(0)
    return outputs
# Extract image features for an input image tensor (Fot the GUI)
def extract_image_features_inst(img):
    outputs = image_features_extraction(pretrained_vit,img)
    outputs = outputs.squeeze(0)
    return outputs
# Extract question features
def extract_text_features(text):
    text_inputs = tokenizer(text, return_tensors="pt").to(device)
    text_inputs = {k:v for k,v in text_inputs.items()}
    text_outputs = text_encoder(**text_inputs)
    text_embedding = text_outputs.pooler_output 
    text_embedding = text_embedding.detach()
    text_embedding = text_embedding.squeeze(0)
    return text_embedding