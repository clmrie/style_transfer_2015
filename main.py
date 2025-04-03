import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests
from io import BytesIO
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128 


content_img_path = "/content.png"  
style_img_path = "/style.png"   

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

# loading images
def image_loader(image_source):
    if image_source.startswith("http://") or image_source.startswith("https://"):
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_source).convert('RGB')

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

content_img = image_loader(content_img_path)
style_img = image_loader(style_img_path)

assert content_img.size() == style_img.size(), "Images must be the same size"

unloader = transforms.ToPILImage() 


# We use pretrained VGG19 as a model
from torchvision.models import VGG19_Weights
weights = VGG19_Weights.DEFAULT
preprocess = weights.transforms()

# mean and stf values used for training of VGG19 on ImageNet dataset
cnn_normalization_mean = torch.tensor(preprocess.mean).to(device)
cnn_normalization_std = torch.tensor(preprocess.std).to(device)

# normalizing
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std  = std.clone().detach().view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

# --- Loss Modules ---
def gram_matrix(input):
    b, c, h, w = input.size()
    # flattening 
    features = input.view(c, h * w)
    # computing Gram matrix 
    G = torch.mm(features, features.t())
    # normalizing by number of elements
    return G.div(c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # We detach target content from graph
        self.target = target.detach()
        self.loss = 0
    def forward(self, input):
        # computing MSE between target and generated features
        self.loss = nn.MSELoss()(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0
    def forward(self, input):
        # gram matrix of generate image of same layer
        G = gram_matrix(input)
        # difference in style
        self.loss = nn.MSELoss()(G, self.target)
        return input

# --- Build the Model with Loss Layers ---
# the paper uses specifically these layers to compute content and style losses
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# making sure input has same distribution as the data VGG was trained on 
model = nn.Sequential(Normalization(cnn_normalization_mean, cnn_normalization_std).to(device))

content_losses = []
style_losses = []

i = 0 
# going through all layers and giving them a consistent name 
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f"conv_{i}"
    elif isinstance(layer, nn.ReLU):
        name = f"relu_{i}"
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = f"pool_{i}"
    elif isinstance(layer, nn.BatchNorm2d):
        name = f"bn_{i}"
    else:
        raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

    model.add_module(name, layer)

    # inserting modules into custom  model 
    if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)

    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

# we remove all layers after the last style/content loss layer
for j in range(len(model) - 1, -1, -1):
    if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
        break
model = model[:(j + 1)]

input_img = content_img.clone()

input_img.requires_grad_(True)
optimizer = optim.LBFGS([input_img])

num_steps = 300
style_weight = 1e6
content_weight = 1

print("Optimizing...")
run = [0]
while run[0] <= num_steps:
    def closure():
        # Clamp to maintain valid image range
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        # implementing loss from the paper
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()

        if run[0] % 50 == 0:
            print(f"Step {run[0]}: Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}")

        run[0] += 1
        return loss

    optimizer.step(closure)

input_img.data.clamp_(0, 1)

final_img = input_img.cpu().clone().squeeze(0)
final_img = unloader(final_img)
final_img.save("output.png")
print("Style transfer complete. Output saved as output.png")
