
# 🎨 Neural Style Transfer

*Deep Learning class project – Université Paris-Saclay* <br>
This is an implementation of the following paper:

**A Neural Algorithm of Artistic Style**  
*Leon A. Gatys, Alexander S. Ecker, Matthias Bethge*  
📄 [https://arxiv.org/abs/1508.06576](https://arxiv.org/abs/1508.06576)

---

This project applies **neural style transfer** to generate a new image that combines the **content of one image** with the **artistic style of another**. For example, you can take a photo and repaint it in the style of Van Gogh's *Starry Night*.

---

## 🧠 How It Works

The algorithm uses a **pretrained VGG19** convolutional neural network from PyTorch. It extracts features from:

- The **content image** to preserve its structure
- The **style image** to capture textures and patterns

Then, it optimizes a copy of the content image to match the style features, using **content and style loss functions** based on the CNN's feature maps.

---

## 🚀 How to Run

### 1. Setup

Install dependencies:
```bash
pip install torch torchvision pillow requests
```

### 2. Place Your Images

Save your input images in the same directory:
- `content.png` – the image you want to preserve
- `style.png` – the image you want to mimic stylistically

### 3. Run the Script

```bash
python style_transfer.py
```

The final output will be saved as:
```
output.png
```


