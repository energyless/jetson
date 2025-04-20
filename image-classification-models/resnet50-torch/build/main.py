import base64
import json
import sys
import os
from io import BytesIO
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

# Load the EfficientNetV2 model globally with pre-trained ImageNet weights.
# Load the EfficientNetV2 model globally with pre-trained ImageNet weights.
if os.environ["USE_GPU"] == "true":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}", file=sys.stderr)

weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights).to(device)
model.eval()

# Use the default preprocessing transforms provided by the weights metadata.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def main(args):
    try:
        if "image" not in args:
            raise ValueError("Missing 'image' key in input.")

        img_data = base64.b64decode(args["image"])
        img = Image.open(BytesIO(img_data)).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)

        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_catid = torch.topk(probs, 3)

        with open("/action/imagenet_classes.txt") as f:
            classes = [line.strip() for line in f]

        predictions = [
            {
                "class_id": str(top3_catid[i].item()),
                "label": classes[top3_catid[i].item()],
                "score": top3_prob[i].item()
            }
            for i in range(top3_prob.size(0))
        ]

        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        image_data_files = os.listdir("/images")
        total = len(image_data_files)
        for i, image_data_file in enumerate(image_data_files):
            if i%100 == 0:
                print(f"done: {i+1}/{total}")
            fullpath = f"/images/{image_data_file}"
            data_file = open(fullpath, "r")
            output = main(json.load(data_file))
            f = open(f"/results/{image_data_file}", "w")
            json.dump(output, f)
            f.close()


    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
