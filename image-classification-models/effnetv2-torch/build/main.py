import base64
import json
import sys
import os
from io import BytesIO
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import transforms
from PIL import Image

# Load the EfficientNetV2 model globally with pre-trained ImageNet weights.
# Load the EfficientNetV2 model globally with pre-trained ImageNet weights.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", file=sys.stderr)

weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights).to(device)
model.eval()

# Use the default preprocessing transforms provided by the weights metadata.
preprocess = weights.transforms()


def main(args):
    if "image" not in args:
        raise ValueError("Missing 'image' key in input.")

    # Decode the base64 image and open it using PIL.
    img_data = base64.b64decode(args["image"])
    img = Image.open(BytesIO(img_data)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    # Run inference without gradient tracking.
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get probabilities and the top 3 predictions.
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top3_prob, top3_catid = torch.topk(probs, 3)

    # Load ImageNet class names from a file (same as your original torch script).
    try:
        with open("/action/imagenet_classes.txt") as f:
            classes = [line.strip() for line in f]
    except Exception as e:
        raise RuntimeError(f"Error loading imagenet classes file: {str(e)}")

    predictions = [
        {
            "class_id": str(top3_catid[i].item()),
            "label": classes[top3_catid[i].item()],
            "score": top3_prob[i].item()
        }
        for i in range(top3_prob.size(0))
    ]
    return {"predictions": predictions}

if __name__ == "__main__":
    try:
        os.makedirs("/results")
        image_data_files = os.listdir("/images")
        for image_data_file in image_data_files:
            fullpath = f"/images/{image_data_file}"
            data_file = open(fullpath, "r")
            output = main(json.load(data_file))
            f = open(f"/results/{image_data_file}", "w")
            json.dump(output, f)


    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)