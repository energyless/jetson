import base64
import json
import sys
import os
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input



tf.keras.utils.disable_interactive_logging()

USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
device_name = "/GPU:0" if USE_GPU and tf.config.list_physical_devices('GPU') else "/CPU:0"

print("using device: ", device_name)

tf.device(device_name)
tf.debugging.set_log_device_placement(True)
model = ResNet50(weights='imagenet')

# ✅ Load model once globally

def preprocess_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")

    img = img.resize((224, 224))  # ResNet50 expects 224x224 input
    img_array = np.array(img)

    # Expand to batch and apply preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # This handles mean/std normalization
    return img_array


def main(args):
    try:
        if "image" not in args:
            raise ValueError("Missing 'image' key in input.")

        input_tensor = preprocess_image(args["image"])
        predictions = model.predict(input_tensor)

        decoded = decode_predictions(predictions, top=3)[0]  # List of tuples (class, description, score)

        result = [
            {
                "class_id": str(i),
                "label": label,
                "score": float(score)
            }
            for i, (imagenet_id, label, score) in enumerate(decoded)
        ]

        return {"predictions": result}

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
