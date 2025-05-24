import base64
import json
import sys
import os
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2B0,
    preprocess_input,
    decode_predictions
)


# tf.keras.utils.disable_interactive_logging()

print("devices: ")
print(tf.config.list_physical_devices('GPU'))

USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
device_name = "/GPU:0" if USE_GPU and tf.config.list_physical_devices('GPU') else "/CPU:0"

tf.device(device_name)
tf.debugging.set_log_device_placement(True)

print("using device: ", device_name)

# ✅ Load model once globally
try:
    model = EfficientNetV2B0(weights='imagenet')
    model.trainable = False
except Exception as e:
    print(json.dumps({"error": f"Model load failed: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

def preprocess_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")

    img = img.resize((224, 224))  # EfficientNetV2B0 expects 224x224 input
    img_array = np.array(img).astype('float16')

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def main(args):
    try:
        if "image" not in args:
            raise ValueError("Missing 'image' key in input.")

        input_tensor = preprocess_image(args["image"])
        predictions = model.predict(input_tensor)

        decoded = decode_predictions(predictions, top=3)[0]

        result = [
            {
                "class_id": class_id,
                "label": label,
                "score": float(score)
            }
            for (class_id, label, score) in decoded
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
