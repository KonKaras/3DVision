from infer import InferenceHelper
from PIL import Image

infer_helper = InferenceHelper(dataset='nyu', custom=True)

# predict depth of a single pillow image
img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img, visualized=True)
