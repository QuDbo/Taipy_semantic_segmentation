from taipy.gui import Markdown
import time
from prediction_function.predmodel import *
import tensorflow as tf
import tensorflow.keras.backend as K
from io import BytesIO

image_originale = None
image_traitee = None
image_solution = None
ville=None
inference_time = 0.

model = load_model("./model/unet_vgg16_v3_256_2Aug")

def load_image(path_img):
    dimension = (256, 256)
    X = np.empty((1, *dimension, 3), dtype="uint8")
    image = imgtf.load_img(path_img,
    color_mode="rgb",
    target_size=dimension, # Depends of the model trained
    interpolation="nearest",
    )
    X[0,] = image
    return X

liste_ville = [
    ("./static/examples/frankfurt_000000_001016","Frankfurt"),
    ("./static/examples/lindau_000000_000019","Lindau"),
    ("./static/examples/munster_000046_000019","Munster")
]

def on_change_selector(state):

    path_img = state.ville[0]+"_leftImg8bit.png"
    X = load_image(path_img)

    t0 = time.time()
    byte_im = make_predict(X,model)
    inference_time = round(time.time() - t0,2)

    print("inference")

    state.image_originale = path_img
    state.image_traitee = byte_im
    state.image_solution = state.ville[0]+"_mask_colors.png"
    state.inference_time = inference_time

    print("finish")


page_examples = Markdown("""
# Exemple d'application de segmentation sémantique

Cette démo utilise un modèle Unet (avec encodeur VGG16) pour de la segmentation sémantique

# Sélection de l'image example

<|{ville}|selector|lov={liste_ville}|on_change=on_change_selector|>

# Affichage du résultat

<|{image_originale}|image|label=L'image originale|>
<|{image_traitee}|image|label=La segmentation issue du modèle|>
<|{image_solution}|image|label=Le résultat cible|>

Le tout a pris <|{inference_time}|>s pour s'executer.

""")