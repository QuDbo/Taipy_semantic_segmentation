import tensorflow as tf
import tensorflow.keras.preprocessing.image as imgtf
import numpy as np
from io import BytesIO

def load_model(local_model_path):
    model = tf.keras.models.load_model(local_model_path,
                                       custom_objects={"iou_coef":iou_coef}
                                      )
    return model

def convert_mask_to_color(masks):
    '''
    Convert a array of mask (height,width,8) in color visualisation array (height,width,3)
    '''
    dic_color = {
        0 : [76,0,153], # flat
        1 : [0,0,204], # vehicle
        2 : [96,96,96], # construction
        3 : [224,224,224], # object
        4 : [0,204,0], # nature
        5 : [255,0,0], # human
        6 : [153,255,255], # sky
        7 : [0,0,0] # void
    }
    dic_r = {k:dic_color[k][0] for k in dic_color.keys()}
    dic_g = {k:dic_color[k][1] for k in dic_color.keys()}
    dic_b = {k:dic_color[k][2] for k in dic_color.keys()}
    
    id_mask = np.argmax(masks,axis=-1)
    color_masks_r = np.vectorize(dic_r.__getitem__)(id_mask)
    color_masks_g = np.vectorize(dic_g.__getitem__)(id_mask)
    color_masks_b= np.vectorize(dic_b.__getitem__)(id_mask)
    color_masks = np.stack((color_masks_r,color_masks_g,color_masks_b),axis=-1)
    return color_masks

def make_predict(X,model):
    '''
    Load the image store as static/image_submit.png and convert to an array
    Load the model and make a prediction
    Convert the masks array into a image and store it
    '''
    
    ### Prediction
    y_pred = model.predict(X)
    masks = y_pred[0,]
    image_masks = convert_mask_to_color(masks)
    image_to_save = imgtf.array_to_img(image_masks)
    buf = BytesIO()

    tf.keras.utils.save_img(buf, image_to_save,
                            data_format="channels_last",
                            file_format="PNG")
    byte_im = buf.getvalue()
    return byte_im
    ###

def iou_coef(y_true, y_pred, smooth=1e-1):

    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    smooth = tf.cast(smooth, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou