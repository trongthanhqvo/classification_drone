import os
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# os.environ["CUDA_VISIBLE_DEVICES"] = "8"

images_dir = "/home/sonhh/thanhnt/classification_drone/images/"
model_path = "./50-0.9780-20220818-2class-drone-effb0.h5"
class_name = {0: 'ambient', 1: 'drone'}

if __name__ == '__main__':
    model = load_model(model_path)

    for AK_or_SK in os.listdir(images_dir):
        for picture_name in os.listdir(os.path.join(images_dir, AK_or_SK)):
            img_path = os.path.join(images_dir, AK_or_SK, picture_name)
            print(img_path)
            img = load_img(img_path, target_size=(300, 300))  
            img = img_to_array(img)  
            img = np.expand_dims(img, axis=0) 

            
            preds = model.predict(img)

            
            # print(preds)
            y_pred = np.argmax(preds, axis=1)

            label = class_name[y_pred[0]]   
            print(picture_name, 'preds：')
            print(preds, ' --> ', label)