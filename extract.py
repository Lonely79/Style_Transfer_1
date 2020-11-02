import tensorflow as tf
import numpy as np
from vgg19.vgg19 import Network_vgg
import cv2

class Extract:

    def __init__(self,content_path, style_path,result_path,H=512, W=512, C=3,):
        self.content_layers = ['relu1_2','relu2_2','relu3_2','relu4_2','relu5_2']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.mean = [116.779, 103.939, 123.68]

        self.content_img = tf.placeholder("float", [1, H, W, C])
        self.style_img = tf.placeholder("float", [1, H, W, C])

        style_feature = Network_vgg(self.style_img)
        content_feature = Network_vgg(self.content_img)

        content_img = self.imread(content_path)
        style_img = self.imread(style_path)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {self.content_img: content_img, self.style_img: style_img}
            target_img=0

            for name in self.content_layers:
                target_img = sess.run(self.extract_content(name,content_feature), feed_dict)
                self.save_image("content_"+str(name),target_img,result_path)

    def save_image(self,name,target_img,result_path):
        image = np.clip(target_img[0, :, :, 0], 0, 255).astype(np.uint8)
        image = cv2.resize(image, (512, 512))
        image = np.squeeze(image)
        cv2.imwrite(result_path + str(name) + ".jpg", image)

    def extract_content(self,name,content_feature):
        return content_feature[name]

    def imread(self,image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        image = np.array([image-self.mean]).astype(np.float32)
        return image

if __name__ == "__main__":
    content_path = "content/1.jpg"
    style_path = "style/style.jpg"
    result_path = "extract/"

    Extract(content_path=content_path, style_path=style_path,result_path=result_path)