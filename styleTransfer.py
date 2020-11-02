import tensorflow as tf
import numpy as np
from vgg19.vgg19 import Network_vgg
import cv2

class StyleTransfer:

    def __init__(self,content_path, style_path,result_path,H=256, W=256, C=3, alpha=1e-3, beta=1.0, iteration=1000):
        self.content_layers = ['relu4_2']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.mean=[116.779, 103.939, 123.68]

        self.content_img = tf.placeholder("float", [1, H, W, C])
        self.style_img = tf.placeholder("float", [1, H, W, C])
        self.target_img = tf.get_variable("target", shape=[1, H, W, C], initializer=tf.truncated_normal_initializer(stddev=0.02))

        target_feature = Network_vgg(self.target_img)
        style_feature = Network_vgg(self.style_img)
        content_feature = Network_vgg(self.content_img)

        self.content_loss = self.content_loss(target_feature, content_feature)
        self.style_loss = self.style_loss(target_feature, style_feature)
        self.total_loss = alpha * self.content_loss + beta * self.style_loss
        # self.Opt = tf.train.AdamOptimizer(0.0002).minimize(self.total_loss)
        #L-BFGS
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.total_loss, method='L-BFGS-B',options={'maxiter': iteration, 'disp': 0})
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.train(content_path, style_path,result_path)

    def train(self, content_path, style_path,result_path):
        content_img = self.imread(content_path)
        style_img = self.imread(style_path)
        #喂入计算图
        feed_dict = {self.content_img: content_img, self.style_img: style_img}
        #目标图片初始化为内容图片
        self.sess.run(tf.assign(self.target_img, content_img), feed_dict=feed_dict)
        #loss最小化
        self.optimizer.minimize(self.sess, feed_dict=feed_dict)
        #计算loss
        content_loss = self.sess.run(self.content_loss, feed_dict=feed_dict)
        style_loss = self.sess.run(self.style_loss, feed_dict=feed_dict)
        total_loss = self.sess.run(self.total_loss, feed_dict=feed_dict)
        print("content_loss: %g, style_loss: %g, total_loss: %g" % (content_loss, style_loss, total_loss))
        #得到最终target参数
        target_img = self.sess.run(self.target_img,feed_dict=feed_dict)

        #重新加上参数，clip消去0-255之外的数据，squeeze去掉虚维度
        target_img = target_img+self.mean
        image = np.clip(target_img, 0, 255).astype(np.uint8)
        image = np.squeeze(image)
        cv2.imwrite(result_path,image)

    def imread(self,image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        #vgg19输入图均作了处理，所以这里需要减去mean
        image = np.array([image-self.mean]).astype(np.float32)
        return image

    def content_loss(self, target_feature, content_feature):
        return tf.nn.l2_loss(target_feature["relu4_2"] - content_feature["relu4_2"])

    def style_loss(self, target_feature, style_feature):
        E = 0
        for layer in style_feature.keys():
            if layer in self.style_layers:
                w = 0.2
            else:
                w = 0
            H = int(target_feature[layer].shape[1])
            W = int(target_feature[layer].shape[2])

            # Gram matrix of target
            C = int(target_feature[layer].shape[-1])
            F = tf.reshape(tf.transpose(target_feature[layer], [0, 3, 1, 2]), shape=[C, -1])
            G_x = tf.matmul(F, tf.transpose(F))

            # Gram matrix of style
            C = int(style_feature[layer].shape[-1])
            F = tf.reshape(tf.transpose(style_feature[layer], [0, 3, 1, 2]), shape=[C, -1])
            G_s = tf.matmul(F, tf.transpose(F))

            E += w * tf.reduce_sum(tf.square(G_x - G_s)) / (4 * C**2 * H**2 * W**2)
        return E

if __name__ == "__main__":
    content_path = "content/content.jpg"
    style_path = "style/style.jpg"
    result_path = "result/tiger3.jpg"

    st = StyleTransfer(content_path=content_path, style_path=style_path,result_path=result_path,H=512, W=512, C=3, alpha=1e-5, beta=1.0, iteration=2000)