import tensorflow as tf
import numpy as np
from generator import generate_data
from model import PointNet
SIZE = 100000

def train(model, data, labels):
    b_size = 512
    
    tot = data.shape[0]
    indice = tf.range(tot)
    indice = tf.random.shuffle(indice)
    
    data = tf.gather(data, indice)
    labels = tf.gather(labels, indice)
    
    loss_list = []
    num = 0
    for s in range(0, tot, b_size):
        print(num)
        num += 1
        e = min(tot, s + b_size)
        
        with tf.GradientTape() as tape:
            pred = model.call(data[s:e])
            loss, acc = model.loss(pred, labels[s:e])
            
        loss_list.append(loss)
        #bass = tf.reduce_sum(labels[s:e]) / (e - s)
        print("acc:{}".format(acc))
        #print("bass:{}".format(bass))
        print("loss:{}".format(loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    return loss_list

def test(model, data, labels):
    b_size = 256
    model.probs = 0

    tot = data.shape[0]
    acc_tot = 0.0
    num = 0
    for s in range(0, tot, b_size):
        e = min(tot, s + b_size)
        pred = model.call(data[s:e])
        loss, acc = model.loss(pred, labels[s:e])
        acc_tot += acc
        num += 1

    return acc_tot / num


if __name__ == '__main__':
    data, labels = generate_data(SIZE)
    m = PointNet(0.7)
    loss_list = train(m, data, labels)
    data, labels = generate_data(1000)
    print(test(m, data, labels))

