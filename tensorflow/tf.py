import tensorflow as tf
import time


image_list = ['../images/ILSVRC12_1.jpeg',\
              '../images/ILSVRC12_2.jpeg']

filename_queue = tf.train.string_input_producer(image_list)

image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_file)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    start = time.time()\

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    with tf.device('/gpu:0'):
        for i in range(2):
            my_img = image.eval()
            print my_img.shape

    coord.request_stop()
    coord.join(threads)

    end = time.time()
    print '{} seconds'.format(end - start)
