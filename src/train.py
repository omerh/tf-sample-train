import os
import argparse
import tensorflow as tf
from utils import prepare_fs, load_dataset, save_dataset_local, get_train_data, get_test_data, install_package
from processing import preprocessing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    return parser.parse_known_args()

def get_model():
    inputs = tf.keras.Input(shape=(13,))
    hidden_1 = tf.keras.layers.Dense(13, activation='tanh')(inputs)
    hidden_2 = tf.keras.layers.Dense(6, activation='sigmoid')(hidden_1)
    outputs = tf.keras.layers.Dense(1)(hidden_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def score_model(model, test_dir):
    x_test, y_test = get_test_data(test_dir)
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest MSE :", scores)


def start(_batch_size, _epochs, _learning_rate):
    print("Starting Training script...")
    print("Tensorflow version {}".format(tf.__version__))
    data_dir, raw_dir, train_dir, test_dir = prepare_fs()
    (x_train, y_train), (x_test, y_test) = load_dataset()
    save_dataset_local(raw_dir, x_train, y_train, x_test, y_test)
    preprocessing(raw_dir, data_dir, train_dir, test_dir)
    x_train, y_train = get_train_data(train_dir)
    x_test, y_test = get_test_data(test_dir)

    # Training parameters
    device = '/cpu:0'
    print(device)
    batch_size = _batch_size
    epochs = _epochs
    learning_rate = _learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    # Run training
    with tf.device(device):
        model = get_model()
        optimizer = tf.keras.optimizers.SGD(learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(x_test, y_test))

        # evaluate on test set
        scores = model.evaluate(x_test, y_test, batch_size, verbose=2)
        print("\nTest MSE :", scores)

    # Save model
    model.save(os.getenv("SM_MODEL_DIR") + '/1')


if __name__ == "__main__":
    # Check if in local docker and not in SageMaker
    if os.environ.get("SM_HOSTS") is None:
        os.environ["SM_MODEL_DIR"] = "./model"

    args, _ = parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    start(batch_size, epochs, learning_rate)
    # model = start()
    # score_model(model, '../data/test/')
