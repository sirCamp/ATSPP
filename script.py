import multiprocessing
import time

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

print("***** STARTING JOB ******")


def train_and_test_svm(x_train, y_train, x_test, y_test, C, max_iter=100):

    svc = LinearSVC(C=C, multi_class='ovr', max_iter=max_iter)
    svc.fit(x_train, y_train)

    acc = accuracy_score(y_test, svc.predict(x_test))
    f1 = f1_score(y_test, svc.predict(x_test), average='weighted')

    print("ACCURACY: "+str(round(acc, 1))+" F1: "+str(round(f1, 2)))

def define_train_and_test_svm(x_train, y_train, x_test, y_test, C, epochs=100, batch_size=64):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softmax,
                                    kernel_initializer=tf.keras.initializers.Zeros(),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.05)
                                    )
              )

    def svm_custom_loss(y_true, y_pred, C=C):
        hinge = tf.keras.losses.Hinge(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        return C * hinge(y_pred=y_pred, y_true=y_true)

    # model.add_loss(loss)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.01),
        loss=svm_custom_loss,
        metrics=[tf.keras.metrics.categorical_accuracy, tfa.metrics.F1Score(average='weighted', num_classes=3)])

    model.fit(x_train, tf.keras.utils.to_categorical(y_train), epochs=epochs, batch_size=batch_size)

    evals = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test), batch_size=batch_size)

    print(
        "LOSS: " + str(round(evals[0], 3)) + " ACCURACY: " + str(round(evals[1], 3)) + " F1: " + str(round(evals[2], 3)))

def spectrum_embedding(x, p=2, binary=False):
    '''
	Computes the spectrum embedding of a seuqnce.
	The feature space contains the number of occurrences of all possible substrings

	p: the length of substructures
	binary: counts of occurrences or 1/0
	'''
    vx = {}
    for i in range(len(x) - p + 1):
        u = x[i:i + p]
        vx[u] = 1 if u not in vx or binary else vx[u] + 1
    return vx


def dictionary_dot_internal(data):
    """
    Inner dictionary dot product
    :param data:
    :return:
    """
    iz, vz, EX = data
    row = []
    for ix, vx in enumerate(EX):
        row.append((ix, iz, sum(vx[f] * vz[f] for f in vz.keys() & vx.keys())))

    return np.array(row)


def dictionary_dot_multiprocessing(EX, EZ, n_jobs=4):
    """
    Multiprocessing dictionary dot product
    :param EX:
    :param EZ:
    :param n_jobs
    :return:
    """
    K = np.zeros((len(EX), len(EZ)))

    rows = [(iz, vz, EX) for iz, vz in enumerate(EZ)]
    pool = multiprocessing.Pool(n_jobs)
    results = pool.map(dictionary_dot_internal, rows)

    for row in np.array(results).reshape((-1, 3)):
        K[row[0], row[1]] = row[2]
    del results

    return np.array(K, dtype=np.int8)


def dictionary_dot_parallel(EX, EZ, n_jobs=-1):
    """
    Parallel dictionary dot product with joblib and locky backend
    :param EX:
    :param EZ:
    :param n_jobs
    :return:
    """
    K = np.zeros((len(EX), len(EZ)))
    results = Parallel(n_jobs=n_jobs)(delayed(dictionary_dot_internal)((iz, vz, EX)) for iz, vz in enumerate(EZ))

    for row in np.array(results).reshape((-1, 3)):
        K[row[0], row[1]] = row[2]
    del results

    return np.array(K, dtype=np.int8)


def dictionary_dot(EX, EZ):
    """
    Sequential dictionary dot product
    :param EX:
    :param EZ:
    :return:
    """
    K = np.zeros((len(EX), len(EZ)))
    for iz, vz in enumerate(EZ):
        for ix, vx in enumerate(EX):
            K[ix, iz] = sum(vx[f] * vz[f] for f in vz.keys() & vx.keys())
    return K


def nystrom_parallel(X_train, X_test, c=0.1, random_state=42, n_jobs=-1):
    """
    Computing nystrom spectrum kernel approximation with joblib and Locky backend way
    :param X_train:
    :param X_test:
    :param X_val:
    :param c:
    :param random_state:
    :param n_jobs:
    :return:
    """
    print('**** START NYSTROM APPROXIMATION PARALLEL')
    with tf.device('/GPU:0'):
        print('COMPUTING APPROXIMATION --> CPU')
        start_cpu = time.time()
        _, X_train_idx = train_test_split(X_train, test_size=c, stratify=y_train, random_state=random_state)
        w = dictionary_dot_parallel(X_train_idx, X_train_idx, n_jobs=n_jobs)
        w = np.array(w, dtype=np.float32)
        end_cpu_first_step = time.time()

        print('COMPUTING SVD --> GPU')
        start_gpu = time.time()

        w_tensor = tf.convert_to_tensor(w, dtype=tf.float32)
        s, u, v = tf.linalg.svd(w_tensor, full_matrices=False, compute_uv=True)
        m = tf.tensordot(u, tf.linalg.tensor_diag(tf.constant(1, dtype=tf.float32) / tf.math.sqrt(s)), axes=1)
        end_gpu_first_step = time.time()

        print('COMPUTING APPROXIMATION FOR DATA --> CPU')
        start_cpu_second_step = time.time()
        c_train = dictionary_dot_parallel(X_train, X_train_idx, n_jobs=n_jobs)
        c_test = dictionary_dot_parallel(X_test, X_train_idx, n_jobs=n_jobs)
        end_cpu_second_step = time.time()

        print("COMPUTING DOT PRODUCT --> GPU")
        start_gpu_second_step = time.time()
        X_new_train = tf.tensordot(tf.convert_to_tensor(c_train, dtype=tf.float32), m, axes=1)
        X_new_test = tf.tensordot(tf.convert_to_tensor(c_test, dtype=tf.float32), m, axes=1)
        end_gpu_second_step = time.time()

        time_on_cpu = (end_cpu_first_step - start_cpu) + (end_cpu_second_step - start_cpu_second_step)
        time_on_gpu = (end_gpu_first_step - start_gpu) + (end_gpu_second_step - start_gpu_second_step)
        print('TOTAL TIME ON CPU ', round(time_on_cpu, 2))
        print('TOTAL TIME ON GPU ', round(time_on_gpu, 2))
        print('TOTAL TIME ', round(time_on_cpu + time_on_gpu, 2))
        print('**** END NYSTROM APPROXIMATION PARALLEL')
    return np.array(X_new_train), np.array(X_new_test)


def nystrom_multiprocessing(X_train, X_test, c=0.1, random_state=42, n_jobs=-1):
    """
    Computing nystrom spectrum kernel approximation with multiprocessing way
    :param X_train:
    :param X_test:
    :param X_val:
    :param c:
    :param random_state:
    :param n_jobs:
    :return:
    """
    print('**** START NYSTROM APPROXIMATION MULTIPROCESSING')
    with tf.device('/GPU:0'):
        print('COMPUTING APPROXIMATION --> CPU')
        start_cpu = time.time()
        _, X_train_idx = train_test_split(X_train, test_size=c, stratify=y_train, random_state=random_state)
        w = dictionary_dot_multiprocessing(X_train_idx, X_train_idx, n_jobs=n_jobs)
        w = np.array(w, dtype=np.float32)
        end_cpu_first_step = time.time()

        print('COMPUTING SVD --> GPU')
        start_gpu = time.time()

        w_tensor = tf.convert_to_tensor(w, dtype=tf.float32)
        s, u, v = tf.linalg.svd(w_tensor, full_matrices=False, compute_uv=True)
        m = tf.tensordot(u, tf.linalg.tensor_diag(tf.constant(1, dtype=tf.float32) / tf.math.sqrt(s)), axes=1)
        end_gpu_first_step = time.time()

        print('COMPUTING APPROXIMATION FOR DATA --> CPU')
        start_cpu_second_step = time.time()
        c_train = dictionary_dot_multiprocessing(X_train, X_train_idx, n_jobs=n_jobs)
        c_test = dictionary_dot_multiprocessing(X_test, X_train_idx, n_jobs=n_jobs)
        end_cpu_second_step = time.time()

        print("COMPUTING DOT PRODUCT --> GPU")
        start_gpu_second_step = time.time()
        X_new_train = tf.tensordot(tf.convert_to_tensor(c_train, dtype=tf.float32), m, axes=1)
        X_new_test = tf.tensordot(tf.convert_to_tensor(c_test, dtype=tf.float32), m, axes=1)
        end_gpu_second_step = time.time()

        time_on_cpu = (end_cpu_first_step - start_cpu) + (end_cpu_second_step - start_cpu_second_step)
        time_on_gpu = (end_gpu_first_step - start_gpu) + (end_gpu_second_step - start_gpu_second_step)
        print('TOTAL TIME ON CPU ', round(time_on_cpu, 2))
        print('TOTAL TIME ON GPU ', round(time_on_gpu, 2))
        print('TOTAL TIME ', round(time_on_cpu + time_on_gpu, 2))
        print('**** END NYSTROM APPROXIMATION MULTIPROCESSING')

    return np.array(X_new_train), np.array(X_new_test)


def nystrom_sequential(X_train, X_test, c=0.1, random_state=42):
    """
    Computing nystrom spectrum kernel approximation with sequential way
    :param X_train:
    :param X_test:
    :param X_val:
    :param c:
    :param random_state:
    :return:
    """
    print('**** START NYSTROM APPROXIMATION SEQUENTIAL')
    with tf.device('/GPU:0'):
        print('COMPUTING APPROXIMATION --> CPU')
        start_cpu = time.time()
        _, X_train_idx = train_test_split(X_train, test_size=c, stratify=y_train, random_state=random_state)
        w = dictionary_dot(X_train_idx, X_train_idx)
        w = np.array(w, dtype=np.float32)
        end_cpu_first_step = time.time()

        print('COMPUTING SVD --> GPU')
        start_gpu = time.time()

        w_tensor = tf.convert_to_tensor(w, dtype=tf.float32)
        s, u, v = tf.linalg.svd(w_tensor, full_matrices=False, compute_uv=True)
        m = tf.tensordot(u, tf.linalg.tensor_diag(tf.constant(1, dtype=tf.float32) / tf.math.sqrt(s)), axes=1)
        end_gpu_first_step = time.time()

        print('COMPUTING APPROXIMATION FOR DATA --> CPU')
        start_cpu_second_step = time.time()
        c_train = dictionary_dot(X_train, X_train_idx)
        c_test = dictionary_dot(X_test, X_train_idx)
        end_cpu_second_step = time.time()

        print("COMPUTING DOT PRODUCT --> GPU")
        start_gpu_second_step = time.time()
        X_new_train = tf.tensordot(tf.convert_to_tensor(c_train, dtype=tf.float32), m, axes=1)
        X_new_test = tf.tensordot(tf.convert_to_tensor(c_test, dtype=tf.float32), m, axes=1)
        end_gpu_second_step = time.time()

        time_on_cpu = (end_cpu_first_step - start_cpu) + (end_cpu_second_step - start_cpu_second_step)
        time_on_gpu = (end_gpu_first_step - start_gpu) + (end_gpu_second_step - start_gpu_second_step)
        print('TOTAL TIME ON CPU ', round(time_on_cpu, 2))
        print('TOTAL TIME ON GPU ', round(time_on_gpu, 2))
        print('TOTAL TIME ', round(time_on_cpu + time_on_gpu, 2))
        print('**** END NYSTROM APPROXIMATION SEQUENTIAL')

    return np.array(X_new_train), np.array(X_new_test)


def compute_dataset_embedding(sentence, p=4):
    """
    Computing the spectrum embedding for a sentence
    :param data:
    :param n_jobs:
    :return:
    """
    return spectrum_embedding(sentence, p=p, binary=False)


def compute_dataset_embedding_sequential(data):
    """
    Computing the spectrum embedding with sequential way
    :param data:
    :param n_jobs:
    :return:
    """
    return [compute_dataset_embedding(sentence) for sentence in data ]


def compute_dataset_embedding_parallel(data, n_jobs=32):
    """
    Computing the spectrum embedding with joblib and Locky backend processing
    :param data:
    :param n_jobs:
    :return:
    """
    return Parallel(n_jobs=n_jobs)(delayed(compute_dataset_embedding)(data[i]) for i in range(len(data)))


def compute_dataset_embedding_multiprocessing(data, n_jobs=32):
    """
    Computing the spectrum embedding with multiprocessing
    :param data:
    :param n_jobs:
    :return:
    """
    pool = multiprocessing.Pool(n_jobs)
    return pool.map(compute_dataset_embedding, data)


if __name__ == "__main__":

    n_jobs = 32
    path = './'
    c = 0.05
    C = 100
    batch_size = 128
    epochs = 100

    print("**** READING FILES ****")
    train = pd.read_csv(path + '1ab8ab53-152d-480b-8d36-fcdb4c832ffc_train_no_stem.csv')
    test = pd.read_csv(path + '1ab8ab53-152d-480b-8d36-fcdb4c832ffc_test_no_stem.csv')
    val = pd.read_csv(path + '1ab8ab53-152d-480b-8d36-fcdb4c832ffc_val_no_stem.csv')

    train = pd.concat([train, val], axis=0)

    X_train = train['text'].values
    y_train = train['label'].values

    X_test = test['text'].values
    y_test = test['label'].values

    print("**** COMPUTE EMBEDDINGS PARALLEL ****")
    start = time.time()
    X_train_embedding = np.array(compute_dataset_embedding_parallel(X_train, n_jobs=n_jobs))
    X_test_embedding = np.array(compute_dataset_embedding_parallel(X_test, n_jobs=n_jobs))
    end = time.time()
    print("**** COMPUTE EMBEDDINGS PARALLEL TOOK: " + str(round(end - start, 2)) + " SECONDS ****")

    del X_train_embedding
    del X_test_embedding

    print("**** COMPUTE EMBEDDINGS MULTIPROCESS ****")
    start = time.time()
    X_train_embedding = np.array(compute_dataset_embedding_multiprocessing(X_train, n_jobs=n_jobs))
    X_test_embedding = np.array(compute_dataset_embedding_multiprocessing(X_test, n_jobs=n_jobs))
    end = time.time()
    print("**** COMPUTE EMBEDDINGS MULTIPROCESSING TOOK: " + str(round(end - start, 2)) + " SECONDS ****")

    del X_train_embedding
    del X_test_embedding

    print("**** COMPUTE EMBEDDINGS SEQUENTIAL ****")
    start = time.time()
    X_train_embedding = np.array(compute_dataset_embedding_sequential(X_train))
    X_test_embedding = np.array(compute_dataset_embedding_sequential(X_test))
    end = time.time()
    print("**** COMPUTE EMBEDDINGS SEQUENTIAL TOOK: " + str(round(end - start, 2)) + " SECONDS ****")

    X_new_train, X_new_test = nystrom_parallel(X_train_embedding, X_test_embedding, c=c,
                                                          n_jobs=n_jobs)

    del X_new_train
    del X_new_test

    X_new_train, X_new_test = nystrom_multiprocessing(X_train_embedding, X_test_embedding,
                                                                c=c, n_jobs=n_jobs)

    del X_new_train
    del X_new_test

    X_new_train, X_new_test = nystrom_sequential(X_train_embedding, X_test_embedding, c=c)

    start_nn_svm = time.time()
    define_train_and_test_svm(X_new_train, y_train, X_new_test, y_test, C=C, epochs=epochs, batch_size=batch_size)
    end_nn_svm = time.time()
    print("**** SVM ON GPU TOOK: " + str(round(end_nn_svm - start_nn_svm, 2)) + " SECONDS ****")


    start_nn_svm = time.time()
    train_and_test_svm(X_new_train, y_train, X_new_test, y_test, C=C)
    end_nn_svm = time.time()
    print("**** SVM ON CPU TOOK: " + str(round(end_nn_svm - start_nn_svm, 2)) + " SECONDS ****")
