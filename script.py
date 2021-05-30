import multiprocessing
import time

import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

print("***** STARTING JOB ******")

def spectrum_embedding(x, p=2, binary=False):
	'''
	Computes the spectrum embedding of a seuqnce.
	The feature space contains the number of occurrences of all possible substrings

	p: the length of substructures
	binary: counts of occurrences or 1/0
	'''
	vx = {}
	for i in range(len(x)-p+1):
		u = x[i:i+p]
		vx[u] = 1 if u not in vx or binary else vx[u] + 1
	return vx


def dictionary_dot_internal(data):

    iz, vz, EX = data
    row = []
    for ix, vx in enumerate(EX):
        row.append((ix, iz, sum(vx[f] * vz[f] for f in vz.keys() & vx.keys())))

    return np.array(row)


def dictionary_dot_multiprocessing(EX, EZ, n_jobs=4):
    K = np.zeros((len(EX), len(EZ)))

    rows = [(iz, vz, EX) for iz, vz in enumerate(EZ)]
    pool = multiprocessing.Pool(n_jobs)
    results = pool.map(dictionary_dot_internal, rows)

    for row in np.array(results).reshape((-1, 3)):
        K[row[0], row[1]] = row[2]
    del results

    return np.array(K, dtype=np.int8)


def dictionary_dot_parallel(EX, EZ, n_jobs=-1):

    K = np.zeros((len(EX), len(EZ)))
    results = Parallel(n_jobs=n_jobs)(delayed(dictionary_dot_internal)( (iz, vz, EX) ) for iz, vz in enumerate(EZ))

    for row in np.array(results).reshape((-1, 3)):
        K[row[0], row[1]] = row[2]
    del results

    return np.array(K, dtype=np.int8)


def dictionary_dot(EX, EZ):
    K = np.zeros((len(EX), len(EZ)))
    for iz, vz in enumerate(EZ):
        for ix, vx in enumerate(EX):
            K[ix, iz] = sum(vx[f] * vz[f] for f in vz.keys() & vx.keys())
    return K

def nystrom_parallel(X_train, X_test, X_val, c=0.1, k=10000, seed=42, n_jobs=-1):
    print('**** START NYSTROM APPROXIMATION PARALLEL')
    with tf.device('/GPU:0'):

        print('COMPUTING APPROXIMATION --> CPU')
        start_cpu = time.time()
        _, X_train_idx = train_test_split(X_train, test_size=c, stratify=y_train)
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
        c_val = dictionary_dot_parallel(X_val, X_train_idx, n_jobs=n_jobs)
        end_cpu_second_step = time.time()

        print("COMPUTING DOT PRODUCT --> GPU")
        start_gpu_second_step = time.time()
        X_new_train = tf.tensordot(tf.convert_to_tensor(c_train, dtype=tf.float32), m, axes=1)
        X_new_test = tf.tensordot(tf.convert_to_tensor(c_test, dtype=tf.float32), m, axes=1)
        X_new_val = tf.tensordot(tf.convert_to_tensor(c_val, dtype=tf.float32), m, axes=1)
        end_gpu_second_step = time.time()

        time_on_cpu = (end_cpu_first_step-start_cpu) + (end_cpu_second_step-start_cpu_second_step)
        time_on_gpu = (end_gpu_first_step-start_gpu) + (end_gpu_second_step-start_gpu_second_step)
        print('TOTAL TIME ON CPU ', round( time_on_cpu,2 ))
        print('TOTAL TIME ON GPU ', round( time_on_gpu,2 ))
        print('TOTAL TIME ', round(time_on_cpu + time_on_gpu,2))
        print('**** END NYSTROM APPROXIMATION PARALLEL')
    return np.array(X_new_train), np.array(X_new_test), np.array(X_new_val)


def nystrom_multiprocessing(X_train, X_test, X_val, c=0.1, k=10000, seed=42, n_jobs=-1):
    print('**** START NYSTROM APPROXIMATION MULTIPROCESSING')
    with tf.device('/GPU:0'):
        print('COMPUTING APPROXIMATION --> CPU')
        start_cpu = time.time()
        _, X_train_idx = train_test_split(X_train, test_size=c, stratify=y_train)
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
        c_val = dictionary_dot_multiprocessing(X_val, X_train_idx, n_jobs=n_jobs)
        end_cpu_second_step = time.time()

        print("COMPUTING DOT PRODUCT --> GPU")
        start_gpu_second_step = time.time()
        X_new_train = tf.tensordot(tf.convert_to_tensor(c_train, dtype=tf.float32), m, axes=1)
        X_new_test = tf.tensordot(tf.convert_to_tensor(c_test, dtype=tf.float32), m, axes=1)
        X_new_val = tf.tensordot(tf.convert_to_tensor(c_val, dtype=tf.float32), m, axes=1)
        end_gpu_second_step = time.time()

        time_on_cpu = (end_cpu_first_step - start_cpu) + (end_cpu_second_step - start_cpu_second_step)
        time_on_gpu = (end_gpu_first_step - start_gpu) + (end_gpu_second_step - start_gpu_second_step)
        print('TOTAL TIME ON CPU ', round(time_on_cpu, 2))
        print('TOTAL TIME ON GPU ', round(time_on_gpu, 2))
        print('TOTAL TIME ', round(time_on_cpu + time_on_gpu,2))
        print('**** END NYSTROM APPROXIMATION MULTIPROCESSING')
    return np.array(X_new_train), np.array(X_new_test), np.array(X_new_val)

def compute_dataset_embedding(sentence, p=3):
    return spectrum_embedding(sentence, p=p, binary=False)


def compute_dataset_embedding_parallel(data, n_jobs=32):
    return Parallel(n_jobs=n_jobs)(delayed(compute_dataset_embedding)(data[i]) for i in range(len(data)))


def compute_dataset_multiprocessing(data, n_jobs=32):
    pool = multiprocessing.Pool(n_jobs)
    return pool.map(compute_dataset_embedding, data)


if __name__ == "__main__":

    n_jobs = 32
    path='./'

    print("**** READING FILES ****")
    train = pd.read_csv(path+'1ab8ab53-152d-480b-8d36-fcdb4c832ffc_train_no_stem.csv')
    test = pd.read_csv(path+'1ab8ab53-152d-480b-8d36-fcdb4c832ffc_test_no_stem.csv')
    val = pd.read_csv(path+'1ab8ab53-152d-480b-8d36-fcdb4c832ffc_val_no_stem.csv')

    X_train = train['text'].values
    y_train = train['label'].values

    X_test = test['text'].values
    y_test = test['label'].values

    X_val = val['text'].values
    y_val = val['label'].values

    print("*** COMPUTE EMBEDDINGS PARALLEL ****")
    start = time.time()
    X_train_embedding = np.array(compute_dataset_embedding_parallel(X_train, n_jobs=n_jobs))
    X_test_embedding = np.array(compute_dataset_embedding_parallel(X_test, n_jobs=n_jobs))
    X_val_embedding = np.array(compute_dataset_embedding_parallel(X_val, n_jobs=n_jobs))
    end = time.time()
    print("*** COMPUTE EMBEDDINGS PARALLEL TOOK: "+str(round(end - start,2))+" SECONDS ****")

    del X_train_embedding
    del X_test_embedding
    del X_val_embedding


    print("*** COMPUTE EMBEDDINGS MULTIPROCESS ****")
    start = time.time()
    X_train_embedding = np.array(compute_dataset_multiprocessing(X_train, n_jobs=n_jobs))
    X_test_embedding = np.array(compute_dataset_multiprocessing(X_test, n_jobs=n_jobs))
    X_val_embedding = np.array(compute_dataset_multiprocessing(X_val, n_jobs=n_jobs))
    end = time.time()
    print("*** COMPUTE EMBEDDINGS MULTIPROCESSING TOOK: " + str(round(end - start,2)) + " SECONDS ****")

    X_new_train, X_new_test , X_val_new = nystrom_parallel(X_train_embedding, X_test_embedding, X_val_embedding, c=0.0005, k=5000, seed=42, n_jobs=n_jobs)
    del X_new_train
    del X_new_test
    del X_val_new
    X_new_train, X_new_test , X_val_new = nystrom_multiprocessing(X_train_embedding, X_test_embedding, X_val_embedding, c=0.0005, k=5000, seed=42, n_jobs=n_jobs)


