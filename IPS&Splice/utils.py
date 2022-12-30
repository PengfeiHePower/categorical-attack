import numpy as np
import pickle


def pad_matrix(seq_diagnosis_codes, seq_labels, n_diagnosis_codes):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    maxlen = np.max(lengths)

    #f_1 = 1e-5
    f_1 = 0.0
    batch_diagnosis_codes = f_1 * np.ones((maxlen, n_samples, n_diagnosis_codes), dtype=np.float32)

    for idx, c in enumerate(seq_diagnosis_codes):
        for x, subseq in zip(batch_diagnosis_codes[:, idx, :], c[:]):
            l = 1
            f_2 = float((l - f_1 * (n_diagnosis_codes - l)) / l)
            x[subseq] = f_2

    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_diagnosis_codes, batch_labels


def load_data(Dataset, test=True):
    if test:
        data = pickle.load(open(Test_Data_File[Dataset], 'rb'))
        label = pickle.load(open(Test_Label_File[Dataset], 'rb'))
        data = np.array(data)
        return data, label
    data = pickle.load(open(Whole_Data_File[Dataset], 'rb'))
    label = pickle.load(open(Whole_Label_File[Dataset], 'rb'))
    data = np.array(data)
    return data, label


def write_file(Dataset, Model_Type, budget, algorithm, time_limit):
    log_f = open('./Logs/%s/MF_%s_%d_%a.bak' % (Dataset, Model_Type, budget, algorithm), 'w+')
    TITLE = '=== ' + Dataset + Model_Type + str(budget) + algorithm + ' time = ' + str(time_limit) + ' ==='
    print(TITLE, file=log_f, flush=True)
    directory = './Logs/%s/%s/%s' % (Dataset, Model_Type, algorithm)
    print()
    print(directory)
    print(directory, file=log_f, flush=True)
    Algorithm = directory
    mf_process_temp = pickle.load(open(Algorithm + 'mf_process_%d.pickle' % budget, 'rb'))
    changed_set_process_temp = pickle.load(open(directory + 'changed_set_process_%d.pickle' % budget, 'rb'))
    robust_flag = pickle.load(open(directory + 'robust_flag_%d.pickle' % budget, 'rb'))
    query_num = pickle.load(open(directory + 'querynum_%d.pickle' % budget, 'rb'))
    time = pickle.load(open(directory + 'time_%d.pickle' % budget, 'rb'))
    iteration_file = pickle.load(open(directory + 'iteration_%d.pickle' % budget, 'rb'))
    mf_process = []
    changed_set_process = []
    time_attack = []
    query_num_attack = []
    flip_changed_num = []
    iteration = []
    for j in range(len(robust_flag)):
        if robust_flag[j] == 0:
            mf_process.append(mf_process_temp[j])
            changed_set_process.append(changed_set_process_temp[j])
            time_attack.append(time[j])
            query_num_attack.append(query_num[j])
            flip_changed_num.append(len(changed_set_process_temp[j][-1]))
            iteration.append(iteration_file[j])

    sorted_flip_changed_num = np.sort(flip_changed_num)
    change_medium = sorted_flip_changed_num[len(flip_changed_num) // 2]

    print('success rate:', len(iteration) / len(mf_process_temp))
    print('average iteration:', np.mean(iteration))
    print('average changed code', np.mean(flip_changed_num))
    print('average time:', np.mean(time_attack))
    print('average query number', np.mean(query_num_attack))
    print('medium changed number', change_medium)

    print('success rate:', len(iteration) / len(mf_process_temp), file=log_f, flush=True)
    print('average iteration:', np.mean(iteration), file=log_f, flush=True)
    print('average changed code', np.mean(flip_changed_num), file=log_f, flush=True)
    print('average time:', np.mean(time_attack), file=log_f, flush=True)
    print('average query number', np.mean(query_num_attack), file=log_f, flush=True)
    print('medium changed number', change_medium, file=log_f, flush=True)

    print('end')
    print()
    print()
    print()



Test_Data_File = {
    'Splice': './dataset/gene_test_funccall.pickle',
    'IPS': './dataset/mal_test_funccall.pickle'
}

Test_Label_File = {
    'Splice': './dataset/gene_test_label.pickle',
    'IPS': './dataset/mal_test_label.pickle'
}

Whole_Data_File = {
    'Splice': './dataset/spliceX.pickle',
}

Whole_Label_File = {
    'Splice': './dataset/spliceY.pickle',
}

num_category = {'Splice': 5, 'IPS': 1104}
num_feature = {'Splice': 60, 'IPS': 20}