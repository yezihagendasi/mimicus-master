# -*- coding: UTF-8 -*-
'''
Copyright 2014 Nedim Srndic, University of Tuebingen

This file is part of Mimicus.

Mimicus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Mimicus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Mimicus.  If not, see <http://www.gnu.org/licenses/>.
##############################################################################
common.py

Code common to all scenarios.

Created on March 5, 2014.
'''

import multiprocessing
from os import path
import pickle
import random
import shutil
import sys

from matplotlib import pyplot
from mimicus import config as _ # Just to create the configuration file
from mimicus.attacks.mimicry import mimicry
from mimicus.attacks.gdkde import gdkde
from mimicus.classifiers.RandomForest import RandomForest
from mimicus.classifiers.sklearn_SVC import sklearn_SVC
from mimicus.tools import datasets, utility
from mimicus.tools.featureedit import FeatureDescriptor, FeatureEdit
import numpy
from sklearn.metrics import accuracy_score

import config

'''
Top features sorted by variable importance as reported by the R 
randomForest package for the trained model in the FTC scenario. 
'''
top_feats = ['count_font', 
             'count_js', 
             'count_javascript', 
             'pos_box_max', 
             'pos_eof_avg', 
             'pos_eof_max', 
             'len_stream_min', 
             'count_obj', 
             'count_endobj',
             'producer_len']

def get_benign_mean_stddev(data, labels):
    '''
    Returns feature medians and standard deviations for benign 
    training data. 
    '''
    print 'Getting medians and std. dev. for features of benign training data'
    benign_vectors = data[[i for i, l in enumerate(labels) if l == 0],]
    return (numpy.mean(benign_vectors, axis = 0), 
            numpy.std(benign_vectors, axis = 0))

def get_FTC_mimicry():
    '''
    Returns a numpy.array of size (number of samples, number of 
    features) with feature values of all mimicry attack results in 
    the FTC scenario.
    '''
    # pdfs = utility.get_pdfs(config.get('results', 'FTC_mimicry'))
    pdfs = utility.get_pdfs('I:\\mimicus\\mimicus-master/results/FTC_mimicry')
    if not pdfs:
        # Generate the attack files
        attack_mimicry('FTC')
        pdfs = utility.get_pdfs('I:\\mimicus\\mimicus-master/results/FTC_mimicry')
        # pdfs = utility.get_pdfs(config.get('results', 'FTC_mimicry'))

    print 'Loading feature vectors from mimicry attack results...'
    results = numpy.zeros((len(pdfs), FeatureDescriptor.get_feature_count()))
    for i in range(len(pdfs)):
        results[i,] = FeatureEdit(pdfs[i]).retrieve_feature_vector_numpy()
    
    return results, [1.0 for i in range(len(pdfs))]

def evaluate_classifier(data, labels, test_data, test_labels):
    '''
    Returns the classification accuracies of the RandomForest 
    classifier trained on (data, labels) and tested on a list of 
    (test_data, test_labels). 
    '''
    rf = RandomForest()
    rf.fit(data, labels)
    accs = []
    for ted, tel in zip(test_data, test_labels):
        pred = rf.predict(ted)
        accs.append(accuracy_score(tel, pred))
    return accs

'''
A dictionary encoding adversarial knowledge for every scenario.
'''
_scenarios = \
{'F' : {'classifier' : 'svm', 
        'model' : 'I:\\mimicus\\mimicus-master/data/F-scaled.model',
        'targets' : 'I:\\mimicus\\mimicus-master/data/google-ben.csv',
        'training' : 'I:\\mimicus\\mimicus-master/data/surrogate-scaled.csv'},
 'FT' : {'classifier' : 'svm',
         'model' : 'I:\\mimicus\\mimicus-master/data/FT-scaled.model',
         'targets' : 'I:\\mimicus\\mimicus-master/data/contagio-ben.csv',
         'training' : 'I:\\mimicus\\mimicus-master/data/contagio-scaled.csv'},
 'FC' : {'classifier' : 'rf',
         'model' : 'I:\\mimicus\\mimicus-master/data/FC.model',
         'targets' : 'I:\\mimicus\\mimicus-master/data/google-ben.csv',
         'training' : 'I:\\mimicus\\mimicus-master/data/surrogate.csv'},
 'FTC' : {'classifier' : 'rf',
         'model' : 'I:\\mimicus\\mimicus-master/data/FTC.model',
         'targets' : 'I:\\mimicus\\mimicus-master/data/contagio-ben.csv',
         'training' : 'I:/mimicus/mimicus-master/data/contagio.csv'},
 }

def _learn_model(scenario_name):
    '''
    Learns a classifier model for the specified scenario if one does 
    not already exist. 
    '''
    # 学习指定场景的分类器模型（如果尚未存在）。
    # 'F': {'classifier': 'svm',
    #       'model': 'I:\\mimicus\\mimicus-master/data/F-scaled.model',
    #       'targets': 'I:\\mimicus\\mimicus-master/data/google-ben.csv',
    #       'training': 'I:\\mimicus\\mimicus-master/data/surrogate-scaled.csv'},
    scenario = _scenarios[scenario_name]
    if path.exists(scenario['model']):
        return
    
    print 'Training the model for scenario {}...'.format(scenario_name)
    # Decide on classifier
    classifier = 0
    if scenario['classifier'] == 'rf':
        classifier = RandomForest()
        sys.stdout.write('TRAINING RANDOM FOREST\n')
        cutoff = [c * 0.1 for c in range(1, 10)]
    elif scenario['classifier'] == 'svm':
        classifier = sklearn_SVC(kernel='rbf', C=10, gamma=0.01)
        sys.stdout.write('TRAINING SVM\n')
        # cutoff表示截止值，如果大于该值表明属于1，否则属于0（1表示是恶意的，0表示是良性的）
        cutoff = [0.0]
    
    # Load the required dataset and train the model
    # 将使用的数据集转换成数组  X特征  y标签
    X, y, _ = datasets.csv2numpy(scenario['training'])
    # 根据给定的训练数据拟合SVM模型
    classifier.fit(X, y)
    
    # Evaluate the model on the training dataset
    # decision_function：样本X到分离超平面的距离
    # 在二分类的情况下，分类模型的decision_function返回结果的形状与样本数量相同，且返回结果的数值表示模型预测样本属于positive正样本的可信度。并且，二分类情况下classes_中的第一个标签代表是负样本，第二个标签代表正样本。
    y_pred = classifier.decision_function(X)
    sys.stdout.write('Performance on training data:\n')
    # y真实结果  y_pred预测结果
    # 打印结果 包括accuracy and confusion matrix（评测模型的指标，包括tp\fp\tn\fn）
    utility.print_stats_cutoff(y, y_pred, cutoff)
    
    # Save the model in the corresponding file
    classifier.save_model(scenario['model'])

def _attack_files_missing(attack_files):
    sys.stderr.write('Unable to locate list of attack files {}. '
                     .format(attack_files))
    sys.stderr.write(('Please list the paths to your attack files in '
                      'this file, one per line.\n'))
    sys.exit()

def _initialize():
    '''
    Assembles missing datasets and learns missing models. 
    '''
    # 组装缺失的数据集并学习缺失的模型。
    def merge_CSVs(csv1, csv2, out):
        '''
        Merges two CSV files into out. Skips any header or comment 
        lines in the second file.
        '''
        # 将两个CSV文件合并到输出中。跳过第二个文件中的任何标题行或注释行
        with open(out, 'wb+') as f:
            # Copy csv1
            f.write(open(csv1).read())
            # Skip junk in csv2
            with open(csv2) as csv2in:
                l = 'a'
                while l:
                    l = csv2in.readline()
                    if l and l[:4].lower() in ('true', 'fals'):
                        f.write(l)

    # 如果contagio.csv数据集不存在就组装contagio.csv数据集
    # contagio.csv = contagio-ben.csv + contagio-mal.csv
    if not path.exists('I:\\mimicus\\mimicus-master\\data\\contagio.csv'):
        print 'Creating the contagio dataset...'
        # merge_CSVs(config.get('datasets', 'contagio_ben'),
        #            config.get('datasets', 'contagio_mal'),
        #            config.get('datasets', 'contagio'))
        merge_CSVs('I:\\mimicus\\mimicus-master\\data\\contagio-ben.csv','I:\\mimicus\\mimicus-master\\data\\contagio-mal.csv',
                   'I:\\mimicus\\mimicus-master\\data\\contagio.csv')

    # 如果contagio_full.csv数据集不存在就组装contagio_full.csv数据集
    # contagio_full.csv = contagio.csv + contagio-nopdfrate.csv
    if not path.exists('I:\\mimicus\\mimicus-master\\data\\contagio_full.csv'):
        print 'Creating the contagio-full dataset...'
        # merge_CSVs(config.get('datasets', 'contagio'),
        #            config.get('datasets', 'contagio_nopdfrate'),
        #            config.get('datasets', 'contagio_full'))
        merge_CSVs('I:\\mimicus\\mimicus-master\\data\\contagio.csv', 'I:\\mimicus\\mimicus-master\\data\\contagio-nopdfrate.csv',
                   'I:\\mimicus\\mimicus-master\\data\\contagio_full.csv')
    # 获取一个标准化工具
    standardize_csv = datasets.standardize_csv
    
    if not path.exists('I:\\mimicus\\mimicus-master\\data\\contagio.scaler'):
        print 'Creating the contagio-scaled-full dataset...'
        # scaler = standardize_csv(config.get('datasets', 'contagio_full'),
        #                          config.get('datasets', 'contagio_scaled_full'))
        # pickle.dump(scaler, open(config.get('datasets', 'contagio_scaler'),
        #                          'wb+'))
        # contagio_full.csv经过标准化之后的数据存放在contagio-scaled-full.csv
        scaler = standardize_csv('I:\\mimicus\\mimicus-master\\data\\contagio_full.csv',
                                 'I:\\mimicus\\mimicus-master\\data\\contagio-scaled-full.csv')
        # pickle.dump(scaler, open(config.get('datasets', 'contagio_scaler'),
        #                          'wb+'))
        # pickle.dump 将scaler序列化之后写入contagio.scaler中
        pickle.dump(scaler, open('I:\\mimicus\\mimicus-master\\data\\contagio.scaler','wb+'))
    
    if not path.exists('I:\\mimicus\\mimicus-master\\data\\contagio-scaled.csv'):
        print 'Creating the contagio-scaled dataset...'
        standardize_csv('I:\\mimicus\\mimicus-master\\data\\contagio.csv',
                                    'I:\\mimicus\\mimicus-master\\data\\contagio-scaled.csv',
                                    scaler)
    
    if not path.exists('I:\\mimicus\\mimicus-master\\data\\contagio-test.csv'):
        print 'Creating the contagio-test dataset...'
        shutil.copy('I:\\mimicus\\mimicus-master\\data\\contagio-nopdfrate.csv',
                    'I:\\mimicus\\mimicus-master\\data\\contagio-test.csv')
    
    if not path.exists('I:\\mimicus\\mimicus-master\\data\\contagio-scaled-test.csv'):
        print 'Creating the contagio-scaled-test dataset...'
        standardize_csv('I:\\mimicus\\mimicus-master\\data\\contagio-test.csv',
                        'I:\\mimicus\\mimicus-master\\data\\contagio-scaled-test.csv',
                        scaler)
    
    if not path.exists('I:\\mimicus\\mimicus-master\\data\\surrogate.csv'):
        print 'Creating the surrogate dataset...'
        merge_CSVs('I:\\mimicus\\mimicus-master\\data\\google-ben.csv',
                   'I:\\mimicus\\mimicus-master\\data\\virustotal-mal.csv',
                   'I:\\mimicus\\mimicus-master\\data\\surrogate.csv')
    
    if not path.exists('I:\\mimicus\\mimicus-master\\data\\surrogate-scaled.csv'):
        print 'Creating the surrogate-scaled dataset...'
        standardize_csv('I:\\mimicus\\mimicus-master\\data\\surrogate.csv',
                        'I:\\mimicus\\mimicus-master\\data\\surrogate-scaled.csv',
                        scaler)

    # 获取几个模型
    _learn_model('F')
    _learn_model('FC')
    _learn_model('FT')
    _learn_model('FTC')
    
    # utility.mkdir_p(config.get('results', 'F_gdkde'))
    # utility.mkdir_p(config.get('results', 'F_mimicry'))
    # utility.mkdir_p(config.get('results', 'FC_mimicry'))
    # utility.mkdir_p(config.get('results', 'FT_gdkde'))
    # utility.mkdir_p(config.get('results', 'FT_mimicry'))
    # utility.mkdir_p(config.get('results', 'FTC_mimicry'))
    utility.mkdir_p('I:\\mimicus\\mimicus-master\\results\\F_gdkde')
    utility.mkdir_p('I:\\mimicus\\mimicus-master\\results\\F_mimicry')
    utility.mkdir_p('I:\\mimicus\\mimicus-master\\results\\FC_mimicry')
    utility.mkdir_p('I:\\mimicus\\mimicus-master\\results\\FT_gdkde')
    utility.mkdir_p('I:\\mimicus\\mimicus-master\\results\\FT_mimicry')
    utility.mkdir_p('I:\\mimicus\\mimicus-master\\results\\FTC_mimicry')

def _gdkde_wrapper(ntuple):
    '''
    A helper function to parallelize calls to gdkde().
    '''
    try:
        return gdkde(*ntuple)
    except Exception as e:
        return e

def attack_gdkde(scenario_name, plot=False):
    '''
    Invokes the GD-KDE attack for the given scenario and saves the 
    resulting attack files in the location specified by the 
    configuration file. If plot evaluates to True, saves the resulting 
    plot into the specified file, otherwise shows the plot in a window. 
    '''
    print 'Running the GD-KDE attack...'
    _initialize()
    scenario = _scenarios[scenario_name]
    # output_dir = config.get('results', '{}_gdkde'.format(scenario_name))
    output_dir = 'I:\\mimicus\\mimicus-master\\results\\%s_gdkde',scenario_name
    # Make results reproducible
    random.seed(0)
    # Load and print malicious files
    # wolves = config.get('experiments', 'contagio_attack_pdfs')
    wolves = 'I:\\mimicus\\mimicus-master/data/test.list'
    if not path.exists(wolves):
        _attack_files_missing(wolves)
    # print 'Loading attack samples from "{}"'.format(wolves)
    malicious = utility.get_pdfs(wolves)
    if not malicious:
        _attack_files_missing(wolves)
    
    # Load an SVM trained with scaled data
    scaler = pickle.load(open(
                        'I:\\mimicus\\mimicus-master\\data\\contagio.scaler'))
    print 'Using scaler'
    svm = sklearn_SVC()
    print 'Loading model from "{}"'.format(scenario['model'])
    svm.load_model(scenario['model'])
    
    # Load the training data used for kernel density estimation
    print 'Loading dataset from file "{}"'.format(scenario['training'])
    X_train, y_train, _ = datasets.csv2numpy(scenario['training'])
    # Subsample for faster execution
    ind_sample = random.sample(range(len(y_train)), 500)
    X_train = X_train[ind_sample, :]
    y_train = y_train[ind_sample]
    
    # Set parameters
    kde_reg = 10
    kde_width = 50
    step = 1
    max_iter = 50
    
    # Set up multiprocessing
    pool = multiprocessing.Pool()
    pargs = [(svm, fname, scaler, X_train, y_train, kde_reg, 
                  kde_width, step, max_iter, False) for fname in malicious]
    
    if plot:
        pyplot.figure(1)
    print 'Running the attack...'
    for res, oldf in zip(pool.imap(_gdkde_wrapper, pargs), malicious):
        if isinstance(res, Exception):
            print res
            continue
        (_, fseq, _, _, attack_file) = res
        print 'Processing file "{}":'.format(oldf)
        print '  scores: {}'.format(', '.join([str(s) for s in fseq]))
        print 'Result: "{}"'.format(attack_file)
        if path.dirname(attack_file) != output_dir:
            shutil.move(attack_file, output_dir)
        if plot:
            pyplot.plot(fseq, label=oldf)
    
    print 'Saved resulting attack files to {}'.format(output_dir)
    
    if plot:
        pyplot.title('GD-KDE attack')
        axes = pyplot.axes()
        axes.set_xlabel('Iterations')
        axes.set_xlim(0, max_iter + 1)
        axes.set_ylabel('SVM score')
        axes.yaxis.grid()
        fig = pyplot.gcf()
        fig.set_size_inches(6, 4.5)
        fig.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.96)
        if plot == 'show':
            pyplot.show()
        else:
            pyplot.savefig(plot, dpi=300)
            print 'Saved plot to file {}'.format(plot)

def _mimicry_wrap(ntuple):
    '''
    A helper function to parallelize calls to mimicry().
    '''
    # 一个辅助函数，用于并行化对mimicy（）的调用
    try:
        return mimicry(*ntuple)
    except Exception as e:
        return e

def attack_mimicry(scenario_name, plot=False):
    '''
    Invokes the mimcry attack for the given scenario and saves the 
    resulting attack files in the location specified by the 
    configuration file. If plot evaluates to True, saves the resulting 
    plot into the specified file, otherwise shows the plot in a window. 
    '''
    # 调用给定场景的mimcry攻击，并将生成的攻击文件保存在配置文件指定的位置。
    # 如果绘图计算结果为True，则将生成的绘图保存到指定文件中，否则在窗口中显示该绘图。
    print 'Running the mimicry attack...'
    # 初始化(包括数据集的处理、模型的训练)
    _initialize()
    scenario = _scenarios[scenario_name]
    # output_dir = config.get('results', '{}_mimicry'.format(scenario_name))
    output_dir = 'I:\\mimicus\\mimicus-master\\results\\%s_mimicry',scenario_name
    # Make results reproducible
    # 使结果可复现
    random.seed(0)
    # Load benign files
    # 加载良性文件
    print 'Loading attack targets from file "{}"'.format(scenario['targets'])
    # 将文件转换成数组  返回每行的特征的向量表示  标签  文件名列表
    target_vectors, _, target_paths = datasets.csv2numpy(scenario['targets'])
    # 压缩文件名和对应的特征向量
    targets = zip(target_paths, target_vectors)
    # Load malicious files
    # wolves = config.get('experiments', 'contagio_attack_pdfs')
    # 恶意文件列表
    wolves = 'I:\\mimicus\\mimicus-master/data/test.list'
    if not path.exists(wolves):
        _attack_files_missing(wolves)
    print 'Loading attack samples from file "{}"'.format(wolves)

    # 根据提供的文件名称找到对应的PDF文件  malicious是一个列表
    malicious = sorted(utility.get_pdfs(wolves))
    if not malicious:
        _attack_files_missing(wolves)
    
    # Set up classifier
    classifier = 0
    if scenario['classifier'] == 'rf':
        classifier = RandomForest()
        print 'ATTACKING RANDOM FOREST'
    elif scenario['classifier'] == 'svm':
        classifier = sklearn_SVC()
        print 'ATTACKING SVM'
    print 'Loading model from "{}"'.format(scenario['model'])
    classifier.load_model(scenario['model'])
    
    # Standardize data points if necessary
    # 如果模型中包含'scaled'，就使用contagio.scaler
    scaler = None
    if 'scaled' in scenario['model']:
        scaler = pickle.load(open('I:\\mimicus\\mimicus-master\\data\\contagio.scaler'))
        print 'Using scaler'
    
    # Set up multiprocessing
    # Multiprocessing.Pool可以提供指定数量的进程供用户调用，当有新的请求提交到pool中时，
    # 如果池还没有满，那么就会创建一个新的进程用来执行该请求；但如果池中的进程数已经达到规定最大值，那么该请求就会等待，直到池中有进程结束，才会创建新的进程来执行它。
    # Pool类用于需要执行的目标很多，而手动限制进程数量又太繁琐时
    pool = multiprocessing.Pool()
    # mal:恶意的pdf     targets：压缩后的文件名和对应的特征向量 classifier：分类器   scaler：表明是否需要标准化
    pargs = [(mal, targets, classifier, scaler) for mal in malicious]

    if plot:
        pyplot.figure(1)
    # 开始
    print 'Running the attack...'
    # print pool.imap(_mimicry_wrap, pargs)
    # _mimicry_wrap返回 best_ben_path, best_mimic_path, best_mimic_score, wolf_score
    for wolf_path, res in zip(malicious, pool.imap(_mimicry_wrap, pargs)):

        if isinstance(res, Exception):
            print res
            continue
        # best_ben_path, best_mimic_path, best_mimic_score, wolf_score
        (target_path, mimic_path, mimic_score, wolf_score) = res
        print 'Modifying {p} [{s}]:'.format(p=wolf_path, s=wolf_score)
        print '  BEST: {p} [{s}]'.format(p=target_path, s=mimic_score)
        if path.dirname(mimic_path) != output_dir:
            print '  Moving best to {}\n'.format(path.join(output_dir, 
                                                 path.basename(mimic_path)))
            shutil.move(mimic_path, output_dir)
        if plot:
            pyplot.plot([wolf_score, mimic_score])
    
    print 'Saved resulting attack files to {}'.format(output_dir)
    
    if plot:
        pyplot.title('Mimicry attack')
        axes = pyplot.axes()
        axes.set_xlabel('Iterations')
        axes.set_ylabel('Classifier score')
        axes.yaxis.grid()
        fig = pyplot.gcf()
        fig.set_size_inches(6, 4.5)
        fig.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.96)
        if plot == 'show':
            pyplot.show()
        else:
            pyplot.savefig(plot, dpi=300)
            print 'Saved plot to file {}'.format(plot)
