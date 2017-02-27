#!/usr/bin/env python
# coding=utf-8
#

"""
Training sklearn models with flect.model (helper functions to bin/train.py).
"""



import re
import os
import pickle
import marshal
import types

from flect.config import Config
from flect.model import Model, SplitModel
from flect.cluster import Job
from flect.logf import log_info

__author__ = "Ondřej Dušek"
__date__ = "2012"


MEMORY = 16


def append_name(file_name, suffix):
    """\
    Append a suffix to a file name (before the extension).
    """
    if file_name.endswith('.gz'):
        file_name, gz = os.path.splitext(file_name)
    else:
        gz = ''
    name, ext = os.path.splitext(file_name)
    return name + '-' + suffix + ext + gz


def create_job(config, name, work_dir, train_file, model_file,
               test_file, classif_file, memory=MEMORY):
    """\
    Save configuration to a pickle file, create a corresponding cluster job
    and run it.
    """
    train_file = os.path.abspath(train_file)
    cfg_file = append_name('config.pickle', name)
    model_file = append_name(model_file, name)
    if test_file is not None and classif_file is not None:
        test_file = os.path.abspath(test_file)
        classif_file = append_name(classif_file, name)
        test_str = "'{0}', '{1}'".format(test_file, classif_file)
    else:
        test_str = 'None, None'
    # save unfolded config file
    fh = open(os.path.join(work_dir, cfg_file), mode='wb')
    marshal_lambda(config, 'filter_attr')
    marshal_lambda(config, 'postprocess')
    pickle.dump(config, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()
    # create the training job
    job = Job(name=name, work_dir=work_dir)
    job.header = "from flect.experiment.train_model import run_training\n"
    job.code = "run_training('{0}', '{1}',".format(work_dir, cfg_file) + \
            "'{0}', '{1}', {2})\n".format(train_file, model_file, test_str)
    job.submit(memory=memory)
    print('Job', job, 'submitted.')
    # print job.get_script_text()


def marshal_lambda(config, key):
    """\
    Marshal a lambda function under the given key in the config.
    """
    if key in config:
        code = config[key].__code__
        config[key] = marshal.dumps(code)


def demarshal_lambda(config, key):
    """\
    De-marshal a lambda function under the given key in the config.
    """
    if key in config:
        code = marshal.loads(config[key])
        config[key] = types.FunctionType(code, globals())


def run_training(work_dir, config_file, train_file, model_file,
                 test_file=None, classif_file=None, memory=MEMORY,
                 name='train'):
    """\
    Run the model training.
    """
    # initialization from the configuration file
    _, ext = os.path.splitext(config_file)
    # load configuration from a pickle (we're already in the working directory)
    if ext == '.pickle':
        fh = open(config_file, mode='rb')
        cfg = pickle.load(fh)
        fh.close()
        demarshal_lambda(cfg, 'filter_attr')
        demarshal_lambda(cfg, 'postprocess')
    # load by running Python code (make paths relative to working directory)
    else:
        config_file = os.path.join(work_dir, config_file)
        cfg = Config(config_file)
    # training
    if cfg.get('unfold_pattern'):
        pattern = cfg['unfold_pattern']
        del cfg['unfold_pattern']
        unfold_key = cfg.get('unfold_key', 'unfold_key')
        cfgs = cfg.unfold_lists(pattern, unfold_key)
        for cfg in cfgs:
            key = re.sub(r'[^A-Za-z0-9_]', '', cfg[unfold_key])
            create_job(cfg, name + '-' + key, work_dir, train_file, model_file,
                       test_file, classif_file, memory)
        return
    if cfg.get('divide_func'):
        model = SplitModel(cfg)
        model.train(train_file, work_dir, memory)
    else:
        model = Model(cfg)
        model.train(train_file)
    # evaluation
    if test_file is not None and classif_file is not None:
        if ext != '.pickle':  # this means we're not in the working directory
            classif_file = os.path.join(work_dir, classif_file)
        log_info('Evaluation on file: ' + test_file)
        score = model.evaluate(test_file, classif_file=classif_file)
        log_info('Score: ' + str(score))
    # save the model
    if ext != '.pickle':  # we need to make the path relative to work_dir
        model_file = os.path.join(work_dir, model_file)
    model.save_to_file(model_file)
