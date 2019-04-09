#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDialogTeacher
import parlai.core.agents as core_agents
from .build import build
from glob import glob
import copy
import os


class SingleFileTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'tom:tom:orig:easy:noisy:qa21_train.txt').split(':')
        build(opt, task[:-1])
        opt['datafile'] = os.path.join(opt['datapath'], 'tom', '/'.join(task[2:]))
        opt['cands_datafile'] = os.path.join(opt['datapath'], 'tom', '/'.join(task[2:-1]),
            'qa21_task_AB_train.txt')
        super().__init__(opt, shared)

class TomTeacher(core_agents.MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        task = opt.get('task', 'tom:tom:orig:easy:noisy').split(':')
        build(opt, task)
        base = 'tom:single_file:' + ':'.join(task[2:])
        pth = os.path.join(opt['datapath'], 'tom', '/'.join(task[2:]))
        if opt['datatype'].split(':')[0] == 'train':
            opt['task'] = base + ':qa21_task_AB_train.txt'
        elif opt['datatype'].split(':')[0] == 'valid':
            files = glob(f'{pth}/*val_test.txt')
            opt['task'] = ','.join([base + f':{os.path.basename(f)}' for f in files])
        elif opt['datatype'].split(':')[0] == 'test':
            files = glob(f'{pth}/*test_test.txt')
            opt['task'] = ','.join([base + f':{os.path.basename(f)}' for f in files])
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(TomTeacher):
    pass
