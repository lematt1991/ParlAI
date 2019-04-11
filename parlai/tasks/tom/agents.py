#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDialogTeacher
import parlai.core.agents as core_agents
from .build import build
from glob import glob
import copy
import re
import os


class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        task = opt.get('task', 'tom:task:orig:easy:with_train_noise:noisy:fb_belief_train.txt').split(':')
        build(opt, task[:-1])
        file = os.path.join(opt['datapath'], 'tom', '/'.join(task[2:]))
        opt['datafile'] = file
        opt['cands_datafile'] = re.sub('_([a-z]+).txt', '_train.txt', file)
        super().__init__(opt, shared)


class TomTeacher(core_agents.MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        if shared is None:
            opt = copy.deepcopy(opt)
            task = opt.get('task', 'tom:tom:orig:with_train_noise:easy:noisy').split(':')
            build(opt, task)
            base = 'tom:task:' + ':'.join(task[2:])
            pth = os.path.join(opt['datapath'], 'tom', '/'.join(task[2:]))
            split = opt['datatype'].split(':')[0]
            split = 'val' if split == 'valid' else split
            files = glob(os.path.join(pth, f'*{split}.txt'))
            opt['task'] = ','.join([base + f':{os.path.basename(f)}' for f in files])
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(TomTeacher):
    pass
