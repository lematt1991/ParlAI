#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from subprocess import call
from glob import iglob
from collections import defaultdict
import re


CONFIGS = {
    'orig:with_train_noise:easy:noisy': ['-ptn', '0.1', '-easy', '-tn', 'true'],
    'orig:with_train_noise:hard:noisy': ['-ptn', '0.1', '-tn', 'true'],
    'orig:with_train_noise:easy:clean': ['-ptn', '0', '-easy', '-tn', 'true'],
    'orig:with_train_noise:easy:clean': ['-ptn', '0', '-easy', '-tn', 'true'],
    'orig:without_train_noise:easy:noisy': ['-ptn', '0.1', '-easy', '-tn', 'false'],
    'orig:without_train_noise:hard:noisy': ['-ptn', '0.1', '-tn', 'false'],
    'orig:without_train_noise:easy:clean': ['-ptn', '0', '-easy', '-tn', 'false'],
    'orig:without_train_noise:easy:clean': ['-ptn', '0', '-easy', '-tn', 'false'],
}


def build(opt, task):
    base_path = os.path.join(opt['datapath'], 'tom')
    dpath = os.path.join(opt['datapath'], 'tom', '/'.join(task[2:]))
    version = 'None'
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        ds_repo = os.path.join(base_path, 'tom-qa-dataset')
        repo_url = 'git@github.com:lematt1991/tom-qa-dataset.git'
        if not os.path.exists(ds_repo):
            branch = 'dev'
            # import pdb; pdb.set_trace()
            ret = call(['git', 'clone', repo_url, ds_repo])
            assert ret == 0, "Failed to clone tom-qa-dataset repo!"
            ret = call(['git', 'checkout', '-b', branch], cwd=ds_repo)
            assert ret == 0, 'Failed to checkout branch "{branch}"'
            call(['git', 'pull', 'origin', branch], cwd=ds_repo)
            ret = call(['python', 'create_world.py'], cwd=ds_repo)

        key = ':'.join(task[2:])
        assert key in CONFIGS, "Couldn't recognize task type!"
        args = ['-w', 'world_large.txt', '-n', '1000', '-o', dpath] + CONFIGS[key]
        ret = call(['python', 'generate_tasks.py'] + args, cwd=ds_repo)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
