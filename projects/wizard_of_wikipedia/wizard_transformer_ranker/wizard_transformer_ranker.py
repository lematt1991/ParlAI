# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.transformer.transformer import TransformerRankerAgent
from parlai.core.torch_agent import TorchAgent

import numpy as np
import torch


class WizardTransformerRankerAgent(TransformerRankerAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(WizardTransformerRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Wizard Transformer Ranker Arguments')
        agent.add_argument(
            '--use-knowledge', type='bool', default=True,
            help='use knowledge field instead of personas'
        )
        agent.add_argument(
            '--knowledge-dropout', type=float, default=0.7,
            help='dropout some knowledge during training'
        )
        agent.add_argument(
            '--chosen-sentence', type='bool', default=False,
            help='instead of using all knowledge, use gold'
                 'label, i.e. the chosen sentence'
        )
        agent.add_argument(
            '--knowledge-truncate', type=int, default=50,
            help='truncate knowledge to this length'
        )
        argparser.set_defaults(
            learningrate=0.0008,
            eval_candidates='inline',
            candidates='batch',
            lr_factor=1,
            delimiter=' ',
            add_p1_after_newln=False,
        )
        return agent

    def __init__(self, opt, shared=None):
        """Set up model."""

        super().__init__(opt, shared)
        self.use_knowledge = opt.get('use_knowledge', False)
        if self.use_knowledge:
            self.opt['use_memories'] = True
        self.chosen_sentence = (opt.get('chosen_sentence', False) and
                                self.use_knowledge)
        self.knowledge_dropout = opt.get('knowledge_dropout', 0)
        self.knowledge_truncate = opt.get('knowledge_truncate', 50)

    def _vectorize_memories(self, observation):
        """Override abstract method from TransformerRankerAgent to use
        knowledge field as memories."""

        if not self.use_knowledge:
            return observation

        observation['memory_vecs'] = []

        checked = observation.get('checked_sentence', '')
        if observation.get('knowledge'):
            knowledge = observation['knowledge'].split('\n')[:-1]
        else:
            knowledge = []

        to_vectorize = []
        if checked and self.chosen_sentence:
            # if `self.chosen_sentence` is True, only keep golden knowledge
            to_vectorize = [checked]
        elif (self.knowledge_dropout == 0 or
                observation.get('eval_labels') is not None) and knowledge:
            # during evaluation we use all of the knowledge
            to_vectorize = knowledge
        elif knowledge:
            for line in knowledge:
                if checked and checked in line:
                    # make sure we keep the chosen sentence
                    keep = 1
                else:
                    # dropout knowledge
                    keep = np.random.binomial(1, 1 - self.knowledge_dropout)
                if keep:
                    to_vectorize.append(line)

        # vectorize knowledge
        observation['memory_vecs'] = [
            self._vectorize_text(line, truncate=self.knowledge_truncate) for
            line in to_vectorize
        ]
        return observation