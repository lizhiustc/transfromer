# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2,5)
print(embeds)
hello_idx = torch.LongTensor([word_to_ix['hello']])
print(hello_idx)
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)
print(hello_idx)
print([word_to_ix['hello']])

#   from  website     https://ptorch.com/news/11.html
