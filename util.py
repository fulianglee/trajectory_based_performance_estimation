"""
DevOp codes
"""
import numpy as np
import copy
import torch
import os
import time
from torch.nn.modules.loss import _WeightedLoss, _Loss
import matplotlib.pyplot as plt
import scipy.stats


class WarmUPRate:
    """ Learning rate scheduler
    usage:
    >>> warmup = WarmUPRate(factor=2, model_bandwidth=128, warmup_step=4000)
    >>> warmup.set_lr(model.optimizer, 3000)  # lr = 0.000741158826601964
    """

    def __init__(self, factor, model_bandwidth, warmup_step):
        self.factor = factor
        self.model_bandwidth = model_bandwidth
        self.warmup_step = warmup_step

    def cal_lr(self, step: int):
        assert step >= 0
        f, b, w = self.factor, self.model_bandwidth, self.warmup_step
        return f * (b ** (-0.5) * min(step ** (-0.5), step * w ** (-1.5)))

    def set_lr(self, optimizer, step: int):
        lr = self.cal_lr(step)
        for p in optimizer.param_groups:
            p['lr'] = lr

    def plot(self, max_step=20000):
        plt.plot([self.cal_lr(step) for step in range(1, max_step)])


class Timer:
    def __init__(self):
        ...

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = time.time()
        print(f'time spent={self.t1 - self.t0:.4f} sec')


class AverageMeter:
    def __init__(self, max_len=99):
        # self.array = np.full([max_len], 1.0)
        self.array = np.array([])
        self.max_len = max_len
        self.index = 0

    def log(self, v):
        # if self.index == 0:
        #     self.array.fill(v)
        if self.index < self.max_len:
            self.array = np.append(self.array, v)
        self.array[self.index % self.max_len] = v
        self.index += 1

    def __call__(self, v):
        self.log(v)

    @property
    def value(self):
        return self.array.mean() if self.array.size != 0 else 0

    @property
    def std(self):
        return self.array.std() if self.array.size != 0 else 0

    def __repr__(self):
        return self.__class__.__name__ + f'(value={self.value:.4f}, std={self.std:.2f})'


class OptimizerContext:
    def __init__(self, optimizer: torch.optim.Adam):
        self.optimizer = optimizer

    def __enter__(self):
        self.optimizer.zero_grad()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.optimizer.step()


def get_nn_params(model, model_name=None, print_out=False):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    if print_out:
        print(f'{model_name if model_name is not None else "Model"} initiated with number of parameters: {pp}')
    return pp


def max_grad_params(model: torch.nn.Module, print_out=False):
    max_grad = 0.
    for p in list(model.parameters()):
        max_grad = max(p.grad.max().abs().item(), max_grad)
    if print_out:
        print(f'model max_grad={max_grad}')
    return max_grad


def add_util(cls, instance, param=None, optimizer='Adam', save_pth='check_points'):
    """
    Adding saving/loading capacity to module:
    >>> instance.save('file.name.pt')
    >>> instance.load('file.name.pt')
    Adding optimization (default to Adam) capacity to module that
    remove the need to zeroing grad:
    >>> with instance.optimize_c():
    >>>    loss.backward()
    >>>    if i % epoch_size == 0: model.scheduler.step(avm.value)
    """

    assert not hasattr(instance, 'save_path')
    assert not hasattr(instance, 'optimizer')

    instance.save_path = save_pth

    param = instance.parameters() if param is None else param

    instance.optimizer = torch.optim.Adam(param)

    print(f'model optimizer is {instance.optimizer}')
    instance.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer,
                                                                    factor=0.5,
                                                                    patience=50, )
    print(f'model scheduler is {instance.scheduler} \n'
          f'usage snippet: \n'
          f'>>> with instance.optimize_c(): \n'
          f'>>>    loss.backward() \n'
          f'>>>    if i % epoch_size == 0: model.scheduler.step(avm.value)')

    instance.optim_cont = OptimizerContext(instance.optimizer)

    def save(self, name, verbose=True):
        pth = os.path.join(self.save_path, name)
        torch.save(self.state_dict(), pth)
        if verbose:
            print(f'{self.save_path}/{name} saved')

    setattr(cls, 'save', save)

    def load(self, name):
        pth = os.path.join(self.save_path, name)
        self.load_state_dict(torch.load(pth))
        self.eval()
        print('load successful')
        return self

    setattr(cls, 'load', load)

    def optimize_c(self):
        return self.optim_cont

    setattr(cls, 'optimize_c', optimize_c)


def add_auto_save(cls, instance, mode='min', n_delay=500):
    '''Example:
        >>> model = ...
        >>>model.set_auto_save_name('name').set_auto_save_delay(1000).toggle_auto_save()
        >>>'model save to path: path/name/pt'
        >>>loss = ...
        >>>model.save_by_score(loss)
        '''

    assert hasattr(instance, 'save_path')
    assert mode == 'min' or mode == 'max'
    assert not hasattr(instance, '_model_best_score')
    assert not hasattr(instance, '_auto_save_name')
    assert not hasattr(instance, '_auto_save_toggle')
    assert not hasattr(instance, '_auto_save_first_n_delay')

    instance._auto_save_toggle = False
    instance._auto_save_first_n_delay = n_delay
    if mode == 'min':
        instance._model_best_score = float('inf')
    else:
        instance._model_best_score = float('-inf')

    def save_by_score(self, score):
        if self._auto_save_first_n_delay > 0:
            self._auto_save_first_n_delay -= 1
            self._model_best_score = score
            return

        if self._auto_save_toggle is True:
            if mode == 'min':
                if score < self._model_best_score:
                    self._model_best_score = score
                    print(f'smallest score: {score} ', end='')
                    self.save(self._auto_save_name)
            elif mode == 'max':
                if score > self._model_best_score:
                    self._model_best_score = score
                    print(f'biggest score: {score} ', end='')
                    self.save(self._auto_save_name)

    def set_auto_save_name(self, name: str):
        self._auto_save_name = name
        print(f'set save name to {self._auto_save_name}')
        return self

    def set_auto_save_delay(self, n: int):
        self._auto_save_first_n_delay = n
        print(f'delay for {self._auto_save_first_n_delay} step')
        return self

    def toggle_auto_save(self):
        if self._auto_save_name == None and self._auto_save_toggle is False:
            raise Exception('Need a auto save name end with .pt! ')

        self._auto_save_toggle = not self._auto_save_toggle
        if self._auto_save_toggle:
            print(f'Auto saving is on! will save model to {self.save_path}/{self._auto_save_name}, start after '
                  f'{self._auto_save_first_n_delay} step if the given score to model.save_by_score(score) is the {mode}imum')
        else:
            print(f'Auto save is off')

        return self

    setattr(cls, 'save_by_score', save_by_score)
    setattr(cls, 'set_auto_save_name', set_auto_save_name)
    setattr(cls, 'set_auto_save_delay', set_auto_save_delay)
    setattr(cls, 'toggle_auto_save', toggle_auto_save)


def add_checkpoint_util(cls, instance, mode='descend', auto_checkpoint_on=False):
    """
        >>>loss = torch.Tensor
        >>>model.auto_checkpoint_on = True
        >>>model.check_point(loss.tolist(), 'checkpoint_file_name.pt', verbose=True)
    """
    assert mode == 'descend' or mode == 'ascend'
    assert hasattr(instance, 'save_path')
    assert not hasattr(instance, 'auto_checkpoint_on')
    assert not hasattr(instance, '_model_best_score')

    instance._model_best_score = float('inf') if mode == 'descend' else float('-inf')
    instance.auto_checkpoint_on = auto_checkpoint_on

    def check_point(self, score, name, verbose=False):
        do_update = True if any([
            all([mode == 'descend', score < self._model_best_score]),
            all([mode == 'ascend', score > self._model_best_score]),
        ]) else False
        if do_update:
            self._model_best_score = score
            if self.auto_checkpoint_on:
                self.save(name, verbose=verbose)

    setattr(cls, 'check_point', check_point)


def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypeError:
            print(f"{func.__name__} only takes numbers as the argument")

    return inner_function


class SmoothNLLLoss(_WeightedLoss):
    """
    >>>log_softmax_pred = torch.randn(3, 5, requires_grad=True)
    >>>target = torch.empty(3, dtype=torch.long).random_(5)
    >>>criteria = SmoothNLLLoss()
    >>>loss = loss(log_softmax_pred, target)
    >>>loss.backward()
    """

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', smoothing=0.0):
        super(SmoothNLLLoss, self).__init__(weight=weight, reduction=reduction)
        assert 0 <= smoothing < 1
        self.ignore_index = ignore_index
        self.smoothing = smoothing

    def forward(self, log_softmax_pred: torch.Tensor, target):
        # Version 1:
        # targets = SmoothNLLLoss._smooth_one_hot(targets, inputs.size(-1),
        #                                         self.smoothing)
        # lsm = F.log_softmax(inputs, -1)
        #
        # if self.weight is not None:
        #     lsm = lsm * self.weight.unsqueeze(0)
        #
        # loss = -(targets * lsm).sum(-1)
        #
        # if self.reduction == 'sum':
        #     loss = loss.sum()
        # elif self.reduction == 'mean':
        #     loss = loss.mean()

        # Version 2:
        # def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
        #     ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        #
        #     gold = gold.contiguous().view(-1)
        #
        #     if smoothing:
        #         eps = 0.1
        #         n_class = pred.size(1)
        #
        #         one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        #         one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        #         log_prb = F.log_softmax(pred, dim=1)
        #
        #         non_pad_mask = gold.ne(trg_pad_idx)
        #         loss = -(one_hot * log_prb).sum(dim=1)
        #         loss = loss.masked_select(non_pad_mask).sum()  # average later
        #     else:
        #         loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
        #     return loss

        smoothing = self.smoothing
        n_class = log_softmax_pred.size(-1)

        one_hot = torch.zeros_like(log_softmax_pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - smoothing) + (1 + smoothing) * smoothing / (n_class - 1)

        loss = -(one_hot * log_softmax_pred).sum(dim=-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class SmoothDistNLLLoss(_Loss):
    def __init__(self, size, spread):
        super(SmoothDistNLLLoss, self).__init__()
        self.size = size
        self.spread = spread
        self.probs = self.get_prob_spreads()
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')

    def get_prob_spreads(self):
        cls_ls = [i for i in range(self.size)]
        probs_ls = []
        for idx in cls_ls:
            max_dist = max([abs(i - idx) for i in cls_ls])
            probs = {i: 0 for i in cls_ls}
            probs.update([(idx, 1 - self.spread)])
            for dist in range(max_dist, 0, -1):
                indice = [i for i in cls_ls if abs(i - idx) <= dist]
                probs.update([(i, probs[i] + (self.spread / max_dist) / len(indice)) for i in indice])
            probs_ls.append(probs)
        return torch.tensor([list(probs.values()) for probs in probs_ls])

    def forward(self, log_softmax_pred: torch.FloatTensor, target: torch.LongTensor):
        return self.criterion(log_softmax_pred, torch.stack([self.probs[i] for i in target], 0))


class SmoothDistNLLLossV2(_Loss):
    def __init__(self, size, base_spread=0.1, max_sigma=20.0, device=None):
        super(SmoothDistNLLLossV2, self).__init__()
        self.size = size
        self.base_spread = base_spread
        self.MAX_SIGMA = max_sigma
        self.probs = self.get_prob_spreads().to('cpu' if device is None else device)
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')

    def get_prob_spreads(self):
        dist = scipy.stats.norm(0, 1)
        cls_ls = [i for i in range(self.size)]
        probs_ls = []

        for idx in cls_ls:
            step_size = self.MAX_SIGMA / (self.size + 1)
            probs = [dist.cdf((cls - idx + 0.5) * step_size) - dist.cdf((cls - idx - 0.5) * step_size) for cls in
                     cls_ls]
            probs = [prob + self.base_spread / self.size for prob in probs]

            probs_ls.append(probs)
        result = torch.tensor(probs_ls)
        return result / result.sum(-1).unsqueeze(1)

    def forward(self, log_softmax_pred: torch.FloatTensor, target: torch.LongTensor):
        return self.criterion(log_softmax_pred, torch.stack([self.probs[i] for i in target], 0))


def domain_tran(input_lst: torch.Tensor, part_lst: list, padend=True):
    input_lst = copy.deepcopy(input_lst)
    assert len(part_lst) > 0
    if padend:
        part_lst = [...] + part_lst + [...]
    partitions = [(a, b) for a, b in zip(part_lst[:-1], part_lst[1:])]
    for i, (a, b) in enumerate(partitions):
        left = True if a is ... else (a <= input_lst)
        right = True if b is ... else (input_lst < b)
        slicer = (left & right) if isinstance(right, bool) else (right & left)
        input_lst[slicer] = i
    return input_lst
