"""


    Domain Adaptation for Mitotic Figure Assessment
    
    Marc Aubreville, FAU Erlangen-Nürnberg
    

    For results and further information, please read our paper:
    
    Learning New Tricks from Old Dogs - Inter-Species, Inter-Tissue Domain Adaptation for Mitotic Figure Assessment
    M. Aubreville, C. Bertram, S. Jabari, C. Marzahl, R. Klopfleisch, A. Maier
    submitted for:
    Bildverarbeitung für die Medizin, 2020


    With lots of inspiration from Denis Vilar:
        https://gist.github.com/denisvlr/802f980ff6b6296beaaea1a592724a51

"""
from fastai import *
from fastai.vision import *
from fastai.data_block import _maybe_squeeze


class NanLabelImageList(ImageList):
    def label_from_df(self, cols:IntsOrStrs=1, label_cls:Callable=None, **kwargs):
        "Label `self.items` from the values in `cols` in `self.inner_df`."
        labels = self.inner_df.iloc[:,df_names_to_idx(cols, self.inner_df)]

        if is_listy(cols) and len(cols) > 1 and (label_cls is None or label_cls == MultiCategoryList):
            new_kwargs,label_cls = dict(one_hot=True, classes= cols),MultiCategoryList
            kwargs = {**new_kwargs, **kwargs}
        return self._label_from_list(_maybe_squeeze(labels), label_cls=label_cls, **kwargs)

# Monkey patch FloatItem with a better default string formatting.
def float_str(self):
    return "{:.4g}".format(self.obj)
FloatItem.__str__ = float_str

class MultitaskItem(MixedItem):    
    def __init__(self, *args, mt_names=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.mt_names = mt_names
    
    def __repr__(self):
        return '|'.join([f'{self.mt_names[i]}:{item}' for i, item in enumerate(self.obj)])

class MultitaskItemList(MixedItemList):
    
    def __init__(self, *args, mt_names=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.mt_classes = [getattr(il, 'classes', None) for il in self.item_lists]
        self.mt_types = [type(il) for il in self.item_lists]
        self.mt_lengths = [len(i) if i else 1 for i in self.mt_classes]
        self.mt_names = mt_names
        
    def get(self, i):
        return MultitaskItem([il.get(i) for il in self.item_lists], mt_names=self.mt_names)
    
    def reconstruct(self, t_list):
        items = []
        t_list = self.unprocess_one(t_list)
        for i,t in enumerate(t_list):
            if self.mt_types[i] == CategoryList:
                items.append(Category(t, self.mt_classes[i][t]))
            elif issubclass(self.mt_types[i], FloatList):
                items.append(FloatItem(t))
        return MultitaskItem(items, mt_names=self.mt_names)
    
    def analyze_pred(self, pred, thresh:float=0.5):         
        predictions = []
        start = 0
        predictions.append(pred[0].argmax())
        predictions.append(pred[1].argmax())
        return predictions

    def unprocess_one(self, item, processor=None):
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: 
            item = _processor_unprocess_one(p, item)
        return item

def _processor_unprocess_one(self, item:Any): # TODO: global function to avoid subclassing MixedProcessor. To be cleaned.
    res = []
    for procs, i in zip(self.procs, item):
        for p in procs: 
            if hasattr(p, 'unprocess_one'):
                i = p.unprocess_one(i)
        res.append(i)
    return res

class MultitaskLabelList(LabelList):
    def get_state(self, **kwargs):
        kwargs.update({
            'mt_classes': self.mt_classes,
            'mt_types': self.mt_types,
            'mt_lengths': self.mt_lengths,
            'mt_names': self.mt_names
        })
        return super().get_state(**kwargs)

    @classmethod
    def load_state(cls, path:PathOrStr, state:dict) -> 'LabelList':
        res = super().load_state(path, state)
        res.mt_classes = state['mt_classes']
        res.mt_types = state['mt_types']
        res.mt_lengths = state['mt_lengths']
        res.mt_names = state['mt_names']
        return res
    
class MultitaskLabelLists(LabelLists):
    @classmethod
    def load_state(cls, path:PathOrStr, state:dict):
        path = Path(path)
        train_ds = MultitaskLabelList.load_state(path, state)
        valid_ds = MultitaskLabelList.load_state(path, state)
        return MultitaskLabelLists(path, train=train_ds, valid=valid_ds)

def label_from_mt_project(self, multitask_project):
    mt_train_list = MultitaskItemList(
        [task['label_lists'].train.y for task in multitask_project.values()], 
        mt_names=list(multitask_project.keys(),
                     )
    )
    mt_valid_list = MultitaskItemList(
        [task['label_lists'].valid.y for task in multitask_project.values()], 
        mt_names=list(multitask_project.keys())
    )
    
    self.train = self.train._label_list(x=self.train, y=mt_train_list)
    self.valid = self.valid._label_list(x=self.valid, y=mt_valid_list)
    self.__class__ = MultitaskLabelLists # TODO: Class morphing should be avoided, to be improved.
    self.train.__class__ = MultitaskLabelList
    self.valid.__class__ = MultitaskLabelList
    return self

def _remove_nan_values(input, target, mt_type, mt_classes):
    if mt_type == CategoryList and 'NA' in mt_classes:
        index = mt_classes.index('NA')
        nan_mask = target == index
    elif issubclass(mt_type, FloatList):
        nan_mask = (torch.isnan(target)) | (target < 0)
    else:
        return input, target
    return input[nan_mask], target[nan_mask]

class MultitaskAverageMetric(AverageMetric):
    def __init__(self, func, name=None):
        super().__init__(func)
        self.name = name # subclass uses this attribute in the __repr__ method.


def _format_metric_name(field_name, metric_func):
    return f"{field_name} {metric_func.__name__.replace('root_mean_squared_error', 'RMSE')}"


def _mt_parametrable_metric(inputs, *targets, func, data, start=0, length=1, i=0):
    input = inputs[i]
    target = targets[i]

    _remove_nan_values(input, target, data.mt_types[i], data.mt_classes[i]) # TODO: Avoid data global reference.
    
    if func.__name__ == 'root_mean_squared_error':
        processor = listify(data.y.processor)
        input = processor[0].procs[i][0].unprocess_one(input) # TODO: support multi-processors
        target = processor[0].procs[i][0].unprocess_one(target.float()) 
    return func(input, target)

def mt_metrics_generator(multitask_project, mt_lengths, data):
    metrics = []
    start = 0
    for i, ((name, task), length) in enumerate(zip(multitask_project.items(), mt_lengths)):
        metric_func = task.get('metric')
        if metric_func:
            partial_metric = partial(_mt_parametrable_metric, data=data, start=start, length=length, i=i, func=metric_func)
            metrics.append(MultitaskAverageMetric(partial_metric, _format_metric_name(name,metric_func)))
        start += length
    return metrics


def accuracy_source(input:Tensor, targs_class:Tensor, targs_domain:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs_class.shape[0]
    input = input[0].argmax(dim=-1).view(n,-1)
    targs_class = targs_class.view(n,-1)
    targetmask = targs_domain==0
    return (input[targetmask]==targs_class[targetmask]).float().mean()

def accuracy_target(input:Tensor, targs_class:Tensor, targs_domain:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs_class.shape[0]
    input = input[0].argmax(dim=-1).view(n,-1)
    targs_class = targs_class.view(n,-1)
    targetmask = targs_domain==1
    return (input[targetmask]==targs_class[targetmask]).float().mean()
