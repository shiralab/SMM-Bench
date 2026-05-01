import json
import os
import numpy as np
import lightgbm as lgb


class SMMBenchPS(object):
    def __init__(self, model_dir):
        train_model_file = os.path.join(model_dir, 'train_acc_ja_best_model.json')
        if os.path.exists(train_model_file):
            self.lgb_train_acc = lgb.CVBooster(model_file=train_model_file)
        else:
            print(f'[Warning] Cannot find model file "{model_dir}/train_acc_ja_best_model.json" in {self.__class__.__name__}')
            self.lgb_train_acc = None

        test_model_file = os.path.join(model_dir, 'test_acc_ja_best_model.json')
        if os.path.exists(test_model_file):
            self.lgb_test_acc = lgb.CVBooster(model_file=test_model_file)
        else:
            print(f'[Warning] Cannot find model file "{model_dir}/test_acc_ja_best_model.json" in {self.__class__.__name__}')
            self.lgb_test_acc = None

        label_file = os.path.join(model_dir, 'label.json')
        if os.path.exists(label_file):
            with open(label_file) as f:
                labels = json.load(f)
            self.x_labels = labels['x']
        else:
            print(f'[Warning] Cannot find label file "{model_dir}/label.json" in {self.__class__.__name__}')
            self.x_labels = None

        self.D = 64
        self.L = 32
        self.n_source_model = 2        

    def __call__(self, x, mode='train_acc'):
        if len(x.shape) == 1:
            xx = np.array([x])
        else:
            xx = np.array(x)
        
        # min max range of x
        xx = np.clip(xx, 0., 1.)

        # Prediction
        if mode == 'train_acc' and self.lgb_train_acc is not None:
            y_pred = np.array(self.lgb_train_acc.predict(xx)).mean(axis=0)
            # min max range of fx
            y_pred = np.clip(y_pred, 0., 1.)
        elif mode == 'test_acc' and self.lgb_test_acc is not None:
            y_pred = np.array(self.lgb_test_acc.predict(xx)).mean(axis=0)
            # min max range of fx
            y_pred = np.clip(y_pred, 0., 1.)
        else:
            print(f'[Warning] Unknown mode of evaluation in {self.__class__.__name__}')
            y_pred = np.zeros(len(xx))
        
        if len(x.shape) == 1:
            return y_pred[0]
        else:
            return y_pred


class SMMBenchDFS(object):
    def __init__(self, model_dir):
        train_file = os.path.join(model_dir, 'train_acc_ja_best_model.json')
        if os.path.exists(train_file):        
            self.lgb_train_acc = lgb.CVBooster(model_file=train_file)
        else:
            print(f'[Warning] Cannot find model file "{model_dir}/train_acc_ja_best_model.json" in {self.__class__.__name__}')
            self.lgb_train_acc = None

        test_file = os.path.join(model_dir, 'test_acc_ja_best_model.json')
        if os.path.exists(test_file):        
            self.lgb_test_acc = lgb.CVBooster(model_file=test_file)
        else:
            print(f'[Warning] Cannot find model file "{model_dir}/test_acc_ja_best_model.json" in {self.__class__.__name__}')
            self.lgb_test_acc = None

        label_file = os.path.join(model_dir, 'label.json')
        if os.path.exists(label_file):
            with open(label_file) as f:
                labels = json.load(f)
            self.x_labels = labels['x']
        else:
            print(f'[Warning] Cannot find label file "{model_dir}/label.json" in {self.__class__.__name__}')
            self.x_labels = None

        self.D = 95
        self.D_c = 32
        self.D_x = 63
        self.num_categories = 3
        self.n_source_model = 2        

    def __call__(self, c, x, mode='train_acc'):
        if len(c.shape) == 1:
            cc = np.array([c])
        else:
            cc = np.array(c)

        if len(x.shape) == 1:
            xx = np.array([x])
        else:
            xx = np.array(x)

        # min max range of x
        xx = np.clip(xx, 0.4, 1.5)

        # Concat
        X = np.concatenate([cc, xx], axis=1)

        # Prediction
        if mode == 'train_acc' and self.lgb_train_acc is not None:
            y_pred = np.array(self.lgb_train_acc.predict(X)).mean(axis=0)
            # min max range of fx
            y_pred = np.clip(y_pred, 0., 1.)
        elif mode == 'test_acc' and self.lgb_test_acc is not None:
            y_pred = np.array(self.lgb_test_acc.predict(X)).mean(axis=0)
            # min max range of fx
            y_pred = np.clip(y_pred, 0., 1.)
        else:
            print(f'[Warning] Unknown mode of evaluation in {self.__class__.__name__}')
            y_pred = np.zeros(len(X))
        
        if len(x.shape) == 1:
            return y_pred[0]
        else:
            return y_pred
