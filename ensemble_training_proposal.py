import numpy as np 
import torch

import tqdm
import time
import os

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid

from src.multicls_trainer import PromptBoostingMLTrainer
from src.ptuning import BaseModel, RoBERTaVTuningClassification, OPTVTuningClassification
from src.saver import PredictionSaver, TestPredictionSaver
from src.template import SentenceTemplate, TemplateManager
from src.utils import ROOT_DIR, BATCH_SIZE, create_logger, MODEL_CACHE_DIR
from src.data_util import get_class_num, get_template_list_with_filter, load_dataset, get_task_type, get_template_list

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--adaboost_lr", type = float, default = 1.0)
parser.add_argument("--adaboost_weak_cls", type = int, default = 200)
parser.add_argument("--dataset", type = str, default = 'sst')
parser.add_argument("--model", type = str, default = 'roberta')
parser.add_argument("--template_name", type = str, default = 't5_template3')
parser.add_argument("--stop_criterior", type = str, default = 'best')
parser.add_argument("--label_set_size", type = int, default = 5)
parser.add_argument("--max_template_num", type = int, default = 0)

parser.add_argument("--pred_cache_dir", type = str, default = '')
parser.add_argument("--use_logits", action = 'store_true')
parser.add_argument("--use_wandb", action = 'store_true')
parser.add_argument("--change_template", action = 'store_true')
parser.add_argument("--manual", action = 'store_true')

parser.add_argument("--use_part_templates", action = 'store_true')
parser.add_argument("--start_idx", type = int, default = 0)
parser.add_argument("--end_idx", type = int, default = 10)

parser.add_argument("--second_best", action = 'store_true')
parser.add_argument("--sort_dataset", action = 'store_true')

parser.add_argument("--fewshot", action = 'store_true')
parser.add_argument("--low", action = 'store_true')
parser.add_argument("--fewshot_k", type = int, default = 0)
parser.add_argument("--fewshot_seed", type = int, default = 100, choices = [100, 13, 21, 42, 87])

parser.add_argument("--filter_templates", action = 'store_true')
parser.add_argument("--classifier", type=str, default="svm")
parser.add_argument("--use_logits_final", action="store_true", default=False)

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda')
    adaboost_lr = args.adaboost_lr
    adaboost_weak_cls = args.adaboost_weak_cls
    template_name = args.template_name
    dataset = args.dataset
    sentence_pair = get_task_type(dataset)
    num_classes = get_class_num(dataset)
    model = args.model

    pred_cache_dir = args.pred_cache_dir
    sort_dataset = args.sort_dataset
    stop_criterior = args.stop_criterior
    use_logits = args.use_logits
    use_wandb = args.use_wandb
    label_set_size = args.label_set_size
    max_template_num = args.max_template_num

    adaboost_maximum_epoch = 20000

    fewshot = args.fewshot
    low = args.low
    fewshot_k = args.fewshot_k
    fewshot_seed = args.fewshot_seed

    assert not (fewshot and low), "fewshot and low resource can not be true!"
    filter_templates = args.filter_templates

    suffix = ""
    if args.use_part_templates:
        suffix = f"({args.start_idx}-{args.end_idx})"
    if filter_templates:
        suffix += f"filtered"

    wandb_name = f"{model}-{dataset}-{suffix}"

    if use_wandb:
        if fewshot:
            wandb_name += f"-{fewshot_k}shot-seed{fewshot_seed}"
        elif low:
            wandb_name += f"-low{fewshot_k}-seed{fewshot_seed}"
        wandb.init(project = f'vtuning-{dataset}', name = f'{wandb_name}')


    train_dataset, valid_dataset, test_dataset = load_dataset(dataset_name = dataset, sort_dataset = sort_dataset, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed,
                                                            low_resource = low)

    num_training = len(train_dataset[0])
    num_valid = len(valid_dataset[0])
    num_test = len(test_dataset[0])
    train_labels = torch.LongTensor(train_dataset[1]).to(device)
    valid_labels = torch.LongTensor(valid_dataset[1]).to(device)
    test_labels = torch.LongTensor(test_dataset[1]).to(device)

    weight_tensor = torch.ones(num_training, dtype = torch.float32).to(device) / num_training

    if model == 'roberta':
        vtuning_model = RoBERTaVTuningClassification(model_type = 'roberta-large', cache_dir = os.path.join(MODEL_CACHE_DIR, 'roberta_model/roberta-large/'),
                                                device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    elif model == 'opt-6.7b':
        vtuning_model = OPTVTuningClassification(model_type = 'facebook/opt-6.7b', cache_dir = os.path.join(MODEL_CACHE_DIR, 'opt_model/opt-6.7b/'),
                                                device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    else:
        raise NotImplementedError

    if filter_templates:
        template_dir_list = get_template_list_with_filter(dataset, fewshot = fewshot, low = low,  fewshot_seed = fewshot_seed, 
                                                          fewshot_k = fewshot_k,  topk = 10, return_source_dir = False)
    else:
        template_dir_list = get_template_list(dataset, model = args.model)
    template_manager = TemplateManager(template_dir_list = template_dir_list, output_token = vtuning_model.tokenizer.mask_token, max_template_num = max_template_num,
                                        use_part_templates = args.use_part_templates, start_idx = args.start_idx, end_idx = args.end_idx)

    dir_list = "\n\t".join(template_manager.template_dir_list)
    print(f"using templates from: {dir_list}",)

    trainer = PromptBoostingMLTrainer(adaboost_lr = adaboost_lr, num_classes = num_classes, adaboost_maximum_epoch = adaboost_maximum_epoch, use_logits=args.use_logits_final)

    if pred_cache_dir != '':
        prediction_saver = PredictionSaver(save_dir = os.path.join(ROOT_DIR, pred_cache_dir), model_name = model,
                                            fewshot = fewshot, fewshot_k = fewshot_k, fewshot_seed = fewshot_seed,
                                            low = low,
                                            )
    else:
        prediction_saver = PredictionSaver(model_name = model,
                                            fewshot = fewshot, fewshot_k = fewshot_k, fewshot_seed = fewshot_seed,        
                                            )
    test_pred_saver = TestPredictionSaver(save_dir = os.path.join(ROOT_DIR, f'cached_test_preds/{dataset}/'), model_name = model)
    train_probs, valid_probs = [],[]

    word2idx = vtuning_model.tokenizer.get_vocab()

    def to_numpy(x):
        if isinstance(x, np.ndarray): return x
        return x.detach().cpu().numpy()

    # Obtain features/probs
    print("Collecting features")
    all_train_probs = None
    all_valid_probs = None
    all_test_probs = None
    for template in template_manager.template_list:
        template = template_manager.change_template()
        template.visualize()
        cached_preds, flag = prediction_saver.load_preds(template)
        if not flag:
            train_probs = to_numpy(trainer.pre_compute_logits(vtuning_model, template, train_dataset,))
            valid_probs = to_numpy(trainer.pre_compute_logits(vtuning_model, template, valid_dataset,))
            prediction_saver.save_preds(template, train_probs, valid_probs)
        else:
            train_probs, valid_probs = cached_preds
            train_probs = to_numpy(train_probs)
            valid_probs = to_numpy(valid_probs)
        
        test_probs = to_numpy(trainer.pre_compute_logits(vtuning_model, template, test_dataset))

        if all_train_probs is None:
            all_train_probs = train_probs
            all_valid_probs = valid_probs
            all_test_probs = test_probs
        else:
            all_train_probs = np.concatenate((all_train_probs, train_probs), axis=1)
            all_valid_probs = np.concatenate((all_valid_probs, valid_probs), axis=1)
            all_test_probs = np.concatenate((all_test_probs, test_probs), axis=1)


    # Data
    train_features = all_train_probs
    train_labels = to_numpy(train_labels)
    valid_features = all_valid_probs
    valid_labels = to_numpy(valid_labels)
    test_features = all_test_probs
    test_labels = to_numpy(test_labels)

    estimator = None
    params = {

    }
    if args.classifier == "svm":
        estimator = SVC()
        params = {
            "kernel" : ["rbf"],
            "gamma" : np.arange(0.1, 4.1, 0.1)
        }
    elif args.classifier == "xgboost":
        estimator = GradientBoostingClassifier(random_state=0)
        params = {
            "n_estimators" : [10, 25, 50, 100],
            "learning_rate" : [0.1, 1.0, 2.0],
            "criterion" : ["friedman_mse", "squared_error"],
            "max_depth" : [1, 2, 3],
            "max_features" : ["sqrt", "log2"]
        }
    elif args.classifier == "random-forest":
        estimator = RandomForestClassifier(random_state=0)
        params = {
            "n_estimators" : [10, 25, 50, 100],
            "criterion" : ["gini", "entropy", "log_loss"],
            "max_depth" : [1, 2, 3],
            "max_features" : ["sqrt", "log2"],
            "bootstrap" : [True, False]
        }

    best_params = {}
    best_score = 0
    grid = ParameterGrid(params)
    for i, cur_params in enumerate(grid):
        print(f"Fit {i+1}/{len(grid)}") 
        print(cur_params)
        estimator.set_params(**cur_params)
        estimator.fit(train_features, train_labels)
        train_acc = estimator.score(train_features, train_labels)
        valid_acc = estimator.score(valid_features, valid_labels)
        test_acc = estimator.score(test_features, test_labels)

        print(cur_params)
        print(f"{args.classifier} | Train: {round(train_acc, 4)*100}% - Val: {round(valid_acc, 4)*100}% - Test: {round(test_acc, 4)*100}%")
        if valid_acc > best_score:
            best_score = valid_acc
            best_params = cur_params

    """
    classifier = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        cv=5,
        n_jobs=5,
        verbose=3
    )
    """

    print("Best results")

    estimator.set_params(**best_params)
    estimator.fit(train_features, train_labels)
    train_acc = estimator.score(train_features, train_labels)
    valid_acc = estimator.score(valid_features, valid_labels)
    test_acc = estimator.score(test_features, test_labels)

    print(best_params)
    print(f"{args.classifier} | Train: {round(train_acc, 4)*100}% - Val: {round(valid_acc, 4)*100}% - Test: {round(test_acc, 4)*100}%")

     