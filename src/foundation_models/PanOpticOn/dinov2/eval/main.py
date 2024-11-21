import os

from dinov2.eval.wrapper import backbone_to_features
from dinov2.data.loaders import make_dataset
from dinov2.utils.config import resolve_configs, write_config
from dinov2.eval.setup import setup_logger
from dinov2.eval.linear import run_eval_linear
from dinov2.eval.knn import eval_knn_with_model
from functools import partial
from omegaconf import OmegaConf
import pandas as pd
from dinov2.configs import default_eval_linear_config, default_eval_knn_config, default_eval_linear_multilabel_config
import fire
import time
from dinov2.eval.setup import parse_model_obj, parse_config_obj


def parse_cfgs(*cfgs):
    return OmegaConf.merge(*resolve_configs(cfgs))

def _setup(model_obj, cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    global logger
    logger = setup_logger('dinov2', os.path.join(output_dir, 'log'), reset_logger=True)
    cfg.output_dir = output_dir
    write_config(cfg, output_dir)

    _model_wrapper = parse_model_obj(model_obj)
    return _model_wrapper


def do_linear(model_obj, cfg, output_dir):
    _model_wrapper = _setup(model_obj, cfg, output_dir)

    results_dict = run_eval_linear(
        _model_wrapper_partial=_model_wrapper,
        output_dir=output_dir,

        train_dataset_cfg=cfg.train_dataset,
        val_dataset_cfg=cfg.val_dataset,
        test_dataset_cfgs=cfg.test_datasets_list,

        batch_size=cfg.optim.dl.batch_size,
        num_workers=cfg.optim.dl.num_workers,

        epochs=cfg.optim.epochs,
        iter_per_epoch=cfg.optim.iter_per_epoch,
        save_checkpoint_frequency_epoch=cfg.optim.save_checkpoint_frequency_epoch,
        eval_period_epoch=cfg.optim.eval_period_epoch,
        eval_period_iter=cfg.optim.eval_period_iter,

        heads=cfg.heads,

        val_metrics=cfg.task.val_metrics,
        test_metrics_list=cfg.task.test_metrics_list,
        criterion_cfg = cfg.task.criterion_cfg,

        # add dinov2 eval args, not sure why
        resume=not cfg.no_resume,
        classifier_fpath=cfg.classifier_fpath,
        val_class_mapping_fpath=cfg.val_class_mapping_fpath,
        test_class_mapping_fpaths=cfg.test_class_mapping_fpaths,)

    return results_dict


def do_knn(model_obj, cfg, output_dir):
        # this corresponds to vanilla DINOv2 ModelWithNormalize

    # build model
    _model_wrapper = _setup(model_obj, cfg, output_dir)
    bb_to_feat_adapter = partial(
        backbone_to_features, use_n_blocks=1, pooling='knn')
    feature_model = _model_wrapper(
        n_last_blocks=1, 
        bb_to_feat_adapter=bb_to_feat_adapter)

    # config equals train_config.evaluation
    train_dataset = make_dataset(cfg.train_dataset, seed=cfg.seed)
    val_dataset = make_dataset(cfg.test_dataset, seed=cfg.seed)

    # actual computations
    results_dict = eval_knn_with_model(
        model=feature_model,
        output_dir=output_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        nb_knn=cfg.heads.nb_knn,
        temperature=cfg.heads.temperature,
        metric_cfg=cfg.task.metrics,
        gather_on_cpu=cfg.heads.gather_on_cpu,
        n_per_class_list=cfg.heads.n_per_class_list,
        n_tries=cfg.heads.n_tries,
        dl_cfg=cfg.optim.dl,
        is_multilabel=cfg.task.is_multilabel,)

    return results_dict


def main(model_obj, config_obj, output_dir, **overwrites):

    # logger for task overview
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger('eval', os.path.join(output_dir, 'log'), to_sysout=True, reset_logger=True)
    logger.info(f'model-obj: {str(model_obj)}')
    logger.info(f'config-obj: {str(config_obj)}')
    logger.info(f'output-dir: {output_dir}')
    logger.info(f'overwrites: {overwrites}')

    # get tasks
    configs = parse_config_obj(config_obj)
    task_names = [f.split('/')[-1].replace('.yaml','') for f in configs]
    configs = [OmegaConf.load(f) for f in configs]
    output_dirs = [os.path.join(output_dir, name) for name in task_names]

    # build tasks
    tasks = {}
    for tn, cfg, out_dir in zip(task_names, configs, output_dirs):
        id = cfg.task.id
        if id == 'classification':
            default_cfg = default_eval_linear_config
            fct = do_linear

        elif id == 'multilabelclassification':
            default_cfg = default_eval_linear_multilabel_config
            fct = do_linear

        elif id == 'knn':
            default_cfg = default_eval_knn_config
            fct = do_knn

        else: 
            raise ValueError(f'Unknown id {id}')
        
        cfg = parse_cfgs(default_cfg, cfg, overwrites)
        task = partial(fct, model_obj, cfg, out_dir)
        tasks[tn] = task


    # run tasks
    all_start = time.time()
    for tn, task in tasks.items():
        start = time.time()
        logger.info(f'Running task {tn} ...')

        results_dict = task()

        tasks[tn] = results_dict
        logger.info(f'Finished in {time.time()-start:.2f}s')

    # return all results
    logger.info(f'All tasks finished in {time.time()-all_start:.2f}s')
    rows = []
    for tn, res in tasks.items():
        if not isinstance(res, list):
            res = [res]
        for res_dict in res:
            rows.append(dict(
                task = os.path.join(tn, res_dict.get('postfix','')),
                value = res_dict['val'],
                metric = res_dict['metric_str'],
                best_classifier = res_dict['name']
            ))

    top_metric_df = pd.DataFrame(rows).set_index(['task','metric'])
    with open(os.path.join(output_dir, 'results.csv'),'w') as f:
        top_metric_df.to_csv(f)
    logger.info(f'\n{top_metric_df.to_string()}')

    

if __name__ == '__main__':
    fire.Fire()