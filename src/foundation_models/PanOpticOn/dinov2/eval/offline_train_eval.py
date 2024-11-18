from omegaconf import OmegaConf
import os
import pandas as pd
import logging
import itertools
import torch.distributed as dist
import time
from dinov2.eval.main import main as eval_main
import dinov2.distributed as distributed
from pathlib import Path
import torch

logger = logging.getLogger("dinov2")




def do_offline_eval(
        base_dir, 
        config_dir, 
        remove_ckpts=False,
        return_all_res=False, 
        recompute=False,
        include_final_ckpt=False,
        overwrites=None,
    ):
    overwrites = overwrites or {}

    # get model config
    train_config = OmegaConf.load(os.path.join(base_dir, 'config.yaml'))
    model_kwargs = train_config.student
    eval_base_dir = os.path.join(base_dir, 'eval')

    # create teacher_checkpoint.pth tasks
    
    tasks = []
    if os.path.exists(eval_base_dir):
        for d in os.listdir(eval_base_dir):
            if os.path.exists(os.path.join(eval_base_dir, d, 'teacher_checkpoint.pth')):
                
                if not recompute and os.path.exists(os.path.join(eval_base_dir, d, 'results.csv')):
                    continue

                output_dir = os.path.join(eval_base_dir, d)
                ckpt_path = os.path.join(eval_base_dir, d, 'teacher_checkpoint.pth')
                
                eval_model_obj = OmegaConf.create(dict(
                    id = 'dinov2',
                    pretrained_weights = [dict(
                        path=ckpt_path,
                        checkpoint_key='teacher',
                        exclude='cached',
                        prefix_map = {'backbone.': ''})],
                    model_kwargs = model_kwargs))
                task_args = dict(
                    model_obj = eval_model_obj,
                    config_obj = config_dir,
                    output_dir = output_dir)

                tasks.append(task_args)

    # create model_final.pth task

    if include_final_ckpt and os.path.exists(os.path.join(base_dir, 'model_final.pth')):
        output_dir = os.path.join(base_dir, 'eval_model_final')
        ckpt_path = os.path.join(base_dir, 'model_final.pth')
        
        eval_model_obj = OmegaConf.create(dict(
            id = 'dinov2',
            pretrained_weights = [dict(
                path=ckpt_path,
                checkpoint_key='model',
                include='teacher',
                exclude='cached_optical_embs',
                prefix_map = {'teacher.backbone.': ''})],
            model_kwargs = model_kwargs))
        task_args = dict(
            model_obj = eval_model_obj,
            config_obj = config_dir,
            output_dir = output_dir,
            iteration = torch.load(ckpt_path)['iteration'])

        tasks.append(task_args)

    # assign tasks to rank
    tasks_rank = itertools.islice(tasks, 
                                  distributed.get_global_rank(), 
                                  len(tasks), 
                                  distributed.get_global_size())
    logger.info(f'Computing {len(tasks)} tasks with {distributed.get_global_size()} workers ...')

    # destroy progress group for independent eval
    is_main = distributed.is_main_process() # need variable since dist.destroy_progress_group() will be called
    if distributed.is_enabled():
        dist.barrier()
        dist.destroy_process_group()

    # work tasks
    start = time.time()
    results = []
    for task in tasks_rank:
        results_dict = eval_main(
            model_obj=task['model_obj'],
            config_obj=task['config_obj'],
            output_dir=task['output_dir'], 
            **overwrites)
        results.append(results_dict)
        if remove_ckpts and not 'model_final' in task['output_dir']: # final ckpt never removed
            os.remove(os.path.join(task['output_dir'], 'teacher_checkpoint.pth'))
    avg_time = (time.time() - start) / max(1, len(list(tasks_rank)))

    # gather results
    results = []
    if is_main: 

        # (naive) wait until all tasks are done
        timeout = max(10, 2.0 * avg_time)
        start = time.time()
        tasks_done = 0
        while tasks_done < len(tasks) and time.time() - start < timeout:
            tasks_done = len([d for d in tasks 
                if os.path.exists(os.path.join(d['output_dir'], 'results.csv'))])
            time.sleep(1)
        if tasks_done < len(tasks):
            raise RuntimeError(f'Not all tasks done after timeout {timeout:.2f}s')

        # gather tasks of interest
        if return_all_res:
            assert not include_final_ckpt, 'return_all_res only possible with include_final_ckpt=False'
            tasks = []
            for d in os.listdir(eval_base_dir):
                if os.path.exists(os.path.join(eval_base_dir, d, 'results.csv')):
                    tasks.append(dict(output_dir=os.path.join(eval_base_dir, d))) 

        # gather results
        for t in tasks: 
            output_dir = t['output_dir']
            iteration = t.get('iteration', str(Path(*Path(output_dir).parts[-1:])))
            task_results = dict(iteration=iteration)
            
            df = pd.read_csv(os.path.join(output_dir, 'results.csv'))
            for i in range(df.shape[0]):
                k = os.path.join(df.at[i,'task'], df.at[i,'metric'])
                v = float(df.at[i,'value'])
                task_results[k] = v
            results.append(task_results)

    return results