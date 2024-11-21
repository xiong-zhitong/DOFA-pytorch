import wandb
import os
import time

# kwargs = dict(
#     project='dino',
#     name='debug3.141',
#     dir='/home/lewaldm/code/PanOpticOn/scripts/runs/11',
# )

# os.makedirs(kwargs['dir'], exist_ok=True)
# wandb.init(**kwargs, resume='auto')
# print('initialized')
# wandb.run.log({'loss': 0.}, step=0)
# wandb.run.log({'loss': 1.0}, step=1)
# wandb.run.finish()
# print('finished')
# time.sleep(2)

# wandb.init(**kwargs, resume='auto')
# print('intialized')
# wandb.run.log({'loss': 10.0}, step=0)
# wandb.run.log({'loss': 11.0}, step=1)

wandb.init(project='dino', dir='/data/panopticon/logs/dino_logs/debug/7', resume='auto')
wandb.run.log({'loss/total_loss':0}, step=0)