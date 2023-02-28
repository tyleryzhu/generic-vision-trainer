# GenViT: Generic Vision Trainer

This repository is a generic, extensible, and robust vision model trainer. It is designed to be a good starting point for any vision project, and to be easily extensible to new models and datasets, and tasks, while also offering robust support for multi-gpu training and automatic mixed-precision training. It's an amalgamation of all the nice parts that I like about the repos I've worked with over the years. 

While there exist many repositories which are lighter-weight and hackable, I usually desire something more robust for experimentation which offers reproducibility while still being reasonably hackable. As a result, I've chosen to adopt a yacs-based configuration system, which works reasonably well for most of my image recognition related tasks. 

I generally subscribe to the philosophy that everything should be written in the config, where having a config and the same version of code is all that is needed to reproduce a result. However, I'm not nearly as strict about there only being one way to doing this, as using command line args for hacking is very useful. Another point that I follow is a factoring of the codebase into a few key components which are separated, making it easier to hack separately on. Every component is built in its corresponding ```build.py``` file, and configs should be entirely handled within that file so that the components themselves can be used in isolation or in other applications. 

The optimizer is also step-based rather than epoch-based, which is a bit more flexible for my use cases.

## Quickstart

To run an example, try running the ```train_resnet18.sh``` script. This will train a resnet18 on the CIFAR10 dataset.

## Todos:
- [ ] Multiple slurm submit options (PySlurm and submitit). 
- [ ] Proper wandb support with loggers (D2 like?). 


## Nice repository references and inspirations

Tiny, hackable codebases:
- Karpathy's [minGPT](https://github.com/karpathy/minGPT)
- Bring Your Own Latent ([BYOL](https://github.com/sthalles/PyTorch-BYOL))
- Kaiming's [Masked Autoencoders](https://github.com/facebookresearch/mae) (absolutely superb for submitit, helpful for slurm).
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) (slightly more structure but enough to train large models)
- [CleanRL](https://github.com/vwxyzjn/cleanrl), nice for RL... obviously


Medium sized repositories with good abstractions (actual folder breakdowns):
- Berkeley's [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) (for graphics however). 
- Microsoft [Swin Transformer](https://github.com/microsoft/Swin-Transformer). Most of the code is taken from here because I enjoyed my time working with this repoository.

Larger codebases that are decent:
- Facebook's [PySlowFast](https://github.com/facebookresearch/SlowFast), but too bloated to hack with (much like detectron2).
