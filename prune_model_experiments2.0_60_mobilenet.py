import torch
import pickle
from pathlib import Path
import os
from models import mobilenetv2, resnet56
from torchvision.models import resnet50
from embedl.plumbing.torch.metrics.target import Target
from embedl.torch.pruning.methods import UniformPruning
from embedl.torch.viewer import view_model
from embedl.torch.metrics.performances import Flops  
from embedl.torch.metrics.measure_performance import measure_flops
import torchvision.datasets as datasets
from embedl.torch.pruning.methods import plot_pruning_profile 
import torchvision.transforms as transforms
import torch.nn as nn
from embedl.torch.metrics.performances import Flops
from embedl.torch.pruning.methods import (
    PruningMethod,
    ChannelPruningTactic,
)
from embedl.plumbing.torch.metrics.scorers import ChannelPruningScorer, PruningBalancer
from embedl.torch.metrics.importance_scores import WeightMagnitude
from embedl.plumbing.torch.pruning.method import apply_pruning_steps

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="/home/jonna/data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ),
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """

    prec1 = 0
    count = 0
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 += accuracy(output.data, target)[0] * target.size(0)
            # print(accuracy(output.data, target)[0])
            count += target.size(0)

    print(f" * Prec@1 {prec1/count:.3f}")
    return

# build folder structure
save_dir = "/home/jonna/hyperparameters-under-pruning/experiments2.0/cifar10/mobilenetv2/finetuning/magnitude_60/results"

for lr in [-1.0, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -0.8, -0.6, -0.4, -0.2]:
    for wd in [-4.0, -4.2, -4.6, -4.4, -3.4, -3.8, -3.6, -3.2, -2.2, -2.4, -2.8, -4.8, -3.0, -2.6]:


        savedir = save_dir + f"/lr_10**{lr:.2f}_wd_10**{wd:.2f}"

        # Check the save_dir exists or not
        if not os.path.exists(savedir):
            os.makedirs(savedir)


for root, dirs, files in os.walk(
    "/home/jonna/hyperparameter_sensitivity_pruning/experiments/cifar10/mobilenetv2/base_model"
):
    for file in files:
        if file.endswith("checkpoint_final.th"):
            name = root.split("/")[-1]
            savedir = save_dir + f"/{name}"
            if Path(f"{savedir}/mobilenetv2_magnitude_60.th").is_file():
                break

            print(f"{root}/checkpoint_final.th")
            state_dict = torch.load(f"{root}/checkpoint_final.th")["state_dict"]
            state_dict = {key[7:]: weights for key, weights in state_dict.items()}
            model = mobilenetv2()
            model.load_state_dict(state_dict)
            model.cuda()

            print("Validate before pruning")
            validate(
                val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda()
            )

            input_shape = [1, 3, 32, 32]
            base_flops = measure_flops(model=model, input_shape=input_shape)

            scorer = ChannelPruningScorer(
                importance_score=WeightMagnitude(), channel_pruning_balancer=None
            )
            tactic = ChannelPruningTactic(
                step_size=1, search_depth=1, speedup_pruning=False
            )

            pruning_method = PruningMethod(
                scorer, [tactic], target=Target(Flops(), fraction=0.6)
            )
            pruning_steps = pruning_method.prune(model, input_shape=input_shape)

            print("\nValidate after pruning")
            validate(
                val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda()
            )

            pruned_flops = measure_flops(model=model, input_shape=input_shape)
            print(pruned_flops / base_flops)

            torch.save(
                model,
                f"{savedir}/mobilenetv2_magnitude_60.th",
            )