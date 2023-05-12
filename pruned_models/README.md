# File Naming Convention

_x :    The fraction of FLOPS we reach through uniform pruning.
        Pruned model will have x/100*orig_FLOPs many FLOPs.


        import torch
        from resnet import resnet56
        from embedl.plumbing.torch.metrics.target import Target
        from embedl.torch.pruning.methods import UniformPruning
        from embedl.torch.metrics.performances import Flops 

## resnet56_uniform_20

        model = resnet56()
        input_shape = [1,3,32,32]       
        base_flops = measure_flops(
                model=model,
                input_shape=input_shape
        )
        pruning_method = UniformPruning(
                target=Target(Flops(), fraction=0.2),
                step_size=1
        )
        pruning_steps = pruning_method.prune(model, input_shape)
        pruned_flops = measure_flops(
                model=model,
                input_shape=input_shape
        )
        print(pruned_flops/base_flops) # 0.19897505715220812
        torch.save(model, 'pruned_models/resnet56_uniform_20.th')

## resnet56_uniform_60

        model = resnet56()
        input_shape = [1,3,32,32]       
        base_flops = measure_flops(
        model=model,
        input_shape=input_shape
        )
        pruning_method = UniformPruning(
        target=Target(Flops(), fraction=0.6),
        step_size=1
        )
        pruning_steps = pruning_method.prune(model, input_shape)
        pruned_flops = measure_flops(
        model=model,
        input_shape=input_shape
        )
        print(pruned_flops/base_flops) # 0.5976937173922103
        torch.save(model, 'pruned_models/resnet56_uniform_60.th')

## mobilenetv2_uniform_60

        model = mobilenetv2()
        input_shape = [1,3,32,32]       
        base_flops = measure_flops(
        model=model,
        input_shape=input_shape
        )
        pruning_method = UniformPruning(
        target=Target(Flops(), fraction=0.6),
        step_size=1
        )
        pruning_steps = pruning_method.prune(model, input_shape)
        pruned_flops = measure_flops(
        model=model,
        input_shape=input_shape
        )
        print(pruned_flops/base_flops) # 0.598694904487919
        torch.save(model, 'pruned_models/mobilenetv2_uniform_60.th')

## mobilenetv2_uniform_20

        model = mobilenetv2()
        input_shape = [1,3,32,32]       
        base_flops = measure_flops(
        model=model,
        input_shape=input_shape
        )
        pruning_method = UniformPruning(
        target=Target(Flops(), fraction=0.2),
        step_size=2
        )
        pruning_steps = pruning_method.prune(model, input_shape)
        pruned_flops = measure_flops(
        model=model,
        input_shape=input_shape
        )
        print(pruned_flops/base_flops) # 0.1980394686585399

## resnet56_magnitude_20

        state_dict = torch.load(
        "/home/jonna/hyperparameter_sensitivity_pruning/experiments/cifar10/resnet56/base_model/results_0/lr_10**-1.00_wd_10**-4.00/checkpoint_final.th"
        )["state_dict"]
        state_dict = {key[7:]: weights for key, weights in state_dict.items()}
        model = resnet56()
        model.load_state_dict(state_dict)
        model.cuda()

        print("Validate before pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())


        input_shape = [1, 3, 32, 32]
        base_flops = measure_flops(model=model, input_shape=input_shape)


        scorer = ChannelPruningScorer(
        importance_score=WeightMagnitude(), channel_pruning_balancer=None
        )
        tactic = ChannelPruningTactic(step_size=1, search_depth=1, speedup_pruning=False)

        pruning_method = PruningMethod(scorer, [tactic], target=Target(Flops(), fraction=0.2))
        pruning_steps = pruning_method.prune(model, input_shape=input_shape)

        print("\nValidate after pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())

        pruned_flops = measure_flops(model=model, input_shape=input_shape)
        print(pruned_flops / base_flops)  # 0.19983580331374945

        plot_pruning_profile(model, pruning_steps)
        torch.save(
        model,
        "/home/jonna/pytorch_resnet_cifar10/pruned_models/resnet56_magnitude_20.th",
        )


## resnet56_magnitude_60

        state_dict = torch.load(
        "/home/jonna/hyperparameter_sensitivity_pruning/experiments/cifar10/resnet56/base_model/results_0/lr_10**-1.00_wd_10**-4.00/checkpoint_final.th"
        )["state_dict"]
        state_dict = {key[7:]: weights for key, weights in state_dict.items()}
        model = resnet56()
        model.load_state_dict(state_dict)
        model.cuda()

        print("Validate before pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())


        input_shape = [1, 3, 32, 32]
        base_flops = measure_flops(model=model, input_shape=input_shape)


        scorer = ChannelPruningScorer(
        importance_score=WeightMagnitude(), channel_pruning_balancer=None
        )
        tactic = ChannelPruningTactic(step_size=1, search_depth=1, speedup_pruning=False)

        pruning_method = PruningMethod(scorer, [tactic], target=Target(Flops(), fraction=0.6))
        pruning_steps = pruning_method.prune(model, input_shape=input_shape)

        print("\nValidate after pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())

        pruned_flops = measure_flops(model=model, input_shape=input_shape)
        print(pruned_flops / base_flops)  # 0.5999830832084312

        plot_pruning_profile(model, pruning_steps)
        torch.save(
        model,
        "/home/jonna/pytorch_resnet_cifar10/pruned_models/resnet56_magnitude_60.th",
        )

## mobilenetv2_magnitude_60

        state_dict = torch.load(
        "/home/jonna/hyperparameter_sensitivity_pruning/experiments/cifar10/mobilenetv2/base_model/results_0/lr_10**-1.00_wd_10**-4.00/checkpoint_final.th"
        )["state_dict"]
        state_dict = {key[7:]: weights for key, weights in state_dict.items()}
        model = mobilenetv2()
        model.load_state_dict(state_dict)
        model.cuda()

        print("Validate before pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())

        input_shape = [1, 3, 32, 32]
        base_flops = measure_flops(model=model, input_shape=input_shape)


        scorer = ChannelPruningScorer(
        importance_score=WeightMagnitude(), channel_pruning_balancer=None
        )
        tactic = ChannelPruningTactic(step_size=1, search_depth=1, speedup_pruning=False)

        pruning_method = PruningMethod(scorer, [tactic], target=Target(Flops(), fraction=0.6))
        pruning_steps = pruning_method.prune(model, input_shape=input_shape)

        print("\nValidate after pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())

        pruned_flops = measure_flops(model=model, input_shape=input_shape)
        print(pruned_flops / base_flops)  # 0.5999830832084312

        plot_pruning_profile(model, pruning_steps)
        torch.save(model, 'pruned_models/mobilenetv2_magnitude_60.th')
        with open('pruned_models/pruningsteps_mobilenetv2_magnitude_60.th', 'wb') as f:
        pickle.dump(pruning_steps, f)

## mobilenetv2_magnitude_20

        state_dict = torch.load(
        "/home/jonna/hyperparameter_sensitivity_pruning/experiments/cifar10/mobilenetv2/base_model/results_0/lr_10**-1.00_wd_10**-4.00/checkpoint_final.th"
        )["state_dict"]
        state_dict = {key[7:]: weights for key, weights in state_dict.items()}
        model = mobilenetv2()
        model.load_state_dict(state_dict)
        model.cuda()

        print("Validate before pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())

        input_shape = [1, 3, 32, 32]
        base_flops = measure_flops(model=model, input_shape=input_shape)


        scorer = ChannelPruningScorer(
        importance_score=WeightMagnitude(), channel_pruning_balancer=None
        )
        tactic = ChannelPruningTactic(step_size=1, search_depth=1, speedup_pruning=False)

        pruning_method = PruningMethod(scorer, [tactic], target=Target(Flops(), fraction=0.2))
        pruning_steps = pruning_method.prune(model, input_shape=input_shape)

        print("\nValidate after pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())

        pruned_flops = measure_flops(model=model, input_shape=input_shape)
        print(pruned_flops / base_flops)  # 0.1999645132889108

        plot_pruning_profile(model, pruning_steps)
        torch.save(model, 'pruned_models/mobilenetv2_magnitude_20.th')
        with open('pruned_models/pruningsteps_mobilenetv2_magnitude_20.th', 'wb') as f:
        pickle.dump(pruning_steps, f)

## mobilenetv2_magnitude_40

        state_dict = torch.load(
        "/home/jonna/hyperparameter_sensitivity_pruning/experiments/cifar10/mobilenetv2/base_model/results_0/lr_10**-1.00_wd_10**-4.00/checkpoint_final.th"
        )["state_dict"]
        state_dict = {key[7:]: weights for key, weights in state_dict.items()}
        model = mobilenetv2()
        model.load_state_dict(state_dict)
        model.cuda()

        print("Validate before pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())

        input_shape = [1, 3, 32, 32]
        base_flops = measure_flops(model=model, input_shape=input_shape)


        scorer = ChannelPruningScorer(
        importance_score=WeightMagnitude(), channel_pruning_balancer=None
        )
        tactic = ChannelPruningTactic(step_size=1, search_depth=1, speedup_pruning=False)

        pruning_method = PruningMethod(scorer, [tactic], target=Target(Flops(), fraction=0.4))
        pruning_steps = pruning_method.prune(model, input_shape=input_shape)

        print("\nValidate after pruning")
        validate(val_loader, torch.nn.DataParallel(model), nn.CrossEntropyLoss().cuda())

        pruned_flops = measure_flops(model=model, input_shape=input_shape)
        print(pruned_flops / base_flops)  # 0.39991562513343865

        plot_pruning_profile(model, pruning_steps)
        torch.save(model, "pruned_models/mobilenetv2_magnitude_40.th")
        with open("pruned_models/pruningsteps_mobilenetv2_magnitude_40.th", "wb") as f:
        psteps = pickle.load(f)
        assert psteps == pruning_steps
        with open("pruned_models/pruningsteps_mobilenetv2_magnitude_40.th", "wb") as f:
        pickle.dump(pruning_steps, f)

## mobilenetv2_uniform_40

        model = mobilenetv2()
        input_shape = [1, 3, 32, 32]
        base_flops = measure_flops(model=model, input_shape=input_shape)
        pruning_method = UniformPruning(target=Target(Flops(), fraction=0.4), step_size=1)
        pruning_steps = pruning_method.prune(model, input_shape)
        pruned_flops = measure_flops(model=model, input_shape=input_shape)
        print(pruned_flops / base_flops)  # 0.3999708840389222
        torch.save(model, "pruned_models/mobilenetv2_uniform_40.th")
        with open("pruned_models/pruningsteps_mobilenetv2_uniform_40.th", "wb") as f:
        pickle.dump(pruning_steps, f)

## resnet50_magnitude_40
        state = torch.load(
        "/home/jonna/hyperparameter_sensitivity_pruning/experiments/imagenet/base_model/results/lr_10**-1.00_wd_10**-4.00/checkpoint.pth"
        )
        model = resnet50()
        model.load_state_dict(state["model"])
        model.cuda()
        input_shape = [1, 3, 224, 224]
        base_flops = measure_flops(model=model, input_shape=input_shape)
        scorer = ChannelPruningScorer(
        importance_score=WeightMagnitude(), channel_pruning_balancer=None
        )
        tactic = ChannelPruningTactic(step_size=1, search_depth=1, speedup_pruning=False)
        pruning_method = PruningMethod(
        scorer,
        [tactic],
        target=Target(Flops(), fraction=0.4),
        )
        pruning_steps = pruning_method.prune(model, input_shape=input_shape)

        pruned_flops = measure_flops(model=model, input_shape=input_shape)
        print(pruned_flops / base_flops)  #

        plot_pruning_profile(model, pruning_steps)

        torch.save(model, "pruned_models/resnet50_magnitude_40.th")

        with open("pruned_models/pruningsteps_resnet50_magnitude_40.th", "wb") as f:
        pickle.dump(pruning_steps, f)

## resnet50_magnitude_40_2
        state = torch.load(
        "/home/jonna/hyperparameter_sensitivity_pruning/experiments/imagenet/base_model/3_3_grid/results/lr_10**-1.00_wd_10**-4.00/checkpoint_middle.pth"
        )
        model = resnet50()
        model.load_state_dict(state["model"])
        model.cuda()

        input_shape = [1, 3, 224, 224]

        base_flops = measure_flops(model, input_shape=input_shape)
        print(base_flops)


        exclude_ops = [
                "conv1",
                "layer1_0_downsample_0",
                "layer2_0_downsample_0",
                "layer3_0_downsample_0",
                "layer4_0_downsample_0",
                "layer4_2_conv3",
                "layer4_1_conv3",
                "layer4_0_conv3",
                "layer3_5_conv3",
                "layer3_4_conv3",
                "layer3_3_conv3",
                "layer3_2_conv3",
                "layer3_1_conv3",
                "layer3_0_conv3",
                "layer2_3_conv3",
                "layer2_2_conv3",
                "layer2_1_conv3",
                "layer2_0_conv3",
                "layer1_2_conv3",
                "layer1_1_conv3",
                "layer1_0_conv3",
        ]

        with open("/home/jonna/pytorch_resnet_cifar10/pruned_models/pruningsteps_resnet50_magnitude_40.th", "rb") as f:
        steps = pickle.load(f)

        pruning_steps = [steps[0]]
        for step in steps[1:]:
        if step.group_dict['name'] in exclude_ops:
                continue
        pruning_steps.append(step)



        apply_pruning_steps(model, pruning_steps, input_shape=input_shape)

        pruned_flops = measure_flops(model, input_shape=input_shape)
        print(pruned_flops)
        initial_ratio = pruned_flops/base_flops
        print(initial_ratio)

        
        scorer = ChannelPruningScorer(
        importance_score=WeightMagnitude(), channel_pruning_balancer=None
        )

        tactic = ChannelPruningTactic(step_size=1, search_depth=1, speedup_pruning=False)
        pruning_method = PruningMethod(
        scorer,
        [tactic],
        target=Target(Flops(), fraction=(0.4 / initial_ratio)),
        exclude_ops=exclude_ops,
        )
        pruning_steps_2 = pruning_method.prune(model, input_shape=input_shape)

        pruned_flops = measure_flops(model=model, input_shape=input_shape)
        print(pruned_flops / base_flops)  #

        torch.save(model, "pruned_models/resnet50_magnitude_40_2.th")

        with open("pruned_models/pruningsteps_resnet50_magnitude_40_2_part.th", "wb") as f:
        pickle.dump(pruning_steps_2, f)

        pruning_steps.append(pruning_steps_2[1:])

        with open("pruned_models/pruningsteps_resnet50_magnitude_40_2_all.th", "wb") as f:
        pickle.dump(pruning_steps, f)
