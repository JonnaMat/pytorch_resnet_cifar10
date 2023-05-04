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