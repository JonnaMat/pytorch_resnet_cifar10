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
