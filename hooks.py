import torch

def gpt_manual_model_summary(model, input_sample, summary, render_queue):
    hooks = []

    def register_hook(module):

        def hook(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(summary)

            m_key = f"{module_idx:03d}_{class_name}"

            params = 0
            trainable = 0
            for p in module.parameters(recurse=False):
                params += p.numel()
                if p.requires_grad:
                    trainable += p.numel()

            if hasattr(output, "last_hidden_state"):
                act = output.last_hidden_state
            elif isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            batch_size = act.size(0)
            flat = act.view(act.size(0), -1)

            vals, idxs = torch.max(flat, dim=1)
            #vals_min, idxs_min = torch.min(flat, dim=1)

            def unflatten_index(flat_idx, shape):
                coords = []
                for dim in reversed(shape):
                    coords.append(flat_idx % dim)
                    flat_idx //= dim
                return list(reversed(coords))

            orig_shape = act.shape[1:]
            top_coords = [unflatten_index(idxs[b].item(), orig_shape) for b in range(batch_size)]
            #min_coords = [unflatten_index(idxs_min[b].item(), orig_shape) for b in range(batch_size)]

            summary[m_key] = {
                # "input_shape": list(input[0].size()) if isinstance(input, tuple) else [],
                # "output_shape": list(output.size()) if isinstance(output, torch.Tensor) else [],
                # "num_params": params,
                # "trainable": trainable,
                "shape": act, #.detach().cpu().numpy(),
                # "top_value": vals,
                # "top_index": idxs,
                "top_index_coords": top_coords,
                # "min_value": vals_min,
                # "min_index": idxs_min,
                # "min_index_coords": min_coords,
            }

        if not isinstance(module, torch.nn.ModuleList) and \
                not isinstance(module, torch.nn.Sequential) and \
                module != model:
            hooks.append(module.register_forward_hook(hook))

        return hook


    model.apply(register_hook)
    model.eval()
    with torch.no_grad():
        model(**input_sample)
        render_queue.put(summary)
        summary.clear()


def cnn_manual_model_summary(model, input_sample, summary, render_queue):
    hooks = []

    def register_hook(module):

        def hook(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(summary)

            m_key = f"{module_idx:03d}_{class_name}"

            params = 0
            trainable = 0
            for p in module.parameters(recurse=False):
                params += p.numel()
                if p.requires_grad:
                    trainable += p.numel()

            if hasattr(output, "last_hidden_state"):
                act = output.last_hidden_state
            elif isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            batch_size = act.size(0)
            flat = act.view(act.size(0), -1)

            vals, idxs = torch.max(flat, dim=1)
            #vals_min, idxs_min = torch.min(flat, dim=1)

            def unflatten_index(flat_idx, shape):
                coords = []
                for dim in reversed(shape):
                    coords.append(flat_idx % dim)
                    flat_idx //= dim
                return list(reversed(coords))

            orig_shape = act.shape[1:]
            top_coords = [unflatten_index(idxs[b].item(), orig_shape) for b in range(batch_size)]
            # min_coords = [unflatten_index(idxs_min[b].item(), orig_shape) for b in range(batch_size)]

            summary[m_key] = {
                # "input_shape": list(input[0].size()) if isinstance(input, tuple) else [],
                # "output_shape": list(output.size()) if isinstance(output, torch.Tensor) else [],
                # "num_params": params,
                # "trainable": trainable,
                "shape": act, #.detach().cpu().numpy(),
                # "top_value": vals,
                # "top_index": idxs,
                "top_index_coords": top_coords,
                # "min_value": vals_min,
                # "min_index": idxs_min,
                # "min_index_coords": min_coords,
            }

        if not isinstance(module, torch.nn.ModuleList) and \
                not isinstance(module, torch.nn.Sequential) and \
                module != model:
            hooks.append(module.register_forward_hook(hook))

        return hook


    model.apply(register_hook)
    model.eval()
    with torch.no_grad():
        model(input_sample)
        render_queue.put(summary)
        summary.clear()
