def set_requires_grad(module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def count_parameters(model) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def apply_freeze_policy(model, freeze_policy: dict | None):
    """
    Apply an explicit freeze policy for controlled experiments.

    The baseline path remains untouched unless a policy is passed.
    """
    if not freeze_policy:
        return

    policy = dict(freeze_policy)

    if policy.get("freeze_all_baseline", False):
        baseline_modules = [
            "visual_proj",
            "question_encoder",
            "kg_node_proj",
            "node_type_emb",
            "gnn_layers",
            "readout_attn",
            "classifier",
        ]
        for name in baseline_modules:
            if hasattr(model, name):
                set_requires_grad(getattr(model, name), False)

    explicit_component_flags = {
        "freeze_visual_proj": "visual_proj",
        "freeze_question_encoder": "question_encoder",
        "freeze_kg_node_proj": "kg_node_proj",
        "freeze_node_type_emb": "node_type_emb",
        "freeze_gnn_layers": "gnn_layers",
        "freeze_readout": "readout_attn",
        "freeze_classifier": "classifier",
        "freeze_graph_link_module": "graph_link_module",
    }
    for flag_name, module_name in explicit_component_flags.items():
        if policy.get(flag_name, False) and hasattr(model, module_name):
            module = getattr(model, module_name)
            if module is not None:
                set_requires_grad(module, False)
