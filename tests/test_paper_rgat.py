import torch
import torch.nn as nn


def test_relation_embedding_mlp_shapes():
    from src.model.paper_rgat import PaperMultiRelationGATLayer

    layer = PaperMultiRelationGATLayer(
        d_model=8,
        num_relations=7,
        num_node_types=5,
        dropout=0.0,
    )

    assert isinstance(layer.f_r, nn.Sequential)
    assert layer.f_r[0].in_features == 7 + 2 * 5
    assert layer.f_r[2].out_features == 8


def test_message_key_and_residual_shapes():
    from src.model.paper_rgat import PaperMultiRelationGATLayer

    layer = PaperMultiRelationGATLayer(d_model=8, num_relations=4, dropout=0.0)

    assert layer.f_m.in_features == 16
    assert layer.f_m.out_features == 8
    assert layer.f_k.in_features == 16
    assert layer.f_k.out_features == 8


def test_fh_has_batchnorm_and_two_linear_layers():
    from src.model.paper_rgat import PaperMultiRelationGATLayer

    layer = PaperMultiRelationGATLayer(d_model=8, num_relations=4, dropout=0.0)

    linear_layers = [module for module in layer.f_h if isinstance(module, nn.Linear)]
    batchnorm_layers = [
        module for module in layer.f_h if isinstance(module, nn.BatchNorm1d)
    ]
    assert len(linear_layers) == 2
    assert len(batchnorm_layers) == 1


def test_forward_shape_and_backward_no_nan():
    from src.model.paper_rgat import PaperMultiRelationGATLayer

    torch.manual_seed(0)
    layer = PaperMultiRelationGATLayer(d_model=8, num_relations=5, dropout=0.0)
    B, N, D = 2, 4, 8
    x = torch.randn(B, N, D, requires_grad=True)
    adj = torch.ones(B, N, N)
    edge_types = torch.randint(0, 5, (B, N, N), dtype=torch.long)
    node_types = torch.randint(0, 5, (B, N), dtype=torch.long)

    out = layer(x, adj, edge_types=edge_types, node_types_src=node_types)
    assert out.shape == (B, N, D)

    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_isolated_nodes_are_finite():
    from src.model.paper_rgat import PaperMultiRelationGATLayer

    torch.manual_seed(1)
    layer = PaperMultiRelationGATLayer(d_model=8, num_relations=3, dropout=0.0)
    B, N, D = 2, 4, 8
    x = torch.randn(B, N, D)
    adj = torch.ones(B, N, N)
    adj[:, -1, :] = 0
    edge_types = torch.zeros(B, N, N, dtype=torch.long)
    node_types = torch.zeros(B, N, dtype=torch.long)

    out = layer(x, adj, edge_types=edge_types, node_types_src=node_types)
    assert torch.isfinite(out).all()
