from src.model.gnn_core import DenseGATLayer, HFQuestionEncoder
from src.model.gqa_model import GQAVQAGNNModel
from src.model.paper_rgat import PaperMultiRelationGATLayer
from src.model.paper_vqa_gnn import PaperGQAModel, PaperVCRModel
from src.model.vcr_model import VCRVQAGNNModel
from src.model.vqa_gnn import VQAGNNModel

__all__ = [
    # Paper-aligned task-specific models (use these in training configs)
    "VCRVQAGNNModel",   # VCR Q→A and QA→R: candidate scoring (output dim=1)
    "GQAVQAGNNModel",   # GQA: answer vocabulary classification (output dim=num_answers)
    # Paper-equation modules. GQA is runnable via the two-subgraph dataset
    # contract; VCR still needs paper-path data artifacts.
    "PaperMultiRelationGATLayer",
    "PaperVCRModel",
    "PaperGQAModel",
    # Shared GNN primitives
    "DenseGATLayer",
    "HFQuestionEncoder",
    # Backward-compatible monolithic model (kept for internal use / testing)
    "VQAGNNModel",
]
