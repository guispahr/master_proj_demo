
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from models.modules import PointModule

class _CheckpointBlock(PointModule):
    """Wraps a Block with gradient checkpointing to trade compute for activation memory."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, point):
        # Save the spconv tensor BEFORE the block runs. spconv.SparseConvTensor is not a
        # registered pytree type, so grad_checkpoint(use_reentrant=False) cannot save/restore
        # it automatically - only standard tensors (e.g. point.feat) are handled by pytree.
        # Without this, sparse_conv_feat.features is stale during backward recomputation,
        # causing self.cpe (SubMConv3d) to run on post-block features instead of pre-block ones.
        saved_spconv = point.sparse_conv_feat

        def run(feat):
            # Restore point to its pre-block state before recomputing.
            point.feat = feat
            point.sparse_conv_feat = saved_spconv.replace_feature(feat)
            return self.block(point).feat

        new_feat = grad_checkpoint(run, point.feat, use_reentrant=False)
        # Replace point's tensors with the checkpointed versions (which carry gradient hooks).
        point.feat = new_feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(new_feat)
        return point
    
def cond_checkpoint(func, *args, use_checkpoint=True, **kwargs):
    if use_checkpoint:
        return grad_checkpoint(func, *args, **kwargs)
    else:
        return func(*args)