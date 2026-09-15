"""Same-dtype token lookup in CPU memory; no learned or quantized-value changes."""
import torch
from torch import nn
from torch.nn import functional as F

class CPUResidentEmbedding(nn.Embedding):
    def __init__(self,original):
        if original.max_norm is not None:raise ValueError('Renormalizing embeddings are unsupported')
        super().__init__(original.num_embeddings,original.embedding_dim,padding_idx=original.padding_idx,
                         max_norm=None,norm_type=original.norm_type,scale_grad_by_freq=original.scale_grad_by_freq,
                         sparse=original.sparse,_weight=original.weight.detach().to(device='cpu'),_freeze=True)
        self.eval()
    def forward(self,input_ids):
        values=F.embedding(input_ids.to(device='cpu'),self.weight,self.padding_idx,None,self.norm_type,
                           self.scale_grad_by_freq,self.sparse)
        return values.to(device=input_ids.device)

def install(model,*,allow_tied_diagnostic=False):
    original=model.get_input_embeddings();head=model.get_output_embeddings()
    if not isinstance(original,nn.Embedding) or isinstance(original,CPUResidentEmbedding):
        raise ValueError('Expected original unmodified embedding')
    if original.weight.dtype!=torch.float16:raise ValueError('Frozen FP16 embedding required')
    tied=original.weight.data_ptr()==head.weight.data_ptr() and original.weight.device==head.weight.device
    if tied and not allow_tied_diagnostic:raise ValueError('Production memory policy requires untied embeddings')
    ids=torch.tensor([0,1,151643,151644,151645,original.num_embeddings-1],device=original.weight.device)
    expected=original(ids).detach().clone();scalar_expected=original(ids[0]).detach().clone()
    replacement=CPUResidentEmbedding(original)
    if not torch.equal(expected,replacement(ids)) or not torch.equal(scalar_expected,replacement(ids[0])):
        raise ValueError('CPU lookup differs from original token values')
    model.set_input_embeddings(replacement)
    names=[name for name,module in model.named_modules() if module is replacement]
    if len(names)!=1:raise ValueError('Ambiguous installed embedding')
    model.hf_device_map={**getattr(model,'hf_device_map',{}),names[0]:'cpu'}
    return dict(policy='CPU FP16 token lookup, result returned to input device',module=names[0],dtype=str(replacement.weight.dtype),
                parameter_bytes=replacement.weight.numel()*replacement.weight.element_size(),tied_source=tied,
                tied_diagnostic_only=bool(tied),lookup_check_equal=True,state_key=names[0]+'.weight')
