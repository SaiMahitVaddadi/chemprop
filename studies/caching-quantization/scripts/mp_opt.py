"""Optimized BondMessagePassing variants, full forward."""
import sys; sys.path.insert(0,'.')
import torch
from common import MP

class MPOpt(MP):
    """mode: current | expand | index_add | fused | fused_pad"""
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        self.padidx=None
    def set_pad(self, padidx): self.padidx=padidx
    def forward(self,V,E,ei,rev):
        nV=V.shape[0]; nE=ei.shape[1]; m=self.mode
        src,dst=ei[0],ei[1]
        H_0=self.W_i(torch.cat([torch.index_select(V,0,src),E],dim=1))
        H=torch.relu(H_0)
        Wt=self.W_h.weight.t()
        for _ in range(1,self.depth):
            if m=="fused_pad":
                Hp=torch.cat([H,torch.zeros(1,H.shape[1],dtype=H.dtype,device=H.device)],0)
                S=Hp[self.padidx].sum(1)
            else:
                S=torch.zeros(nV,H.shape[1],dtype=H.dtype,device=H.device).index_add_(0,dst,H)
            M=torch.index_select(S,0,src); M.sub_(torch.index_select(H,0,rev))
            H=torch.addmm(H_0,M,Wt).relu_()
        if m=="fused_pad":
            Hp=torch.cat([H,torch.zeros(1,H.shape[1],dtype=H.dtype,device=H.device)],0)
            M=Hp[self.padidx].sum(1)
        else:
            M=torch.zeros(nV,H.shape[1],dtype=H.dtype,device=H.device).index_add_(0,dst,H)
        return self.W_o(torch.cat((V,M),dim=1))
