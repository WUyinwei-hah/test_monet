# test_monet
When trying train a MoNet with dgl, I got following errors:
Traceback (most recent call last):
  File "/home/Wuyinwei/Desktop/Distill_GCN_Experiment/test_monet.py", line 116, in <module>
    output = model(blocks, batch_inputs)
  File "/home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/Wuyinwei/Desktop/Distill_GCN_Experiment/test_monet.py", line 56, in forward
    h = self.layers[i](
  File "/home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/nn/pytorch/conv/gmmconv.py", line 220, in forward
    if (graph.in_degrees() == 0).any():
  File "/home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/heterograph.py", line 3433, in in_degrees
    v = self.dstnodes(dsttype)
  File "/home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/view.py", line 51, in __call__
    self._graph._graph.number_of_nodes(ntid),
  File "/home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/heterograph_index.py", line 376, in number_of_nodes
    return _CAPI_DGLHeteroNumVertices(self, int(ntype))
  File "dgl/_ffi/_cython/./function.pxi", line 295, in dgl._ffi._cy3.core.FunctionBase.__call__
  File "dgl/_ffi/_cython/./function.pxi", line 227, in dgl._ffi._cy3.core.FuncCall
  File "dgl/_ffi/_cython/./function.pxi", line 217, in dgl._ffi._cy3.core.FuncCall3
dgl._ffi.base.DGLError: [10:56:50] /opt/dgl/src/graph/./heterograph.h:67: Check failed: meta_graph_->HasVertex(vtype): Invalid vertex type: 1
Stack trace:
  [bt] (0) /home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/libdgl.so(dmlc::LogMessageFatal::~LogMessageFatal()+0x4f) [0x7fd7dd3ab30f]
  [bt] (1) /home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/libdgl.so(dgl::HeteroGraph::NumVertices(unsigned long) const+0xa2) [0x7fd7dd72e162]
  [bt] (2) /home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/libdgl.so(+0x733aed) [0x7fd7dd737aed]
  [bt] (3) /home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/libdgl.so(DGLFuncCall+0x48) [0x7fd7dd6bee58]
  [bt] (4) /home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/_ffi/_cy3/core.cpython-38-x86_64-linux-gnu.so(+0x163fc) [0x7fd8f9e943fc]
  [bt] (5) /home/Wuyinwei/anaconda3/envs/ldgl/lib/python3.8/site-packages/dgl/_ffi/_cy3/core.cpython-38-x86_64-linux-gnu.so(+0x1692b) [0x7fd8f9e9492b]
  [bt] (6) /home/Wuyinwei/anaconda3/envs/ldgl/bin/python(_PyObject_MakeTpCall+0x3eb) [0x4d14db]
  [bt] (7) /home/Wuyinwei/anaconda3/envs/ldgl/bin/python(_PyEval_EvalFrameDefault+0x4f48) [0x4cc578]
  [bt] (8) /home/Wuyinwei/anaconda3/envs/ldgl/bin/python() [0x4e8af7]
  
 
