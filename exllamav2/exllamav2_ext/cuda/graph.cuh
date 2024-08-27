#ifndef _graph_cuh
#define _graph_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#define MIN_GRAPH_INSTANCES 205

#if defined(USE_ROCM)
#define cudaGraphNode_t hipGraphNode_t
#define cudaKernelNodeParams hipKernelNodeParams
#define cudaGraphKernelNodeGetParams hipGraphKernelNodeGetParams
#define cudaGraphNodeType hipGraphNodeType
#define cudaGraphNodeGetType hipGraphNodeGetType
#define cudaGraphNodeTypeKernel hipGraphNodeTypeKernel
#define cudaGraphExecKernelNodeSetParams hipGraphExecKernelNodeSetParams
#endif

class Graph
{
public:
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    int current_node;
    int instances;
    int invoke_count;
//    std::vector<std::vector<int>> vars_node;
//    std::vector<std::vector<int>> vars_position;
    std::vector<std::tuple<int, int>> node_labels;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> node_params;
    std::vector<bool> node_needs_update;

    Graph();
    ~Graph();

    void begin_capture(cudaStream_t stream);
    void end_capture(cudaStream_t stream);
    void inspect_graph();
    bool count();
    bool ready() { return graph_exec != NULL; }
    void launch(cudaStream_t stream);
    void attach_label(cudaStream_t stream, int label, int sublabel);

    template <typename T>
    void update_param(int label, int sublabel, int param, T value, bool debug);

    void update_param_ptr(int label, int sublabel, int param, void* value, bool debug = false);
    void update_param_int(int label, int sublabel, int param, int value, bool debug = false);
};


#endif
