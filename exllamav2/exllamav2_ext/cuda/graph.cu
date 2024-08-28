#include "graph.cuh"
#include "util.cuh"
#include "../config.h"
#include <iostream>

Graph::Graph()
{
    graph = NULL;
    graph_exec = NULL;
    invoke_count = 0;

}

Graph::~Graph()
{
    if (graph) cudaGraphDestroy(graph);
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
}

bool Graph::count()
{
    invoke_count++;
    return invoke_count == MIN_GRAPH_INSTANCES;
}

void Graph::begin_capture(cudaStream_t stream)
{
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
}

void Graph::end_capture(cudaStream_t stream)
{
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
//    inspect_graph();

    // Get nodes

    size_t num_nodes;
    cudaGraphGetNodes(graph, nullptr, &num_nodes);
    nodes.resize(num_nodes);
    cudaGraphGetNodes(graph, nodes.data(), &num_nodes);

    // Get params

    node_params.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
    {
        if (std::get<0>(node_labels[i]) == 0) continue;
        cudaGraphKernelNodeGetParams(nodes[i], &node_params[i]);
    }

    node_needs_update.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        node_needs_update[i] = false;
}

void Graph::inspect_graph()
{
    // Get the number of nodes in the graph

    size_t numNodes;
    cudaGraphGetNodes(graph, nullptr, &numNodes);

    // Get the nodes in the graph

    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    DBGI(nodes.size());

    // Inspect each node

    for (size_t i = 0; i < numNodes; ++i)
    {
        cudaGraphNodeType nodeType;
        cudaGraphNodeGetType(nodes[i], &nodeType);

        if (nodeType == cudaGraphNodeTypeKernel)
        {
            cudaKernelNodeParams nodeParams;
            cudaGraphKernelNodeGetParams(nodes[i], &nodeParams);
            std::cout << "Kernel node " << i << ":" << std::endl;
            std::cout << "  Function pointer: " << nodeParams.func << std::endl;
            std::cout << "  Grid dimensions: (" << nodeParams.gridDim.x << ", " << nodeParams.gridDim.y << ", " << nodeParams.gridDim.z << ")" << std::endl;
            std::cout << "  Block dimensions: (" << nodeParams.blockDim.x << ", " << nodeParams.blockDim.y << ", " << nodeParams.blockDim.z << ")" << std::endl;
            std::cout << "  Shared memory: " << nodeParams.sharedMemBytes << " bytes" << std::endl;

        } else {
            std::cout << "Node " << i << " is not a kernel node." << std::endl;
        }
    }
}

void Graph::launch(cudaStream_t stream)
{
    for (int i = 0; i < nodes.size(); ++i)
    {
        if (!node_needs_update[i]) continue;
        cudaGraphExecKernelNodeSetParams(graph_exec, nodes[i], &node_params[i]);
        node_needs_update[i] = false;
    }

    cudaGraphLaunch(graph_exec, stream);
}

//void Graph::attach_label(cudaStream_t stream, int label, int sublabel)
//{
//    // Get the current capturing graph
//
//    cudaGraph_t capturing_graph;
//    cudaStreamCaptureStatus capture_status;
//    cudaStreamGetCaptureInfo(stream, &capture_status, NULL, &capturing_graph, NULL, NULL);
//
//    // Get the index of the last captured (kernel) node
//
//    size_t numNodes;
//    cudaGraphGetNodes(capturing_graph, nullptr, &numNodes);
//    int node_idx = (int)numNodes - 1;
//
//    // Skip unlabeled kernels
//
//    while (node_labels.size() < node_idx)
//        node_labels.push_back(std::tuple<int, int>(0, 0));
//
//    // Set label
//
//    node_labels.push_back(std::tuple<int, int>(label, sublabel));
//}

void Graph::attach_label(cudaStream_t stream, int label, int sublabel)
{
    node_labels.push_back(std::tuple<int, int>(label, sublabel));
}

template <typename T>
void Graph::update_param(int label, int sublabel, int param, T value, bool debug)
{
    for (int i = 0; i < node_labels.size(); ++i)
    {
        if (std::get<0>(node_labels[i]) != label || (sublabel && std::get<1>(node_labels[i]) != sublabel)) continue;

        T* p_old_value = (T*) node_params[i].kernelParams[param];
        if (*p_old_value == value) continue;
        *p_old_value = value;

        node_needs_update[i] = true;

        if (debug)
        {
            printf("-----------------------------------------------------\n");
            printf("UPDATED: ");
            DBGI(i);
            inspect_graph();
        }
    }
}

void Graph::update_param_ptr(int label, int sublabel, int param, void* value, bool debug)
{
    update_param<void*>(label, sublabel, param, value, debug);
}

void Graph::update_param_int(int label, int sublabel, int param, int value, bool debug)
{
    update_param<int>(label, sublabel, param, value, debug);
}
