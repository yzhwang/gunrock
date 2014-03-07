// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * problem_base.cuh
 *
 * @brief Base struct for all the application types
 */

#pragma once

#include <gunrock/util/basic_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multiple_buffering.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/app/rp/rp_partitioner.cuh>
#include <vector>

namespace gunrock {
namespace app {

/**
 * @brief Enumeration of global frontier queue configurations
 */

enum FrontierType {
    VERTEX_FRONTIERS,       // O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,         // O(m) ping-pong global edge frontiers
    MIXED_FRONTIERS         // O(n) global vertex frontier, O(m) global edge frontier
};


/**
 * @brief Base problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 */
template <
    typename    _VertexId,
    typename    _SizeT,
    typename    _Value,
    bool        _USE_DOUBLE_BUFFER>

struct ProblemBase
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;
    typedef _Value              Value;

    /**
     * Load instruction cache-modifier const defines.
     */

    static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER                    = util::io::ld::cg;             // Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer
    static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER                   = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR column-indices.
    static const util::io::ld::CacheModifier EDGE_VALUES_READ_MODIFIER              = util::io::ld::NONE;           // Load instruction cache-modifier for reading edge values.
    static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER       = util::io::ld::cg;             // Load instruction cache-modifier for reading CSR row-offsets (8-byte aligned)
    static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER     = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR row-offsets (4-byte aligned)
    static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER                   = util::io::st::cg;             // Store instruction cache-modifier for writing outgoing frontier vertex-ids. Valid on SM2.0 or newer

    /**
     * @brief Graph slice structure which contains common graph structural data and input/output queue.
     */
    struct GraphSlice
    {
        //Slice Index
        int             index;
        Csr<VertexId,Value,SizeT>* graph;           //
        int             *partition_table;           //
        VertexId        *convertion_table;          //
        SizeT           *in_offset;                 //
        SizeT           *out_offset;                //

        SizeT           *d_row_offsets;             // CSR format row offset on device memory
        VertexId        *d_column_indices;          // CSR format column indices on device memory
        VertexId        *d_convertion_table;        // 
        int             *d_partition_table;         //
        SizeT           *d_in_offset;               //
        //SizeT           *d_out_offset;              //

        //Frontier queues. Used to track working frontier.
        util::DoubleBuffer<VertexId, VertexId>      frontier_queues;
        SizeT                                       frontier_elements[2];

        //Number of nodes and edges in slice
        VertexId        nodes;
        SizeT           edges;

        //CUDA stream to use for processing the slice
        cudaStream_t    stream;

        /**
         * @brief GraphSlice Constructor
         *
         * @param[in] index GPU index, reserved for multi-GPU use in future.
         * @param[in] stream CUDA Stream we use to allocate storage for this graph slice.
         */
        GraphSlice(int index, cudaStream_t stream)// :
            //index(index),
            //d_row_offsets(NULL),
            //d_column_indices(NULL),
            //d_foreign_nodes(NULL),
            //nodes(0),
            //edges(0),
            //stream(stream)
        {
            this->index        = index;
            graph              = NULL;
            partition_table    = NULL;
            convertion_table   = NULL;
            in_offset          = NULL;
            out_offset         = NULL;
            d_row_offsets      = NULL;
            d_column_indices   = NULL;
            d_convertion_table = NULL;
            d_partition_table  = NULL;
            d_in_offset        = NULL;
            //d_out_offset       = NULL;
            nodes              = 0;
            edges              = 0;
            this->stream       = stream;
            // Initialize double buffer frontier queue lengths
            for (int i = 0; i < 2; ++i)
            {
                frontier_elements[i] = 0;
                frontier_queues.d_keys[i]=NULL;
                frontier_queues.d_values[i]=NULL;
            }
        }

        /**
         * @brief GraphSlice Destructor to free all device memories.
         */
        virtual ~GraphSlice()
        {
            printf("~GraphSlice begin.\n"); fflush(stdout);
            // Set device (use slice index)
            util::GRError(cudaSetDevice(index), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);

            printf("1"); fflush(stdout);
            // Free pointers
            if (d_row_offsets     ) util::GRError(cudaFree(d_row_offsets     ), 
                                        "GpuSlice cudaFree d_row_offsets failed"     , __FILE__, __LINE__); printf("2"); fflush(stdout);
            if (d_column_indices  ) util::GRError(cudaFree(d_column_indices  ), 
                                        "GpuSlice cudaFree d_column_indices failed"  , __FILE__, __LINE__); printf("3"); fflush(stdout);
            if (d_convertion_table) util::GRError(cudaFree(d_convertion_table), 
                                        "GpuSlice cudaFree d_convertion_table failed", __FILE__, __LINE__); printf("4"); fflush(stdout);
            if (d_partition_table ) util::GRError(cudaFree(d_partition_table ),
                                        "GpuSlice cudaFree d_partition_table failed" , __FILE__, __LINE__); printf("5"); fflush(stdout);
            if (d_in_offset       ) util::GRError(cudaFree(d_in_offset       ),
                                        "GpuSlice cudaFree d_in_offset failed"       , __FILE__, __LINE__); printf("6"); fflush(stdout);
            //if (d_out_offset      ) util::GRError(cudaFree(d_out_offset      ),
            //                            "GpuSlice cudaFree d_out_offset failed"      , __FILE__, __LINE__); printf("7"); fflush(stdout);

            for (int i = 0; i < 2; ++i) {
                printf(" %d,%d ",i,frontier_elements[i]); fflush(stdout);
                //if (frontier_queues.d_keys  [i]) util::GRError(cudaFree(frontier_queues.d_keys  [i]), 
                //                                     "GpuSlice cudaFree frontier_queues.d_keys failed"  , __FILE__, __LINE__); printf("8"); fflush(stdout);
                if (frontier_queues.d_values[i]) util::GRError(cudaFree(frontier_queues.d_values[i]), 
                                                     "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__); printf("9"); fflush(stdout);
            }

            /*// Destroy stream
            if (stream) {
                util::GRError(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
            }*/
            printf("~GraphSlice end.\n"); fflush(stdout);
        }

        cudaError_t Init(
            bool                       stream_from_host,
            int                        num_gpus,
            Csr<VertexId,Value,SizeT>* graph,
            int*                       partition_table,
            VertexId*                  convertion_table,
            SizeT*                     in_offset,
            SizeT*                     out_offset)
        {
            printf("GPUSlice Init begin. \n"); fflush(stdout);
            cudaError_t retval     = cudaSuccess;
            this->graph            = graph;
            nodes                  = graph->nodes;
            edges                  = graph->edges;
            this->partition_table  = partition_table;
            this->convertion_table = convertion_table;
            this->in_offset        = in_offset;
            this->out_offset       = out_offset;

            do {
                if (retval = util::GRError(cudaSetDevice(index), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__)) break;
                //printf("1");fflush(stdout);
                if (stream_from_host) {
                    if (retval = util::GRError(cudaHostGetDevicePointer(
                                    (void **)&(d_row_offsets),
                                    (void * )&(graph->row_offsets), 0),
                                 "GpuSlice cudaHostGetDevicePointer d_row_offsets failed", __FILE__, __LINE__)) break;
                    if (retval = util::GRError(cudaHostGetDevicePointer(
                                    (void **)&(d_column_indices),
                                    (void * )&(graph->column_indices), 0),
                                 "GpuSlice cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) break;
                } else {
                    // Allocate and initialize d_row_offsets
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&(d_row_offsets),
                        (nodes+1) * sizeof(SizeT)),
                        "GraphSlice cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;

                    if (retval = util::GRError(cudaMemcpy(
                        d_row_offsets,
                        graph->row_offsets,
                        (nodes+1) * sizeof(SizeT),
                        cudaMemcpyHostToDevice),
                        "GraphSlice cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;

                    // Allocate and initialize d_column_indices
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&(d_column_indices),
                        edges * sizeof(VertexId)),
                        "GraphSlice cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;

                    if (retval = util::GRError(cudaMemcpy(
                        d_column_indices,
                        graph->column_indices,
                        edges * sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                        "GraphSlice cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;
                } //end if (stream_from_host)

                printf("2");fflush(stdout);
                if (num_gpus >1)
                {
                    // Allocate and initalize d_convertion_table
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&(d_partition_table),
                        nodes * sizeof(VertexId)),
                        "GraphSlice cudaMalloc d_partition_table failed", __FILE__, __LINE__)) break;
                    printf("3");fflush(stdout);

                    if (retval = util::GRError(cudaMemcpy(
                        d_partition_table,
                        partition_table,
                        nodes * sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                        "GraphSlice cudaMemcpy d_partition_table failed", __FILE__, __LINE__)) break;
                    printf("4");fflush(stdout);

                    // Allocate and initalize d_convertion_table
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&(d_convertion_table),
                        nodes * sizeof(VertexId)),
                        "GraphSlice cudaMalloc d_convertion_table failed", __FILE__, __LINE__)) break;
                    printf("5");fflush(stdout);

                    if (retval = util::GRError(cudaMemcpy(
                        d_convertion_table,
                        convertion_table,
                        nodes * sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                        "GraphSlice cudaMemcpy d_convertion_table failed", __FILE__, __LINE__)) break;
                    printf("6");fflush(stdout);  

                    // Allocate and initalize d_in_offset
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&(d_in_offset),
                        num_gpus * sizeof(SizeT)),
                        "GraphSlice cudaMalloc d_in_offset failed", __FILE__, __LINE__)) break;
                    printf("7"); fflush(stdout);

                    if (retval = util::GRError(cudaMemcpy(
                        d_in_offset,
                        in_offset,
                        num_gpus * sizeof(SizeT),
                        cudaMemcpyHostToDevice),
                        "GraphSlice cudaMemcpy d_in_offset failed", __FILE__, __LINE__)) break;
                    printf("8");fflush(stdout);

                    // Allocate and initalize d_out_offset
                    /*if (retval = util::GRError(cudaMalloc(
                        (void**)&(d_out_offset),
                        num_gpus * sizeof(SizeT)),
                        "GraphSlice cudaMalloc d_out_offset failed", __FILE__, __LINE__)) break;
                    printf("9"); fflush(stdout);

                    if (retval = util::GRError(cudaMemcpy(
                        d_out_offset,
                        out_offset,
                        num_gpus * sizeof(SizeT),
                        cudaMemcpyHostToDevice),
                        "GraphSlice cudaMemcpy d_out_offset failed", __FILE__, __LINE__)) break;
                    printf("A"); fflush(stdout);*/
               } // end if num_gpu>1
            } while (0);
            printf("GPUSlice Init finished.\n"); fflush(stdout);
            return retval;
        }
     
     /**
     * @brief Performs any initialization work needed for GraphSlice. Must be called prior to each search
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] queue_sizing Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
        double queue_sizing)            // Size scaling factor for work queue allocation
        {
            cudaError_t retval = cudaSuccess;

            // Set device
            if (retval = util::GRError(cudaSetDevice(index),
                             "GpuSlice cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            //
            // Allocate frontier queues if necessary
            //

            // Determine frontier queue sizes
            SizeT new_frontier_elements[2] = {0,0};

            switch (frontier_type) {
                case VERTEX_FRONTIERS :
                    // O(n) ping-pong global vertex frontiers
                    new_frontier_elements[0] = double(nodes) * queue_sizing;
                    new_frontier_elements[1] = new_frontier_elements[0];
                    break;

                case EDGE_FRONTIERS :
                    // O(m) ping-pong global edge frontiers
                    new_frontier_elements[0] = double(edges) * queue_sizing;
                    new_frontier_elements[1] = new_frontier_elements[0];
                    break;

                case MIXED_FRONTIERS :
                    // O(n) global vertex frontier, O(m) global edge frontier
                    new_frontier_elements[0] = double(nodes) * queue_sizing;
                    new_frontier_elements[1] = double(edges) * queue_sizing;
                    break;
             }

            // Iterate through global frontier queue setups
            for (int i = 0; i < 2; i++) {

                // Allocate frontier queue if not big enough
                if (frontier_elements[i] < new_frontier_elements[i]) {

                    // Free if previously allocated
                    if (frontier_queues.d_keys[i]) {
                        if (retval = util::GRError(cudaFree(frontier_queues.d_keys[i]),
                                         "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
                    }

                    // Free if previously allocated
                    if (_USE_DOUBLE_BUFFER) {
                        if (frontier_queues.d_values[i]) {
                            if (retval = util::GRError(cudaFree(frontier_queues.d_values[i]),
                                             "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
                        }
                    }

                    frontier_elements[i] = new_frontier_elements[i];
                    printf(" GraphicSlice %d,%d \n",i,frontier_elements[i]);fflush(stdout);

                    if (retval = util::GRError(cudaMalloc(
                                     (void**) &(frontier_queues.d_keys[i]),
                                     frontier_elements[i] * sizeof(VertexId)),
                                     "GpuSlice cudaMalloc frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
                    if (_USE_DOUBLE_BUFFER) {
                        if (retval = util::GRError(cudaMalloc(
                                     (void**) &(frontier_queues.d_values[i]),
                                     frontier_elements[i] * sizeof(VertexId)),
                                     "GpuSlice cudaMalloc frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
                    }
                } //end if
            } // end for i<2
            
            return retval;
        }

    };

    // Members
public:
    // Number of GPUs to be sliced over
    int                 num_gpus;

    // Device indices
    int                 *gpu_idx;

    // Size of the graph
    SizeT               nodes;
    SizeT               edges;

    // Set of graph slices (one for each GPU)
    GraphSlice**        graph_slices;

    // Subgraphs for multi-gpu implementation
    Csr<VertexId,Value,SizeT> *sub_graphs;

    // Partitioner
    PartitionerBase<VertexId,SizeT,Value> *partitioner;

    // Multi-gpu partition table and convertion table
    int                 **partition_tables;
    SizeT               **convertion_tables;
   
    // Offsets for data movement between GPUs
    SizeT               **in_offsets;
    SizeT               **out_offsets;               
    // Methods

    /**
     * @brief ProblemBase default constructor
     */
    ProblemBase() :
        num_gpus(0),
        nodes(0),
        edges(0)
    {
        partition_tables  = NULL;
        convertion_tables = NULL;
        partitioner       = NULL;
        sub_graphs        = NULL;
        in_offsets        = NULL;
        out_offsets       = NULL;
    }
    
    /**
     * @brief ProblemBase default destructor to free all graph slices allocated.
     */
    virtual ~ProblemBase()
    {
        printf("~ProblemBase begin.\n"); fflush(stdout);
        // Cleanup graph slices on the heap
        for (int i = 0; i < num_gpus; ++i)
        {
            delete   graph_slices     [i  ]; graph_slices     [i  ] = NULL;
            if (num_gpus > 1)
            {
                //delete   sub_graphs       [i  ]; sub_graphs       [i  ] = NULL;
                //delete[] partition_tables [i+1]; partition_tables [i+1] = NULL;
                free (partition_tables    [i+1]); partition_tables [i+1] = NULL;
                //delete[] convertion_tables[i+1]; convertion_tables[i+1] = NULL;
                free (convertion_tables   [i+1]); convertion_tables[i+1] = NULL;
                delete[] out_offsets      [i  ]; out_offsets      [i  ] = NULL;
                delete[] in_offsets       [i  ]; in_offsets       [i  ] = NULL;
            }
        }
        if (num_gpus > 1)
        {
            delete[] partition_tables [0];  partition_tables [0] = NULL;
            delete[] convertion_tables[0];  convertion_tables[0] = NULL;
            delete[] partition_tables;      partition_tables     = NULL;
            delete[] convertion_tables;     convertion_tables    = NULL;
            //delete[] graph_slices;          graph_slices         = NULL;
            delete[] out_offsets;           out_offsets          = NULL;
            delete[] in_offsets;            in_offsets           = NULL;
            delete   partitioner;           partitioner          = NULL;
            delete[] sub_graphs;            sub_graphs           = NULL;
        }
        delete[] graph_slices; graph_slices = NULL;
        delete[] gpu_idx;      gpu_idx      = NULL;
        printf("~ProblemBase end.\n"); fflush(stdout);
    }

    /**
     * @brief Get the GPU index for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Index of the gpu that owns the neighbor list of the specified vertex
     */
    template <typename VertexId>
    int GpuIndex(VertexId vertex)
    {
        if (num_gpus <= 1) {
            
            // Special case for only one GPU, which may be set as with
            // an ordinal other than 0.
            return graph_slices[0]->index;
        } else {
            //return vertex % num_gpus;
            return partition_tables[0][vertex];
        }
    }

    /**
     * @brief Get the row offset for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Row offset of the specified vertex. If a single GPU is used,
     * this will be the same as the vertex id.
     */
    template <typename VertexId>
    VertexId GraphSliceRow(VertexId vertex)
    {
        //return vertex / num_gpus;
        if (num_gpus <= 1) {
            return vertex;
        } else {
            return convertion_tables[0][vertex];
        }
    }

    /**
     * @brief Initialize problem from host CSR graph.
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] nodes Number of nodes in the CSR graph.
     * @param[in] edges Number of edges in the CSR graph.
     * @param[in] h_row_offsets Host-side row offsets array.
     * @param[in] h_column_indices Host-side column indices array.
     * @param[in] num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        bool        stream_from_host,
        std::string partition_method,
        /*SizeT       nodes,
        SizeT       edges,
        SizeT       *h_row_offsets,
        VertexId    *h_column_indices,*/
        Csr<VertexId,Value,SizeT> &graph,
        int         num_gpus,
        int*        gpu_idx
        //Csr<VertexId,Value,SizeT>* &sub_graph,
        //SizeT**     &incoming_offsets,
        //SizeT**     &outgoing_offsets,
        //int*        &partition_table,
        //SizeT*      &convertion_table)
        )
    {
        printf("ProblemBase Init begin.\n"); fflush(stdout);
        cudaError_t retval      = cudaSuccess;
        this->nodes             = graph.nodes;
        this->edges             = graph.edges;
        this->num_gpus          = num_gpus;
        this->gpu_idx           = new int [num_gpus];

       do {
            if (num_gpus==1 && gpu_idx[0]==-1)
            {
                if (retval = util::GRError(cudaGetDevice(&(this->gpu_idx[0])), "ProblemBase cudaGetDevice failed", __FILE__, __LINE__)) break;
            } else {
                for (int gpu=0;gpu<num_gpus;gpu++)
                    this->gpu_idx[gpu]=gpu_idx[gpu];
            }
            //printf("."); fflush(stdout);
            graph_slices  = new GraphSlice*[num_gpus];
            printf("graph_slices created.\n"); fflush(stdout);
            //if (num_gpus <= 1) {
                //SizeT    *h_row_offsets    = graph.row_offsets;
                //VertexId *h_column_indices = graph.column_indices;
                // Create a single graph slice for the currently-set gpu
                //int gpu;
                //if (retval = util::GRError(cudaGetDevice(&gpu), "ProblemBase cudaGetDevice failed", __FILE__, __LINE__)) break;
                //graph_slices[0] = new GraphSlice(gpu_idx[gpu], 0);
                //graph_slices[0]->nodes = nodes;
                //graph_slices[0]->edges = edges;

                //if (stream_from_host) {

                    // Map the pinned graph pointers into device pointers
                    //if (retval = util::GRError(cudaHostGetDevicePointer(
                    //                (void **)&graph_slices[0]->d_row_offsets,
                    //                (void *) h_row_offsets, 0),
                    //            "ProblemBase cudaHostGetDevicePointer d_row_offsets failed", __FILE__, __LINE__)) break;

                    //if (retval = util::GRError(cudaHostGetDevicePointer(
                    //                (void **)&graph_slices[0]->d_column_indices,
                    //                (void *) h_column_indices, 0),
                    //            "ProblemBase cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) break;
                //} else {

                    // Allocate and initialize d_row_offsets
                //    if (retval = util::GRError(cudaMalloc(
                //        (void**)&graph_slices[0]->d_row_offsets,
                //        (graph_slices[0]->nodes+1) * sizeof(SizeT)),
                //        "ProblemBase cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;

                //    if (retval = util::GRError(cudaMemcpy(
                //        graph_slices[0]->d_row_offsets,
                //        h_row_offsets,
                //        (graph_slices[0]->nodes+1) * sizeof(SizeT),
                //        cudaMemcpyHostToDevice),
                //        "ProblemBase cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
                    
                    // Allocate and initialize d_column_indices
                //    if (retval = util::GRError(cudaMalloc(
                //        (void**)&graph_slices[0]->d_column_indices,
                //        graph_slices[0]->edges * sizeof(VertexId)),
                //        "ProblemBase cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;

                //    if (retval = util::GRError(cudaMemcpy(
                //        graph_slices[0]->d_column_indices,
                //        h_column_indices,
                //        graph_slices[0]->edges * sizeof(VertexId),
                //        cudaMemcpyHostToDevice),
                //        "ProblemBase cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;



                //} //end if(stream_from_host)
            //} else {
                if (num_gpus >1)
                {
                    if (partition_method=="random") 
                        partitioner=new rp::RandomPartitioner<VertexId, SizeT, Value>(graph,num_gpus);
                    else util::GRError("partition_method invalid", __FILE__,__LINE__);
                    printf("partitioner created.\n");fflush(stdout);
                    retval = partitioner->Partition(
                        sub_graphs,
                        partition_tables,
                        convertion_tables,
                        in_offsets,
                        out_offsets);
                    printf("partitioner returned.\n");fflush(stdout);
                    //graph.DisplayGraph("original graph");
                    //util::cpu_mt::PrintCPUArray<SizeT,int  >("partition_table ", partition_tables [0], graph.nodes);
                    //util::cpu_mt::PrintCPUArray<SizeT,Value>("convertion_table", convertion_tables[0], graph.nodes);
                    /*for (int gpu=0;gpu<num_gpus;gpu++)
                    {
                        char name[128];
                        sprintf(name,"sub_graphs[%d]",gpu);
                        sub_graphs[gpu].DisplayGraph(name);
                        util::cpu_mt::PrintCPUArray<SizeT,int  >("partition_table ", partition_tables [gpu+1], sub_graphs[gpu].nodes);
                        util::cpu_mt::PrintCPUArray<SizeT,Value>("convertion_table", convertion_tables[gpu+1], sub_graphs[gpu].nodes);
                        util::cpu_mt::PrintCPUArray<SizeT,SizeT>("in_offsets      ", in_offsets       [gpu  ], num_gpus+1);
                        util::cpu_mt::PrintCPUArray<SizeT,SizeT>("out_offsets     ", out_offsets      [gpu  ], num_gpus+1);
                    }*/
                    if (retval) break;
                } else {
                    sub_graphs=&graph;
                }
                //this->sub_graphs       = sub_graphs;
                //this->incoming_offsets = incoming_offsets;
                //this->outgoing_offsets = outgoing_offsets;
                //this->partition_table  = partition_table;
                //this->convertion_table = convertion_table;
                for (int gpu=0;gpu<num_gpus;gpu++)
                {
                    printf("gpu_idx[%d] = %d\n", gpu, this->gpu_idx[gpu]); fflush(stdout);
                    graph_slices[gpu] = new GraphSlice(this->gpu_idx[gpu], 0);
                    printf("graph_slice[%d] created.\n", gpu); fflush(stdout);
                    if (num_gpus > 1)
                    {
                        printf("partition_tables [%d] = %p,%d\n",gpu+1,partition_tables [gpu+1],partition_tables [gpu+1][0]);fflush(stdout);
                        printf("convertion_tables[%d] = %p,%d\n",gpu+1,convertion_tables[gpu+1],convertion_tables[gpu+1][0]);fflush(stdout);
                        printf("in_offsets       [%d] = %p,%d\n",gpu  ,in_offsets       [gpu  ],in_offsets       [gpu  ][0]);fflush(stdout);
                        printf("out_offsets      [%d] = %p,%d\n",gpu  ,out_offsets      [gpu  ],out_offsets      [gpu  ][0]);fflush(stdout);
                        retval = graph_slices[gpu]->Init(
                            stream_from_host,
                            num_gpus,
                            &(sub_graphs[gpu]),
                            partition_tables [gpu+1],
                            convertion_tables[gpu+1],
                            in_offsets[gpu],
                            out_offsets[gpu]);
                    } else retval = graph_slices[gpu]->Init(
                            stream_from_host,
                            num_gpus,
                            &(sub_graphs[gpu]),
                            NULL,
                            NULL,
                            NULL,
                            NULL);
                    if (retval) break;
                    /*graph_slices[gpu]->nodes = sub_graphs[gpu].nodes;
                    graph_slices[gpu]->edges = sub_graphs[gpu].edges;
                    if (num_gpus >1)
                    {
                        incoming_offsets[gpu]=new SizeT[num_gpus+1];
                        outgoing_offsets[gpu]=new SizeT[num_gpus+1];
                        incoming_offsets[gpu][0]=0;
                        for (int tgpu=0;tgpu<num_gpus;tgpu++)
                            incoming_offsets[gpu][tgpu+1]=incoming_offsets[gpu][tgpu]+foreign_count[tgpu*(num_gpus+1)+gpu];
                        outgoing_offsets[gpu][num_gpus]=sub_graphs[gpu].nodes;
                        for (int tgpu=num_gpus-1;tgpu>=0;tgpu--)
                            outgoing_offsets[gpu][tgpu]=outgoing_offsets[gpu][tgpu+1]-foreign_count[gpu*(num_gpus+1)+tgpu];
                    }*/
                    //if (retval = util::GRError(cudaGetDevice(&gpu), "ProblemBase cudaGetDevice failed", __FILE__, __LINE__)) break;
               }// end for (gpu)
           // }//end if(num_gpu<=1)
        } while (0);

        return retval;
    }

    /**
     * @brief Performs any initialization work needed for ProblemBase. Must be called prior to each search
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] queue_sizing Sizing scaling factor for work queue allocation. 1.0 by default. Reserved for future use.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
        FrontierType frontier_type,     // The frontier type (i.e., edge/vertex/mixed)
        double queue_sizing)            // Size scaling factor for work queue allocation
        {
            cudaError_t retval = cudaSuccess;

            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                retval = graph_slices[gpu]->Reset(frontier_type,queue_sizing);
                if (retval) break;
                /*// Set device
                if (retval = util::GRError(cudaSetDevice(graph_slices[gpu]->index),
                            "ProblemBase cudaSetDevice failed", __FILE__, __LINE__)) return retval;


                //
                // Allocate frontier queues if necessary
                //

                // Determine frontier queue sizes
                SizeT new_frontier_elements[2] = {0,0};

                switch (frontier_type) {
                    case VERTEX_FRONTIERS :
                        // O(n) ping-pong global vertex frontiers
                        new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
                        new_frontier_elements[1] = new_frontier_elements[0];
                        break;

                    case EDGE_FRONTIERS :
                        // O(m) ping-pong global edge frontiers
                        new_frontier_elements[0] = double(graph_slices[gpu]->edges) * queue_sizing;
                        new_frontier_elements[1] = new_frontier_elements[0];
                        break;

                    case MIXED_FRONTIERS :
                        // O(n) global vertex frontier, O(m) global edge frontier
                        new_frontier_elements[0] = double(graph_slices[gpu]->nodes) * queue_sizing;
                        new_frontier_elements[1] = double(graph_slices[gpu]->edges) * queue_sizing;
                        break;

                    }

                // Iterate through global frontier queue setups
                for (int i = 0; i < 2; i++) {

                    // Allocate frontier queue if not big enough
                    if (graph_slices[gpu]->frontier_elements[i] < new_frontier_elements[i]) {

                        // Free if previously allocated
                        if (graph_slices[gpu]->frontier_queues.d_keys[i]) {
                            if (retval = util::GRError(cudaFree(
                                            graph_slices[gpu]->frontier_queues.d_keys[i]),
                                        "GpuSlice cudaFree frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
                        }

                        // Free if previously allocated
                        if (_USE_DOUBLE_BUFFER) {
                            if (graph_slices[gpu]->frontier_queues.d_values[i]) {
                                if (retval = util::GRError(cudaFree(
                                                graph_slices[gpu]->frontier_queues.d_values[i]),
                                            "GpuSlice cudaFree frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
                            }
                        }

                        graph_slices[gpu]->frontier_elements[i] = new_frontier_elements[i];

                        if (retval = util::GRError(cudaMalloc(
                                        (void**) &graph_slices[gpu]->frontier_queues.d_keys[i],
                                        graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId)),
                                    "ProblemBase cudaMalloc frontier_queues.d_keys failed", __FILE__, __LINE__)) return retval;
                        if (_USE_DOUBLE_BUFFER) {
                            if (retval = util::GRError(cudaMalloc(
                                            (void**) &graph_slices[gpu]->frontier_queues.d_values[i],
                                            graph_slices[gpu]->frontier_elements[i] * sizeof(VertexId)),
                                        "ProblemBase cudaMalloc frontier_queues.d_values failed", __FILE__, __LINE__)) return retval;
                        }
                    }
                }*/
            }
            
            return retval;
        }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
