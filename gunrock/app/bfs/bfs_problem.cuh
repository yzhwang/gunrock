// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_problem.cuh
 *
 * @brief GPU Storage management Structure for BFS Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief Breadth-First Search Problem structure stores device-side vectors for doing BFS computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _MARK_PREDECESSORS   Boolean type parameter which defines whether to mark predecessor value for each node.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 */
template <
    typename    VertexId,                       
    typename    SizeT,                          
    typename    Value,                          
    bool        _MARK_PREDECESSORS,
    bool        _ENABLE_IDEMPOTENCE,             
    bool        _USE_DOUBLE_BUFFER>
struct BFSProblem : ProblemBase<VertexId, SizeT, Value,
                                _USE_DOUBLE_BUFFER>
{

    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE; 

    //Helper structures

    /**
     * @brief Data slice structure which contains BFS problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        VertexId        *d_labels;              /**< Used for source distance */
        VertexId        *d_preds;               /**< Used for predecessor */
        unsigned char   *d_visited_mask;        /**< used for bitmask for visited nodes */
        int             num_associate,gpu_idx;
        VertexId        **d_associate_in;
        VertexId        **h_associate_in;
        VertexId        **d_associate_out;
        VertexId        **h_associate_out;
        VertexId        **d_associate_org;
        VertexId        **h_associate_org;
        SizeT           * d_out_length;
        SizeT           * out_length;
        SizeT           * in_length;
        VertexId        * d_keys_in;

        DataSlice()
        {
            d_labels        = NULL;
            d_preds         = NULL;
            num_associate   = 0;
            gpu_idx         = 0;
            d_associate_in  = NULL;
            h_associate_in  = NULL;
            d_associate_out = NULL;
            h_associate_out = NULL;
            d_associate_org = NULL;
            h_associate_org = NULL;
            d_out_length    = NULL;
            out_length      = NULL;
            in_length       = NULL;
            d_keys_in       = NULL;
        }

        ~DataSlice()
        {
            printf("~DataSlice begin.\n"); fflush(stdout);
            util::GRError(cudaSetDevice(gpu_idx),
                "~DataSlice cudaSetDevice failed", __FILE__, __LINE__);
            if (d_labels) util::GRError(cudaFree(d_labels), "~DataSlice cudaFree d_labels failed", __FILE__, __LINE__);
            if (d_preds ) util::GRError(cudaFree(d_preds ), "~DataSlice cudaFree d_preds failed" , __FILE__, __LINE__);
            d_labels = NULL; d_preds = NULL;

            if (h_associate_in != NULL)
            {
                for (int i=0;i<num_associate;i++)
                util::GRError(cudaFree(h_associate_in[i]), "~DataSlice cudaFree h_associate_in failed", __FILE__, __LINE__);
                util::GRError(cudaFree(d_associate_in   ), "~DataSlice cudaFree d_associate_in failed", __FILE__, __LINE__);
                util::GRError(cudaFree(d_keys_in        ), "_DataSlice cudaFree d_keys_in failed",      __FILE__, __LINE__);
                delete[] h_associate_in;
                delete[] in_length;
                h_associate_in = NULL;
                d_associate_in = NULL;
                in_length      = NULL;
                d_keys_in      = NULL;
            }

            if (h_associate_out != NULL)
            {
                for (int i=0;i<num_associate;i++)
                util::GRError(cudaFree(h_associate_out[i]), "~DataSlice cudaFree h_associate_out failed", __FILE__, __LINE__);
                util::GRError(cudaFree(d_associate_out   ), "~DataSlice cudaFree d_associate_out failed", __FILE__, __LINE__);
                util::GRError(cudaFree(d_out_length      ), "~DataSlice cudaFree d_out_length failed",    __FILE__, __LINE__);
                delete[] h_associate_out;
                delete[] out_length;
                h_associate_out = NULL;
                d_associate_out = NULL;
                d_out_length    = NULL;
                out_length      = NULL;
            }

            if (h_associate_org != NULL)
            {
                util::GRError(cudaFree(d_associate_org), "~DataSlice cudaFree d_associate_org failed", __FILE__, __LINE__);
                delete[] h_associate_org;
                d_associate_org = NULL;
                h_associate_org = NULL;
            }
            printf("~DataSlice end.\n"); fflush(stdout);
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_associate,
            SizeT num_nodes,
            SizeT num_in_nodes,
            SizeT num_out_nodes)
        {
            printf("DataSlice Init begin."); fflush(stdout);

            cudaError_t retval = cudaSuccess;
            this->gpu_idx       = gpu_idx;
            this->num_associate = num_associate;
            if (retval = util::GRError(cudaSetDevice(gpu_idx), "DataSlice cudaSetDevice failed", __FILE__, __LINE__)) return retval;
            // Create SoA on device
            if (retval = util::GRError(cudaMalloc((void**)&(d_labels),num_nodes * sizeof(VertexId)),
                            "DataSlice cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;

            if (_MARK_PREDECESSORS) 
            {
                if (retval = util::GRError(cudaMalloc((void**)&(d_preds),num_nodes * sizeof(VertexId)),
                                "DataSlice cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
            }

            if (num_associate != 0)
            {
                h_associate_org = new VertexId*[num_associate];
                h_associate_org[0] = d_labels;
                h_associate_org[1] = d_preds;
                if (retval = util::GRError(cudaMalloc((void**)&(d_associate_org), num_associate * sizeof(VertexId*)),
                                "DataSlice cudaMalloc d_associate_org failed", __FILE__, __LINE__)) return retval;
                if (retval = util::GRError(cudaMemcpy(d_associate_org, h_associate_org, 
                                num_associate * sizeof(VertexId*), cudaMemcpyHostToDevice),
                                "DataSlice cudaMemcpy d_associate_org failed", __FILE__, __LINE__)) return retval;
            }
            // Create incoming buffer on device
            if (num_in_nodes > 0)
            {
                h_associate_in = new VertexId*[num_associate];
                for (int i=0;i<num_associate;i++)
                {
                    if (retval = util::GRError(cudaMalloc((void**)&(h_associate_in[i]),num_in_nodes * sizeof(VertexId)),
                                    "DataSlice cudamalloc h_associate_in failed", __FILE__, __LINE__)) break;
                }
                if (retval) return retval;
                if (retval = util::GRError(cudaMalloc((void**)&(d_associate_in),num_associate * sizeof(VertexId*)),
                                "DataSlice cuaaMalloc d_associate_in failed", __FILE__, __LINE__)) return retval;
                if (retval = util::GRError(cudaMalloc((void**)&(d_keys_in     ), num_in_nodes * sizeof(VertexId)),
                                "DataSlice cudaMalloc d_key_in failed",       __FILE__, __LINE__)) return retval;
                if (retval = util::GRError(cudaMemcpy(d_associate_in, h_associate_in,
                                num_associate * sizeof(VertexId*),cudaMemcpyHostToDevice),
                                "DataSlice cudaMemcpy d_associate_in failed", __FILE__, __LINE__)) return retval;
                in_length  = new SizeT[num_gpus];
                //if (retval = util::GRError(cudaMalloc((void**)&(d_in_length),  num_gpus * sizeof(SizeT)),
                //                "DataSlice cudaMalloc d_in_length failed",    __FILE__, __LINE__)) return retval;
            }

             // Create outgoing buffer on device
            if (num_out_nodes > 0)
            {
                h_associate_out = new VertexId*[num_associate];
                for (int i=0;i<num_associate;i++)
                {
                    if (retval = util::GRError(cudaMalloc((void**)&(h_associate_out[i]),num_out_nodes * sizeof(VertexId)),
                                     "DataSlice cudamalloc h_associate_out failed", __FILE__, __LINE__)) break;
                }
                if (retval) return retval;
                if (retval = util::GRError(cudaMalloc((void**)&(d_associate_out),num_associate * sizeof(VertexId*)),
                                "DataSlice cuaaMalloc d_associate_out failed", __FILE__, __LINE__)) return retval;
                if (retval = util::GRError(cudaMemcpy(d_associate_out, h_associate_out,
                                num_associate * sizeof(VertexId*),cudaMemcpyHostToDevice),
                                "DataSlice cudaMemcpy d_associate_out failed", __FILE__, __LINE__)) return retval;
                out_length = new SizeT[num_gpus];
                if (retval = util::GRError(cudaMalloc((void**)&(d_out_length), num_gpus * sizeof(SizeT)),
                                "DataSlice cuaMalloc d_out_length failed",     __FILE__, __LINE__)) return retval;
            }
            printf("DataSlice Init finished."); fflush(stdout);
            return retval;
        }
    };

    // Members
    
    // Number of GPUs to be sliced over
//    int                 num_gpus;

    // Size of the graph
//    SizeT               nodes;
//    SizeT               edges;

    // Set of data slices (one for each GPU)
    DataSlice           **data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    DataSlice           **d_data_slices;

    // Device indices for each data slice
//    int                 *gpu_idx;

    // Subgraphs for multi-gpu implementation
//    Csr<VertexId, Value, SizeT> *sub_graphs;

    // Multi-gpu partition table and convertion table
//    int                 *partition_table;
//    SizeT               *convertion_table;

    // Offsets for data movement between GPUs
//    SizeT               **outgoing_offsets;
//    SizeT               **incoming_offsets;

    // Methods

    /**
     * @brief BFSProblem default constructor
     */

    BFSProblem()//:
    //nodes(0),
    //edges(0),
    //num_gpus(0) 
    {
        data_slices      = NULL;
        d_data_slices    = NULL;
        //gpu_idx          = NULL;
        //sub_graphs       = NULL;
        //incoming_offsets = NULL;
        //outgoing_offsets = NULL;
    }

    /**
     * @brief BFSProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    BFSProblem(bool        stream_from_host,       // Only meaningful for single-GPU
               std::string partition_method,
               const Csr<VertexId, Value, SizeT> &graph,
               int         num_gpus,
               int*        gpu_idx)// :
        //num_gpus(num_gpus)
    {
        Init(
            stream_from_host,
            partition_method,
            graph,
            num_gpus,
            gpu_idx);
    }

    /**
     * @brief BFSProblem default destructor
     */
    ~BFSProblem()
    {
        printf("~BFSProblem begin.\n");fflush(stdout);
        for (int i = 0; i < this->num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(this->gpu_idx[i]),
                "~BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            //if (data_slices[i]->d_labels)    util::GRError(cudaFree(data_slices[i]->d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
            //if (data_slices[i]->d_preds)     util::GRError(cudaFree(data_slices[i]->d_preds),  "GpuSlice cudaFree d_preds failed", __FILE__, __LINE__);
            //if (data_slices[i]->d_ibuffer)   util::GRError(cudaFree(data_slices[i]->d_ibuffer), "GpuSlice cudaFree d_ibuffer failed",__FILE__, __LINE__);
            //if (data_slices[i]->d_obuffer)   util::GRError(cudaFree(data_slices[i]->d_obuffer), "GpuSlice cudaFree d_obuffer failed", __FILE__, __LINE__);
            if (d_data_slices[i]) util::GRError(cudaFree(d_data_slices[i]), "~BFSProblem cudaFree data_slices failed", __FILE__, __LINE__);
        }
        if (d_data_slices) delete[] d_data_slices;
        if (data_slices  ) delete[] data_slices;
        printf("~BFSProblem end.\n");fflush(stdout);
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
     *
     * @param[out] h_labels host-side vector to store computed node labels (distances from the source).
     * @param[out] h_preds host-side vector to store predecessor vertex ids.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(VertexId *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (this->num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(this->gpu_idx[0]),
                            "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_labels,
                                data_slices[0]->d_labels,
                                sizeof(VertexId) * this->nodes,
                                cudaMemcpyDeviceToHost),
                            "BFSProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;

                if (_MARK_PREDECESSORS) {
                    if (retval = util::GRError(cudaMemcpy(
                                    h_preds,
                                    data_slices[0]->d_preds,
                                    sizeof(VertexId) * this->nodes,
                                    cudaMemcpyDeviceToHost),
                                "BFSProblem cudaMemcpy d_preds failed", __FILE__, __LINE__)) break;
                }

            } else {
                VertexId **th_labels=new VertexId*[this->num_gpus];
                VertexId **th_preds =new VertexId*[this->num_gpus];
                for (int gpu=0;gpu<this->num_gpus;gpu++)
                {
                    th_labels[gpu]=new VertexId[this->out_offsets[gpu][1]];
                    th_preds [gpu]=new VertexId[this->out_offsets[gpu][1]];

                    if (util::GRError(cudaSetDevice(this->gpu_idx[gpu]),
                                "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
                    if (retval = util::GRError(cudaMemcpy(
                                    th_labels[gpu],
                                    data_slices[gpu]->d_labels,
                                    sizeof(VertexId) * this->out_offsets[gpu][1],
                                    cudaMemcpyDeviceToHost),
                                "BFSProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
                    if (_MARK_PREDECESSORS) {
                        if (retval = util::GRError(cudaMemcpy(
                                        th_preds[gpu],
                                        data_slices[gpu]->d_preds,
                                        sizeof(VertexId) * this->out_offsets[gpu][1],
                                        cudaMemcpyDeviceToHost),
                                     "BFSProblem cudaMemcpy d_preds failed", __FILE__, __LINE__)) break;
                    }
                } //end for(gpu)
                for (VertexId node=0;node<this->nodes;node++)
                    h_labels[node]=th_labels[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                if (_MARK_PREDECESSORS)
                    for (VertexId node=0;node<this->nodes;node++)
                        h_preds[node]=th_preds[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                for (int gpu=0;gpu<this->num_gpus;gpu++)
                {
                    delete[] th_labels[gpu];th_labels[gpu]=NULL;
                    delete[] th_preds [gpu];th_preds [gpu]=NULL;
                }
                delete[] th_labels;th_labels=NULL;
                delete[] th_preds ;th_preds =NULL;
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief BFSProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool            stream_from_host,       // Only meaningful for single-GPU
            std::string     partition_method,
            Csr<VertexId, Value, SizeT> &graph,
            int             num_gpus,
            int*            gpu_idx)
    {
        //this->num_gpus = _num_gpus;
        //this->nodes    = graph.nodes;
        //this->edges    = graph.edges;
        //VertexId *h_row_offsets = graph.row_offsets;
        //VertexId *h_column_indices = graph.column_indices;
        printf("BFSProblem Init begin.\n"); fflush(stdout);

        ProblemBase<VertexId, SizeT,Value,_USE_DOUBLE_BUFFER>::Init(
            stream_from_host,
            partition_method,
            /*nodes,
            edges,
            h_row_offsets,
            h_column_indices,*/
            graph,
            num_gpus,gpu_idx
            //sub_graphs,
            //incoming_offsets,
            //outgoing_offsets,
            //partition_table,
            //convertion_table);
            );
       
        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices   = new DataSlice*[this->num_gpus];
        d_data_slices = new DataSlice*[this->num_gpus];

        do {
            //if (this->num_gpus <= 1) {
                //gpu_idx = (int*)malloc(sizeof(int));
                // Create a single data slice for the currently-set gpu
                //int gpu;
                //if (retval = util::GRError(cudaGetDevice(&gpu), "BFSProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                //gpu_idx[0] = gpu;

                //data_slices[0] = new DataSlice;
                //if (retval = util::GRError(cudaMalloc(
                //                (void**)&d_data_slices[0],
                //                sizeof(DataSlice)),
                //            "BFSProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                //VertexId    *d_labels;
                //if (retval = util::GRError(cudaMalloc(
                //        (void**)&d_labels,
                //        this->nodes * sizeof(VertexId)),
                //    "BFSProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                //data_slices[0]->d_labels = d_labels;
 
                //VertexId   *d_preds = NULL;
                //if (_MARK_PREDECESSORS) {
                //    if (retval = util::GRError(cudaMalloc(
                //        (void**)&d_preds,
                //        this->nodes * sizeof(VertexId)),
                //    "BFSProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                //}
                //data_slices[0]->d_preds = d_preds;
            //} else {
                //gpu_idx = (int*) malloc(sizeof(int)*this->num_gpus);
                //if (num_gpus==1) this->sub_graphs[0]=graph;

                for (int gpu=0;gpu<this->num_gpus;gpu++)
                {
                    if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]), "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
                    //gpu_idx    [gpu] = gpu;
                    data_slices[gpu] = new DataSlice;
                    if (retval = util::GRError(cudaMalloc(
                                    (void**)&(d_data_slices[gpu]),
                                    sizeof(DataSlice)),
                                 "BFSProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                    if (this->num_gpus > 1)
                    {
                        data_slices[gpu]->Init(this->num_gpus, this->gpu_idx[gpu], 2, 
                            this->sub_graphs[gpu].nodes,
                            this->graph_slices[gpu]->in_offset[this->num_gpus],
                            this->graph_slices[gpu]->out_offset[this->num_gpus]);
                    } else {
                        data_slices[gpu]->Init(this->num_gpus,this->gpu_idx[gpu], 0,
                            this->sub_graphs[gpu].nodes, 0, 0);
                    } 
                } //end for(gpu)
            //} // end if (num_gpus)
        } while (0);
        
        printf("BFSProblem Inited.\n"); fflush(stdout);
        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for BFS problem type. Must be called prior to each BFS run.
     *
     *  @param[in] src Source node for one BFS computing pass.
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     *  @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
            VertexId    src,
            FrontierType frontier_type,             // The frontier type (i.e., edge/vertex/mixed)
            double queue_sizing)                    // Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively). 0.0 is unspecified.
    {
        typedef ProblemBase<VertexId, SizeT,Value,
                                _USE_DOUBLE_BUFFER> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]),
                        "BSFProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output labels if necessary
            if (!data_slices[gpu]->d_labels) {
                VertexId    *d_labels;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_labels,
                                this->sub_graphs[gpu].nodes * sizeof(VertexId)),
                            "BFSProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_labels = d_labels;
            }

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_labels, -1, this->sub_graphs[gpu].nodes);

            // Allocate preds if necessary
            if (_MARK_PREDECESSORS && !data_slices[gpu]->d_preds) {
                VertexId    *d_preds;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_preds,
                                this->sub_graphs[gpu].nodes * sizeof(VertexId)),
                            "BFSProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_preds = d_preds;
            }
            
            if (_MARK_PREDECESSORS)
                util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_preds, -2, this->sub_graphs[gpu].nodes);

                
            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "BFSProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;

        }

        
        // Fillin the initial input_queue for BFS problem
        int gpu;
        VertexId tsrc;
        if (this->num_gpus <= 1) 
        {
           gpu=0;tsrc=src;
        } else {
            gpu = this->partition_tables[0][src];
            tsrc= this->convertion_tables[0][src];
        }
        if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]), "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

        if (retval = util::GRError(cudaMemcpy(
                        BaseProblem::graph_slices[gpu]->frontier_queues.d_keys[0],
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->d_labels+tsrc,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        if (_MARK_PREDECESSORS) {
            VertexId src_pred = -1; 
            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_preds+tsrc,
                            &src_pred,
                            sizeof(VertexId),
                            cudaMemcpyHostToDevice),
                        "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        }

        return retval;
    }

    /** @} */

};

} //namespace bfs
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
