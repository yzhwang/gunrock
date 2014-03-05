// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * partitioner_base.cuh
 *
 * @brief Base struct for all the partitioner types
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

//#include <gunrock/app/problem_base.cuh>
#include <vector>

namespace gunrock {
namespace app {

/**
 * @brief Base partitioner structure.
 *
 */

template <
    typename   _VertexId,
    typename   _SizeT,
    typename   _Value>
struct PartitionerBase
{
    typedef _VertexId  VertexId;
    typedef _SizeT     SizeT;
    typedef _Value     Value;
    typedef Csr<VertexId,Value,SizeT> GraphT;

    // Members
public:
    // Number of GPUs to be partitioned
    int        num_gpus;
    int        Status;

    // Original graph
    const GraphT *graph;

    // Partioned graphs
    GraphT *sub_graphs;

    int       **partition_tables;
    VertexId  **convertion_tables;
    SizeT     **in_offsets;
    SizeT     **out_offsets;
    //Mthods

    template <
        typename VertexId,
        typename SizeT,
        typename Value>
    struct ThreadSlice
    {
    public:
        const GraphT     *graph;
        GraphT     *sub_graph;
        int        thread_num,num_gpus;
        CUTBarrier *cpu_barrier;
        CUTThread  thread_Id;
        int        *partition_table0,**partition_table1;
        VertexId   *convertion_table0,**convertion_table1;
        SizeT      **in_offsets,**out_offsets;
    };

    /**
     * @brief PartitionerBase default constructor
     */
    PartitionerBase()
    {
        Status            = 0;
        num_gpus          = 0;
        graph             = NULL;
        sub_graphs        = NULL;
        partition_tables  = NULL;
        convertion_tables = NULL;
        in_offsets        = NULL;
        out_offsets       = NULL;
    }

    virtual ~PartitionerBase()
    {
        printf("~PartitionerBase begin\n");fflush(stdout);
        if (Status == 0) return;
        
        /*for (int i=0; i< num_gpus; i++)
        {
            delete[] partition_tables [i+1]; partition_tables [i+1] = NULL;
            delete[] convertion_tables[i+1]; convertion_tables[i+1] = NULL;
            delete[] in_offsets       [i  ]; in_offsets       [i  ] = NULL;
            delete[] out_offsets      [i  ]; out_offsets      [i  ] = NULL;
        }
        delete[] partition_tables [0]; partition_tables [0] = NULL;
        delete[] convertion_tables[0]; convertion_tables[0] = NULL;
        delete[] partition_tables    ; partition_tables     = NULL;
        delete[] convertion_tables   ; convertion_tables    = NULL;
        delete[] in_offsets          ; in_offsets           = NULL;
        delete[] out_offsets         ; out_offsets          = NULL;
        delete[] sub_graphs          ; sub_graphs           = NULL;*/
        Status   = 0;
        num_gpus = 0;
        printf("~PartitionerBase end\n");fflush(stdout);
    } 

    cudaError_t Init(
        const GraphT &graph,
        int   num_gpus)
    {   
        cudaError_t retval= cudaSuccess;
        this->num_gpus    = num_gpus;
        this->graph       = &graph;
        sub_graphs        = new GraphT   [num_gpus  ];
        partition_tables  = new int*     [num_gpus+1];
        convertion_tables = new VertexId*[num_gpus+1];
        in_offsets        = new SizeT*   [num_gpus  ];
        out_offsets       = new SizeT*   [num_gpus  ];
        
        for (int i=0;i<num_gpus+1;i++)
        {
            partition_tables [i] = NULL;
            convertion_tables[i] = NULL;
        }
        partition_tables [0] = new int     [graph.nodes];
        convertion_tables[0] = new VertexId[graph.nodes];
        memset(partition_tables [0], 0, sizeof(int     ) * graph.nodes);
        memset(convertion_tables[0], 0, sizeof(VertexId) * graph.nodes);
        for (int i=0;i<num_gpus;i++)
        {
            in_offsets [i] = new SizeT [num_gpus+1];
            out_offsets[i] = new SizeT [num_gpus+1];
            memset(in_offsets [i], 0, sizeof(SizeT) * (num_gpus+1));
            memset(out_offsets[i], 0, sizeof(SizeT) * (num_gpus+1)); 
        }
        Status = 1;

        return retval;
    }
    
    //template <bool LOAD_EDGE_VALUES, bool LOAD_NODE_VALUES>
    static CUT_THREADPROC MakeSubGraph_Thread(void *thread_data_)
    {
        ThreadSlice<VertexId,SizeT,Value> *thread_data = (ThreadSlice<VertexId,SizeT,Value> *) thread_data_;
        const GraphT* graph           = thread_data->graph;
        GraphT*     sub_graph         = thread_data->sub_graph;
        int         gpu               = thread_data->thread_num;
        CUTBarrier* cpu_barrier       = thread_data->cpu_barrier;
        int         num_gpus          = thread_data->num_gpus;
        int*        partition_table0  = thread_data->partition_table0;
        VertexId*   convertion_table0 = thread_data->convertion_table0;
        int**       partition_table1  = thread_data->partition_table1;
        VertexId**  convertion_table1 = thread_data->convertion_table1;
        SizeT**     out_offsets       = thread_data->out_offsets;
        SizeT**     in_offsets        = thread_data->in_offsets;
        SizeT       num_nodes         = 0, node_counter;
        SizeT       num_edges         = 0, edge_counter;
        int*        marker            = new int[graph->nodes];
        SizeT*      cross_counter     = new SizeT[num_gpus];
        VertexId*   tconvertion_table = new VertexId[graph->nodes];

        memset(marker, 0, sizeof(int)*graph->nodes);
        memset(cross_counter, 0, sizeof(SizeT) * num_gpus);

        for (SizeT node=0; node<graph->nodes; node++)
        if (partition_table0[node] == gpu)
        {
            convertion_table0[node] = cross_counter[gpu];
            tconvertion_table[node] = cross_counter[gpu];
            marker[node] =1;
            for (SizeT edge=graph->row_offsets[node]; edge<graph->row_offsets[node+1]; edge++)
            {
                SizeT neibor = graph->column_indices[edge];
                int peer  = partition_table0[neibor];
                if ((peer != gpu) && (marker[neibor] == 0))
                {
                    tconvertion_table[neibor]=cross_counter[peer];
                    cross_counter[peer]++;
                    marker[neibor]=1;
                    num_nodes++;
                }
            }
            cross_counter[gpu]++;
            num_nodes++;
            num_edges+= graph->row_offsets[node+1] - graph->row_offsets[node];
        }
        delete[] marker;marker=NULL;
        printf("%d: cross_counter = {%d, %d}\n", gpu, cross_counter[0], cross_counter[1]);
        out_offsets[gpu][0]=0;
        node_counter=cross_counter[gpu];
        for (int peer=0;peer<num_gpus;peer++)
        {
            if (peer==gpu) continue;
            int peer_=peer < gpu? peer+1 : peer;
            out_offsets[gpu][peer_]=node_counter;
            node_counter+=cross_counter[peer];
        }
        out_offsets[gpu][num_gpus]=node_counter;
        printf("%d: out_offsets = {%d,%d,%d}\n", gpu, out_offsets[gpu][0], out_offsets[gpu][1], out_offsets[gpu][2]); fflush(stdout);
        
        printf("%d: cpu_barrier wait\n", gpu); fflush(stdout);
        cutIncrementBarrier(cpu_barrier);
        cutWaitForBarrier  (cpu_barrier);
        printf("%d: cpu_barrier past\n", gpu); fflush(stdout);
        in_offsets[gpu][0]=0;
        node_counter=0;
        for (int peer=0;peer<num_gpus;peer++)
        {
            if (peer==gpu) continue;
            int peer_ = peer < gpu ? peer+1 : peer;
            int gpu_  = gpu  < peer? gpu +1 : gpu ; 
            in_offsets[gpu][peer_]=node_counter;
            node_counter+=out_offsets[peer][gpu_+1]-out_offsets[peer][gpu_];
        }
        in_offsets[gpu][num_gpus]=node_counter;
        printf("%d: in_offsets = {%d,%d,%d}\n", gpu, in_offsets[gpu][0], in_offsets[gpu][1], in_offsets[gpu][2]); fflush(stdout);
        
        if      (graph->node_values == NULL && graph->edge_values == NULL) 
             sub_graph->template FromScratch < false , false  >(num_nodes,num_edges);
        else if (graph->node_values != NULL && graph->edge_values == NULL) 
             sub_graph->template FromScratch < false , true   >(num_nodes,num_edges);
        else if (graph->node_values == NULL && graph->edge_values != NULL) 
             sub_graph->template FromScratch < true  , false  >(num_nodes,num_edges);
        else sub_graph->template FromScratch < true  , true   >(num_nodes,num_edges);

        if (convertion_table1[0] != NULL) free(convertion_table1[0]);
        if (partition_table1 [0] != NULL) free(partition_table1[0]);
        convertion_table1[0]= (VertexId*) malloc (sizeof(VertexId) * num_nodes);//new VertexId[num_nodes];
        partition_table1 [0]= (int*) malloc (sizeof(int) * num_nodes);//new int     [num_nodes];
        edge_counter=0;
        for (SizeT node=0; node<graph->nodes; node++)
        if (partition_table0[node] == gpu)
        {
            VertexId node_ = tconvertion_table[node];
            sub_graph->row_offsets[node_]=edge_counter;
            if (graph->node_values != NULL) sub_graph->node_values[node_]=graph->node_values[node];
            partition_table1 [0][node_] = 0;
            convertion_table1[0][node_] = node_;
            for (SizeT edge=graph->row_offsets[node]; edge<graph->row_offsets[node+1]; edge++)
            {
                SizeT    neibor  = graph->column_indices[edge];
                int      peer    = partition_table0[neibor];
                int      peer_   = peer < gpu ? peer+1 : peer;
                if (peer == gpu) peer_ = 0;
                VertexId neibor_ = tconvertion_table[neibor] + out_offsets[gpu][peer_];
                
                sub_graph->column_indices[edge_counter] = neibor_;
                if (graph->edge_values !=NULL) sub_graph->edge_values[edge_counter]=graph->edge_values[edge];
                if (peer != gpu)
                {
                    sub_graph->row_offsets[neibor_]=num_edges;
                    partition_table1 [0][neibor_] = peer_;
                    convertion_table1[0][neibor_] = convertion_table0[neibor];
                }
                edge_counter++;
            }   
        }
        sub_graph->row_offsets[num_nodes]=num_edges;

        delete[] cross_counter;     cross_counter     = NULL;
        delete[] tconvertion_table; tconvertion_table = NULL;
        CUT_THREADEND;
    }

    //template <bool LOAD_EDGE_VALUES,bool LOAD_NODE_VALUES>
    cudaError_t MakeSubGraph()
    {
        printf("MakeSubGraph begin.\n");fflush(stdout);
        cudaError_t retval = cudaSuccess;
        ThreadSlice<VertexId,SizeT,Value>* thread_data = new ThreadSlice<VertexId,SizeT,Value>[num_gpus];
        CUTThread*   thread_Ids  = new CUTThread  [num_gpus];
        CUTBarrier   cpu_barrier = cutCreateBarrier(num_gpus);

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            thread_data[gpu].graph             = graph;
            thread_data[gpu].sub_graph         = &(sub_graphs[gpu]);
            thread_data[gpu].thread_num        = gpu;
            thread_data[gpu].cpu_barrier       = &cpu_barrier;
            thread_data[gpu].num_gpus          = num_gpus;
            thread_data[gpu].partition_table0  = partition_tables [0];
            thread_data[gpu].convertion_table0 = convertion_tables[0];
            thread_data[gpu].partition_table1  = &(partition_tables[gpu+1]);
            thread_data[gpu].convertion_table1 = &(convertion_tables[gpu+1]);
            thread_data[gpu].in_offsets        = in_offsets;
            thread_data[gpu].out_offsets       = out_offsets;
            thread_data[gpu].thread_Id         = cutStartThread((CUT_THREADROUTINE)&(MakeSubGraph_Thread)//<LOAD_EDGE_VALUES,LOAD_NODE_VALUES>
                    , (void*)(&(thread_data[gpu])));
            thread_Ids[gpu]=thread_data[gpu].thread_Id;
        }

        cutWaitForThreads(thread_Ids,num_gpus);
        cutDestroyBarrier(&cpu_barrier);
        delete[] thread_Ids ;thread_Ids =NULL;
        delete[] thread_data;thread_data=NULL;
        Status = 2;
        printf("MakeSubGraph end.\n"); fflush(stdout);
        return retval;
    }

    /*cudaError_t MakeSubGraph_Old()
    {
        cudaError_t retval = cudaSuccess;

        SizeT *nodes         =new SizeT [num_gpus];
        SizeT *edges         =new SizeT [num_gpus];
        SizeT **cross_count  =new SizeT*[num_gpus];
        //_SizeT *convertion_table=new _SizeT[input_graph->nodes];
        //SizeT *tconvertion_table=new SizeT[graph->nodes];
        
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            nodes[gpu]=0;
            edges[gpu]=0;
            sub_graphs[gpu].Free();
            cross_count[gpu] = new SizeT[num_gpus];
            memset(cross_count[gpu],0,sizeof(SizeT) * num_gpus);
        }

        for (SizeT node=0;node<input_graph->nodes;node++)
        {
            int gpu = partition_tables[0][node];
            convertion_tables[0][node] = cross_count[gpu][gpu];
            cross_count[gpu][gpu]++;
            edges[gpu] += graph->row_offsets[node+1]-graph->row_offsets[node];
        }

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            _SizeT gpu_offset=gpu*(num_gpus+1);
            _SizeT tnode=0,tedge=0;
            //output_graphs[gpu].FromScratch<false,false>(nodes[gpu],edges[gpu]);
            memset(tconvertion_table,0,sizeof(_VertexId)*input_graph->nodes);
            for (_SizeT node=0;node<input_graph->nodes;node++)
            if (partition_table[node]==gpu)
            {
                for (_SizeT edge=input_graph->row_offsets[node];edge<input_graph->row_offsets[node+1];edge++)
                {
                    _SizeT neibor=input_graph->column_indices[edge];
                    int tgpu=partition_table[neibor];
                    if ((tgpu!=gpu)&&(tconvertion_table[neibor]==0))
                    {
                       foreign_count[gpu_offset+tgpu]++;
                       tconvertion_table[neibor]=1;
                    }
                }
            }
            memset(tconvertion_table,0,sizeof(_VertexId)*input_graph->nodes);
            foreign_offset[0]=nodes[gpu];
            for (int tgpu=0;tgpu<num_gpus;tgpu++)
            {
                foreign_offset[tgpu+1]=foreign_offset[tgpu]+foreign_count[gpu_offset+tgpu];
                if (foreign_nodes_count[tgpu]!=0)
                {
                    _SizeT *tforeign_nodes=new _SizeT[foreign_nodes_count[tgpu]+foreign_count[gpu_offset+tgpu]];
                    memcpy(tforeign_nodes,foreign_nodes[tgpu],sizeof(_VertexId)*foreign_nodes_count[tgpu]);
                    memset(&tforeign_nodes[foreign_nodes_count[tgpu]],0,sizeof(_VertexId)*foreign_count[gpu_offset+tgpu]);
                    delete[] foreign_nodes[tgpu];
                    foreign_nodes[tgpu]=tforeign_nodes;
                } else {
                    foreign_nodes[tgpu]=new _SizeT[foreign_count[gpu_offset+tgpu]];
                    memset(foreign_nodes[tgpu],0,sizeof(_VertexId)*foreign_count[gpu_offset+tgpu]);
                }
                foreign_count[gpu_offset+tgpu]=0;
            }
            //output_graphs[gpu].FromScratch<false,false>(foreign_offset[num_gpus],edges[gpu]);
            output_graphs[gpu].nodes=foreign_offset[num_gpus];
            output_graphs[gpu].edges=edges[gpu];
            output_graphs[gpu].row_offsets   = (_SizeT*) malloc(sizeof(_SizeT) * (output_graphs[gpu].nodes+1));
            output_graphs[gpu].column_indices= (_VertexId*) malloc(sizeof(_VertexId) * (output_graphs[gpu].edges));
            output_graphs[gpu].node_values   = (_Value*) malloc(sizeof(_Value) * (output_graphs[gpu].nodes));
            output_graphs[gpu].edge_values   = (_Value*) malloc(sizeof(_Value) * (output_graphs[gpu].edges));
 
            for (_VertexId node=0;node<(input_graph->nodes);node++)
            if (partition_table[node]==gpu)
            {
                tnode++;
                output_graphs[gpu].node_values[tnode]=input_graph->node_values[node];
                output_graphs[gpu].row_offsets[tnode]=tedge;
                for (_VertexId edge=input_graph->row_offsets[node];edge<input_graph->row_offsets[node+1];edge++)
                {
                    _SizeT tneibor,neibor=input_graph->column_indices[edge];
                    int tgpu=partition_table[neibor];
                    if (tgpu==gpu) tneibor=convertion_table[neibor];
                    else {
                        if (tconvertion_table[neibor]==0)
                        {
                            foreign_nodes[tgpu][foreign_nodes_count[tgpu]+foreign_count[gpu_offset+tgpu]]=convertion_table[neibor];
                            tconvertion_table[neibor]=foreign_offset[tgpu]+foreign_count[gpu_offset+tgpu];
                            foreign_count[gpu_offset+tgpu]++;
                            tneibor=tconvertion_table[neibor];
                            output_graphs[gpu].node_values[tneibor]=input_graph->node_values[neibor];
                            output_graphs[gpu].row_offsets[tneibor]=edges[gpu];
                        } else tneibor=tconvertion_table[neibor];
                    }
                    output_graphs[gpu].edge_values[tedge]=input_graph->edge_values[edge];
                    output_graphs[gpu].column_indices[tedge]=tneibor;
                    tedge++;
                }
            }
            output_graphs[gpu].row_offsets[foreign_offset[num_gpus]]=edges[gpu];
            for (int tgpu=0;tgpu<num_gpus;tgpu++)
                foreign_nodes_count[tgpu]+=foreign_count[gpu_offset+tgpu];
        }
        delete[] nodes;nodes=NULL;
        delete[] edges;edges=NULL;
        delete[] foreign_offset;foreign_offset=NULL;
        //delete[] convertion_table;convertion_table=NULL;
        delete[] tconvertion_table;tconvertion_table=NULL;
        return Status;
    }*/

    //template <bool LOAD_EDGE_VALUES, bool LOAD_NODE_VALUES>
    virtual cudaError_t Partition(
        GraphT*    &sub_graphs,
        int**      &partition_tables,
        VertexId** &convertion_tables,
        SizeT**    &in_offsets,
        SizeT**    &out_offsets)
    {
        printf("PartitionBase:Partition called.\n"); fflush(stdout);
        return util::GRError("PartitionBase::Partition is undefined", __FILE__, __LINE__);
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
