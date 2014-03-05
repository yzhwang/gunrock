// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/scan/multi_scan.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>


namespace gunrock {
namespace app {
namespace bfs {

    template <bool INSTRUMENT> class BFSEnactor;

    //template<
    //    bool     INSTRUMENT,
    //    typename BFSProblem>
    class ThreadSlice
    {
    public:
        int           thread_num;
        int           init_size;
        int           max_grid_size;
        int           edge_map_grid_size;
        int           vertex_map_grid_size;
        CUTThread     thread_Id;
        util::cpu_mt::CPUBarrier* cpu_barrier;
        //BFSProblem*   problem;
        //BFSEnactor<INSTRUMENT>*   enactor;
        void*         problem;
        void*         enactor;

        ThreadSlice()
        {
            printf("ThreadSlice() called.\n"); fflush(stdout);
            cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
        }

        virtual ~ThreadSlice()
        {
            printf("~ThreadSlice() called. \n"); fflush(stdout);
            cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
        }
    };
 
    template <typename VertexId, typename SizeT, bool MARK_PREDECESSORS>
    __global__ void Expand_Incoming (
        const SizeT            num_elements,
        const SizeT            num_associates,
        const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              VertexId*        keys_out,
              VertexId**       associate_in,
              VertexId**       associate_out)
    {  
        SizeT x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x>=num_elements) return;

        VertexId key=keys_in[x];
        keys_out[x]=key;
        if (num_associates <1) return;
        VertexId t=associate_in[0][incoming_offset+x];
        if (t >= associate_out[0][key] && associate_out[0][key]>=0) return;
        associate_out[0][key]=t;
        for (SizeT i=1;i<num_associates;i++)
        {   
            associate_out[i][key]=associate_in[i][incoming_offset+x];   
        }   
    }   

    bool All_Done(volatile int **dones, cudaError_t *retvals,int num_gpus)
    {
        for (int gpu=0;gpu<num_gpus;gpu++)
          if (retvals[gpu]!=cudaSuccess) {printf("Err: gpu=%d\n", gpu); fflush(stdout); return true;}
        for (int gpu=0;gpu<num_gpus;gpu++)
          if (dones[gpu][0]!=0) {printf("not finish: gpu=%d\n", gpu); fflush(stdout); return false;}
        return true;
    }

   /*template<
        bool     INSTRUMENT,
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename BFSProblem>
   static CUT_THREADPROC BFSThread_new(
        void * thread_data_)
    {
        //printf("BFSThread begin.\n"); fflush(stdout);

        ThreadSlice //<INSTRUMENT, BFSProblem> *
            * thread_data = (ThreadSlice *) thread_data_;
        int thread_num = thread_data->thread_num;
        BFSProblem* problem = thread_data->problem;
        int gpu = problem->gpu_idx[thread_num];

        cudaSetDevice(gpu);
        while (true) {int i=i+1;}
        CUT_THREADEND;
    }*/
 
   template<
        bool     INSTRUMENT,
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename BFSProblem>
   static CUT_THREADPROC BFSThread(
        void * thread_data_)
    {
        //printf("BFSThread begin.\n"); fflush(stdout);

        ThreadSlice //<INSTRUMENT, BFSProblem> *
            * thread_data = (ThreadSlice *) thread_data_;
        typedef typename BFSProblem::SizeT      SizeT;
        typedef typename BFSProblem::VertexId   VertexId;

        typedef BFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            BFSProblem> BfsFunctor;
        
        SizeT*       out_offset;
        char*         message               = new char [1024];
        BFSProblem*  problem               =   (BFSProblem*) thread_data->problem;
        BFSEnactor<INSTRUMENT>*  enactor   =   (BFSEnactor<INSTRUMENT>*)thread_data->enactor;
        int          thread_num            =   thread_data->thread_num;
        util::cpu_mt::CPUBarrier* cpu_barrier =   thread_data->cpu_barrier;
        int          gpu                   =   problem    ->gpu_idx        [thread_num];
        //int*         gpu_idx               =   problem    ->gpu_idx;
        int          num_gpus              =   problem    ->num_gpus;
        //int          max_grid_size         =   thread_data->max_grid_size;
        int          edge_map_grid_size    =   thread_data->edge_map_grid_size;
        int          vertex_map_grid_size  =   thread_data->vertex_map_grid_size;
        cudaError_t* retval                = &(enactor    ->retvals        [thread_num]);
        //volatile int* done               =  (enactor    ->dones          [thread_num]);
        volatile int* d_done               =  (enactor    ->d_dones        [thread_num]);
        int*         iteration             = &(enactor    ->iterations     [thread_num]);
        cudaEvent_t* throttle_event        = &(enactor    ->throttle_events[thread_num]);
        volatile int** dones               =   enactor    ->dones;
        cudaError_t* retvals               =   enactor    ->retvals;
        bool         DEBUG                 =   enactor    ->DEBUG;
        unsigned long long* total_queued   = &(enactor    ->total_queued   [thread_num]);
        unsigned long long* total_runtimes = &(enactor    ->total_runtimes [thread_num]);
        unsigned long long* total_lifetimes= &(enactor    ->total_lifetimes[thread_num]);
        //SizeT        num_elements          =   problem    ->sub_graphs     [thread_num].nodes; //?
        typename BFSProblem::GraphSlice*
                     graph_slice           =   problem    ->graph_slices   [thread_num];
        typename BFSProblem::DataSlice*
                     data_slice            =   problem    ->data_slices    [thread_num];
        typename BFSProblem::DataSlice*
                     d_data_slice          =   problem    ->d_data_slices  [thread_num];
        util::CtaWorkProgressLifetime*
                     work_progress         = &(enactor    ->work_progress  [thread_num]);
        util::KernelRuntimeStatsLifetime*
                       edge_map_kernel_stats = &(enactor->  edge_map_kernel_stats[thread_num]);
        util::KernelRuntimeStatsLifetime*
                     vertex_map_kernel_stats = &(enactor->vertex_map_kernel_stats[thread_num]);
        util::scan::MultiScan<VertexId,SizeT,true,256,8> *Scaner = NULL;
        //char message[512]="";

        //printf("%d: Assigment finished.\n", gpu); fflush(stdout);
        //cudaError_t retval = cudaSuccess;

        if (num_gpus >1) 
        {
            Scaner=new util::scan::MultiScan<VertexId,SizeT,true,256,8>;
            //printf("%d: Scanner created.\n", gpu); fflush(stdout);
            out_offset=new SizeT[num_gpus+1];
        }
        do {  
            //char message[]="BFSThread cudaSetDevice failed.";
            //message = "BFSThread cudaSetDevice failed.";
            //cudaSetDevice(gpu);
            //while (true) {int i=gpu+1;}
            //if (retval [0]) { retval[0] = util::GRError(retval [0], message, __FILE__, __LINE__); break; }
            if ( retval[0] = util::GRError(cudaSetDevice(gpu), "BFSThread cudaSetDevice failed." , __FILE__, __LINE__)) break; 
            //printf("%d: Gpu set. Init_size = %d \n", gpu, thread_data->init_size); fflush(stdout); 
            //SizeT    queue_length   = thread_data->init_size;
            VertexId queue_index    = 0;        // Work queue index
            int      selector       = 0;
            SizeT    num_elements   = thread_data->init_size; //?
            bool     queue_reset    = true; 
            //fflush(stdout);
            // Step through BFS iterations
                        
            VertexId *h_cur_queue = new VertexId[graph_slice->edges];
            while (!All_Done(dones,retvals,num_gpus)) {
                util::cpu_mt::PrintMessage("iteration begin.", gpu, iteration[0]);
                if (DEBUG) {
                    SizeT _queue_length;
                    if (queue_reset) _queue_length = num_elements;
                    else if (retval[0] = work_progress->GetQueueLength(queue_index, _queue_length)) break;
                    sprintf(message,"queue_length before edge_map_forward = %lld.", (long long) _queue_length);
                    util::cpu_mt::PrintMessage(message, gpu, iteration[0]);
                    util::cpu_mt::PrintGPUArray("0d_keys     ",graph_slice->frontier_queues.d_keys[selector], _queue_length, gpu, iteration[0]);
                    util::cpu_mt::PrintGPUArray("0d_labels   ",data_slice->d_labels,graph_slice->nodes, gpu,iteration[0]);
                    if (data_slice->d_preds!=NULL) util::cpu_mt::PrintGPUArray("0d_preds    ", data_slice->d_preds, graph_slice->nodes, gpu, iteration[0]);
                    if (retval[0] = work_progress->SetQueueLength(queue_index+1, 0)) break;
                    if (retval[0] = work_progress->GetQueueLength(queue_index+1, _queue_length)) break;
                    sprintf(message,"queue_length1 before edge_map_forward = %lld.", (long long) _queue_length);
                    util::cpu_mt::PrintMessage(message, gpu, iteration[0]);
                }

                // Edge Map
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, BFSProblem, BfsFunctor>
                    <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    iteration[0],
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    graph_slice->frontier_queues.d_values[selector^1],          // d_pred_out_queue
                    graph_slice->frontier_queues.d_keys[selector^1],            // d_out_queue
                    graph_slice->d_column_indices,
                    d_data_slice,
                    work_progress[0],
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    edge_map_kernel_stats[0]);
               
                if (DEBUG && (retval[0] = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event[0]);                                 // give host memory mapped visibility to GPU updates 

                // Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;
               
                queue_index++;
                selector ^= 1;
                if (DEBUG) {
                    SizeT _queue_length;
                    if (retval[0] = work_progress->GetQueueLength(queue_index, _queue_length)) break;
                    sprintf(message,"queue_length after edge_map_forward = %lld.", (long long) _queue_length);
                    util::cpu_mt::PrintMessage(message, gpu, iteration[0]);
                    util::cpu_mt::PrintGPUArray("1d_keys     ",graph_slice->frontier_queues.d_keys[selector], _queue_length, gpu, iteration[0]);
                    util::cpu_mt::PrintGPUArray("1d_labels   ",data_slice->d_labels,graph_slice->nodes, gpu, iteration[0]);
                    if (data_slice->d_preds!=NULL) util::cpu_mt::PrintGPUArray("1d_preds    ",data_slice->d_preds, graph_slice->nodes, gpu, iteration[0]);
                    if (retval[0] = work_progress->SetQueueLength(queue_index+1, 0)) break;
                    /*if (iteration == 2) {
                        cudaMemcpy(h_cur_queue, graph_slice->frontier_queues.d_keys[selector], sizeof(VertexId)*queue_length, cudaMemcpyDeviceToHost);
                        int neg_num = 0;
                        std::sort(h_cur_queue, h_cur_queue + queue_length);
                        for (int i = 0; i < queue_length; ++i)
                        {
                            if (h_cur_queue[i] == -1)
                                neg_num++;
                        }
                        printf("(%d)", neg_num);
                    }*/
                }

                if (INSTRUMENT) {
                    if (retval[0] = edge_map_kernel_stats->Accumulate(
                        edge_map_grid_size,
                        total_runtimes[0],
                        total_lifetimes[0])) break;
                }

                // Throttle
                if (iteration[0] & 1) {
                    if (retval[0] = util::GRError(cudaEventRecord(throttle_event[0]),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval[0] = util::GRError(cudaEventSynchronize(throttle_event[0]),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                //if (done[0] == 0) break;
                if (All_Done(dones,retvals,num_gpus)) break;

                // Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, BFSProblem, BfsFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    iteration[0]+1,
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                    graph_slice->frontier_queues.d_values[selector],    // d_pred_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                    d_data_slice,
                    data_slice->d_visited_mask,
                    work_progress[0],
                    graph_slice->frontier_elements[selector],           // max_in_queue
                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                    vertex_map_kernel_stats[0]);

                if (DEBUG && (retval[0] = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event[0]); // give host memory mapped visibility to GPU updates

                queue_index++;
                selector ^= 1;

                if (INSTRUMENT || DEBUG) {
                    SizeT _queue_length;
                    if (retval[0] = work_progress->GetQueueLength(queue_index, _queue_length)) break;
                    total_queued[0] += _queue_length;
                    if (DEBUG) 
                    {
                        sprintf(message,"queue_length after vertex_map = %lld.", (long long) _queue_length);
                        util::cpu_mt::PrintMessage(message, gpu, iteration[0]);
                        util::cpu_mt::PrintGPUArray("2d_keys     ",graph_slice->frontier_queues.d_keys[selector], _queue_length, gpu, iteration[0]);
                        util::cpu_mt::PrintGPUArray("2d_labels   ",data_slice->d_labels,graph_slice->nodes, gpu, iteration[0]);
                        if (data_slice->d_preds!=NULL) util::cpu_mt::PrintGPUArray("2d_preds    ",data_slice->d_preds, graph_slice->nodes, gpu, iteration[0]);
                    }
                    if (INSTRUMENT) {
                        if (retval[0] = vertex_map_kernel_stats->Accumulate(
                            vertex_map_grid_size,
                            total_runtimes[0],
                            total_lifetimes[0])) break;
                    }
                }
                // Check if done
                //if (done[0] == 0) break;
                if (All_Done(dones,retvals,num_gpus)) break;

                //if (DEBUG) printf("%d: iteration = %lld num_elements= %lld\n", gpu, (long long) iteration[0], (long long) num_elements); fflush(stdout);

                //Use multi_scan to splict the workload into multi_gpus
                if (num_gpus >1)
                {
                    // Split the frontier into multiple frontiers for multiple GPUs, local remains infront
                    SizeT n;
                    if (retval[0] = work_progress->GetQueueLength(queue_index, n)) break;
                    if (n >0)
                    {
                        sprintf(message,"n = %d frontier_elements = %d,%d", n, graph_slice->frontier_elements[selector],graph_slice->frontier_elements[selector^1]);
                        util::cpu_mt::PrintMessage(message,gpu,iteration[0]);
                        util::cpu_mt::PrintGPUArray<SizeT,VertexId>("3d_keys     ",graph_slice->frontier_queues.d_keys[selector],n, gpu,iteration[0]);
                        util::cpu_mt::PrintGPUArray<SizeT,VertexId>("3d_labels   ",data_slice->d_labels,graph_slice->nodes, gpu, iteration[0]);
                        if (data_slice->d_preds!=NULL) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("3d_preds    ",data_slice->d_preds, graph_slice->nodes, gpu, iteration[0]);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("3d_associate",data_slice->h_associate_org[0],graph_slice->out_offset[num_gpus],gpu,iteration[0]);
                        util::cpu_mt::PrintGPUArray<SizeT,int     >("3d_split    ",graph_slice->d_partition_table, graph_slice->out_offset[num_gpus],gpu,iteration[0]);
                        //util::cpu_mt::PrintGPUArray<SizeT,int     >("d_ooff",gpu,graph_slice->d_out_offset, num_gpus);
                        Scaner->Scan_with_Keys(n,
                                  num_gpus,
                                  data_slice->num_associate,
                                  graph_slice->frontier_queues.d_keys[selector],
                                  graph_slice->frontier_queues.d_keys[selector^1],
                                  graph_slice->d_partition_table,
                                  graph_slice->d_convertion_table,
                                  //graph_slice->d_out_offset,
                                  data_slice ->d_out_length,
                                  data_slice ->d_associate_org,
                                  data_slice ->d_associate_out);
                        if (retval[0] = util::GRError(cudaMemcpy(
                                  data_slice->out_length,
                                  data_slice->d_out_length,
                                  sizeof(SizeT)*num_gpus,
                                  cudaMemcpyDeviceToHost),
                                  "BFSEnactor cudaMemcpy h_Length failed", __FILE__, __LINE__)) break;
                        out_offset[0]=0;
                        for (int i=0;i<num_gpus;i++) out_offset[i+1]=out_offset[i]+data_slice->out_length[i];
 
                        queue_index++;
                        selector ^= 1;
                        util::cpu_mt::PrintMessage("multi_scan returned.", gpu, iteration[0]);
                        util::cpu_mt::PrintGPUArray<SizeT,VertexId>("4d_keys     ",graph_slice->frontier_queues.d_keys[selector], n, gpu, iteration[0]);
                        util::cpu_mt::PrintGPUArray<SizeT,VertexId>("4d_associate",data_slice->h_associate_out[0], out_offset[num_gpus]-out_offset[1],gpu,iteration[0]);
                        if (data_slice->d_preds!=NULL) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("4d_associate",data_slice->h_associate_out[1], out_offset[num_gpus]-out_offset[1],gpu,iteration[0]);
                        util::cpu_mt::PrintCPUArray<SizeT,VertexId>("4out_length ",data_slice->out_length, num_gpus, gpu,iteration[0]);
                        util::cpu_mt::PrintCPUArray<SizeT,VertexId>("4out_offset ",out_offset, num_gpus+1, gpu, iteration[0]);
                        
                        if (iteration[0]!=0)
                        {  //CPU global barrier
                            sprintf(message,"cpu_barrier1 wait releaseCount=%d count=%d reseted=%d waken=%d", cpu_barrier[1].releaseCount, cpu_barrier[1].count, cpu_barrier[1].reseted? 0:1, cpu_barrier[1].waken);
                            util::cpu_mt::PrintMessage(message,gpu,iteration[0]);
                            util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[1]),gpu);
                            //if (!util::cpu_mt::IncrementBarriernReset(cpu_barrier))
                            //{
                            //    cutWaitForBarrier  (cpu_barrier);
                            //}
                            sprintf(message,"cpu_barrier1 past releaseCount=%d count=%d reseted=%d waken=%d", gpu, cpu_barrier[1].releaseCount, cpu_barrier[1].count, cpu_barrier[1].reseted? 0:1, cpu_barrier[1].waken);
                            util::cpu_mt::PrintMessage(message,gpu,iteration[0]);
                        }

                        //Move data
                        for (int peer=0;peer<num_gpus;peer++)
                        {
                            int peer_ = peer<gpu? peer+1: peer;
                            int gpu_  = peer<gpu? gpu   : gpu+1;
                            if (peer==gpu) continue;
                            problem->data_slices[peer]->in_length[gpu_]=data_slice->out_length[peer_];
                            if (data_slice->out_length[peer_] == 0) continue;
                            //printf("%d: peer = %d out_length = %d in_offset = %d, out_offset = %d\n", gpu, peer, data_slice->out_length[peer_], problem->graph_slices[peer]->in_offset[gpu_], out_offset[peer_]);
                            if (retval [0] = util::GRError(cudaMemcpy(
                                  problem->data_slices[peer]->d_keys_in + problem->graph_slices[peer]->in_offset[gpu_],
                                  //gpu_idx[peer], 
                                  graph_slice->frontier_queues.d_keys[selector] + out_offset[peer_],
                                  //gpu_idx[gpu],
                                  sizeof(VertexId) * data_slice->out_length[peer_], cudaMemcpyDefault),
                                  "cudaMemcpyPeer d_keys failed", __FILE__, __LINE__)) break;
                            //printf("%d: d_keys_out -> d_keys_in finished.\n", gpu);fflush(stdout);
                            for (int i=0;i<data_slice->num_associate;i++)
                            {
                                //printf("%d: peer = %d associate = %d in_offset = %d out_offset = %d length = %d\n", gpu, peer, i, problem->graph_slices[peer]->in_offset[gpu_], out_offset[peer_] - out_offset[1], data_slice->out_length[peer_]); fflush(stdout);
                                if (retval [0] = util::GRError(cudaMemcpy(
                                    problem->data_slices[peer]->h_associate_in[i] + problem->graph_slices[peer]->in_offset[gpu_],
                                    //gpu_idx[peer],
                                    data_slice->h_associate_out[i] + (out_offset[peer_] - out_offset[1]),
                                    //gpu_idx[gpu],
                                    sizeof(VertexId) * data_slice->out_length[peer_], cudaMemcpyDefault),
                                    "cudaMemcpyPeer associate_out failed", __FILE__, __LINE__)) break;
                            }
                            if (retval [0]) break;
                        }
                        if (retval [0]) break;
                        util::cpu_mt::PrintMessage("data movement finished.",gpu,iteration[0]);
                    }  else {
                        util::cpu_mt::PrintMessage("multi_scan and data movement skipped.",gpu,iteration[0]);
                        if (iteration[0]!=0)
                        {  //CPU global barrier
                            sprintf(message,"cpu_barrier1 wait releaseCount=%d count=%d reseted=%d waken=%d", cpu_barrier[1].releaseCount, cpu_barrier[1].count, cpu_barrier[1].reseted? 0:1, cpu_barrier[1].waken);
                            util::cpu_mt::PrintMessage(message,gpu,iteration[0]);
                            util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[1]),gpu);
                            //if (!util::cpu_mt::IncrementBarriernReset(cpu_barrier))
                            //{
                            //    cutWaitForBarrier  (cpu_barrier);
                            //}
                            sprintf(message,"cpu_barrier1 past releaseCount=%d count=%d reseted=%d waken=%d", cpu_barrier[1].releaseCount, cpu_barrier[1].count, cpu_barrier[1].reseted? 0:1, cpu_barrier[1].waken);
                            util::cpu_mt::PrintMessage(message,gpu,iteration[0]);
                        }

                        for (int peer=0;peer<num_gpus;peer++)
                        {
                            int gpu_ = peer<gpu? gpu : gpu+1;
                            problem->data_slices[peer]->in_length[gpu_]=0;
                        }
                    }

                    //CPU global barrier
                    sprintf(message,"cpu_barrier0 wait releaseCount=%d count=%d reseted=%d waken=%d", cpu_barrier[0].releaseCount, cpu_barrier[0].count, cpu_barrier[0].reseted? 0:1, cpu_barrier[0].waken);
                    util::cpu_mt::PrintMessage(message,gpu,iteration[0]);
                    util::cpu_mt::IncrementnWaitBarrier(cpu_barrier,gpu);
                    //if (!util::cpu_mt::IncrementBarriernReset(cpu_barrier))
                    //{
                    //    cutWaitForBarrier  (cpu_barrier);
                    //}
                    sprintf(message,"cpu_barrier0 past releaseCount=%d count=%d reseted=%d waken=%d", cpu_barrier[0].releaseCount, cpu_barrier[0].count, cpu_barrier[0].reseted? 0:1, cpu_barrier[0].waken);
                    util::cpu_mt::PrintMessage(message,gpu,iteration[0]);
                    //printf("%d: cpu_barrier past.\n", gpu); fflush(stdout);
                    /*printf("%d: wai_barrier releaseCount=%d count=%d\n", gpu, cpu_barrier->releaseCount, cpu_barrier->count);fflush(stdout);
                    if (thread_num ==0)
                    {
                        cutDestroyBarrier(cpu_barrier);
                        cpu_barrier[0]=cutCreateBarrier(num_gpus);
                    }
                    printf("%d: set_barrier releaseCount=%d count=%d\n", gpu, cpu_barrier->releaseCount, cpu_barrier->count);fflush(stdout);
                    */
                    //Expand received data
                    util::cpu_mt::PrintCPUArray<SizeT,SizeT   >("5in_length  ", data_slice->in_length, num_gpus,gpu,iteration[0]);
                    util::cpu_mt::PrintCPUArray<SizeT,SizeT   >("5in_offset  ", graph_slice->in_offset, num_gpus, gpu,iteration[0]);
                    util::cpu_mt::PrintGPUArray<SizeT,VertexId>("5d_keys_in  ", data_slice->d_keys_in, data_slice->in_length[1],gpu,iteration[0]);
                    util::cpu_mt::PrintGPUArray<SizeT,VertexId>("5d_asso_in0 ", data_slice->h_associate_in[0], data_slice->in_length[1],gpu,iteration[0]);
                    if (data_slice->d_preds!=NULL) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("5d_asso_in1 ", data_slice->h_associate_in[1], data_slice->in_length[1],gpu,iteration[0]);
                    SizeT total_length=data_slice->out_length[0];
                    for (int peer=0;peer<num_gpus;peer++)
                    {
                        if (peer==gpu) continue;
                        int peer_ = peer<gpu ? peer+1: peer ;
                        //int gpu_  = peer<gpu ? gpu   : gpu+1;
                        if (data_slice->in_length[peer_] ==0) continue;
                        int grid_size = data_slice->in_length[peer_] / 256;
                        if ((data_slice->in_length[peer_] % 256)!=0) grid_size++;
                        Expand_Incoming <VertexId, SizeT, BFSProblem::MARK_PREDECESSORS>
                            <<<grid_size,256>>> (
                            data_slice  ->in_length[peer_],
                            data_slice  ->num_associate,
                            graph_slice ->in_offset[peer_],
                            data_slice  ->d_keys_in,
                            graph_slice->frontier_queues.d_keys[selector]+data_slice->out_length[0],
                            data_slice  ->d_associate_in,
                            data_slice  ->d_associate_org);
                        if (retval[0] = util::GRError("Expand_Incoming failed", __FILE__, __LINE__)) break;
                        //if (retval[0] = util::GRError(cudaMemcpy(
                        //            &(data_slice->frontier_queues.d_keys[selector][data_slice->h_Length[0]]),
                        //            &(data_slice->d_keys_in[graph_slice->in_offset[tgpu_foreign]]),
                        //            sizeof() * data_slice->in_length[tgpu_foreign], cudaMemcpyDeviceToDevice),
                        //            "cudaMemcpy d_keys_in failed", __FILE__, __LINE__)) break;
                        total_length+=data_slice->in_length[peer_];
                    }
                    if (retval [0]) break;
                    if (retval[0] = work_progress->SetQueueLength(queue_index,total_length)) break;
                }
                util::cpu_mt::PrintMessage("iteration finish.", gpu, iteration[0]);
                iteration[0]++;
            }

            delete[] h_cur_queue;h_cur_queue=NULL;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (retval[0] = work_progress->CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval[0] = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }
        } while(0);

        if (num_gpus >1)
        {
            delete Scaner; Scaner=NULL;
            delete[] out_offset; out_offset=NULL;
        }
        delete[] message;message=NULL;
        printf("BFSThread end.\n"); fflush(stdout);
        CUT_THREADEND; 
    }

/**
 * @brief BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class BFSEnactor : public EnactorBase
{
    // Members
public:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime *edge_map_kernel_stats;
    util::KernelRuntimeStatsLifetime *vertex_map_kernel_stats;
    util::CtaWorkProgressLifetime    *work_progress;

    unsigned long long *total_runtimes;              // Total working time by each CTA
    unsigned long long *total_lifetimes;             // Total life time of each CTA
    unsigned long long *total_queued;

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        **dones;
    int                 **d_dones;
    cudaEvent_t         *throttle_events;
    cudaError_t         *retvals;
    int                 num_gpus;
    int                 *gpu_idx;

    /**
     * Current iteration, also used to get the final search depth of the BFS search
     */
    int                 *iterations;
    
   // Methods
public:
    /**
     * @brief Prepare the enactor for BFS kernel call. Must be called prior to each BFS search.
     *
     * @param[in] problem BFS Problem object which holds the graph data and BFS problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] vertex_map_grid_size CTA occupancy for vertex mapping kernel call.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
        int edge_map_grid_size,
        int vertex_map_grid_size)
    {
        printf("BFSEnactor Setup begin.\n"); fflush(stdout);
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;
        
        cudaError_t retval = cudaSuccess;
        this->num_gpus = problem->num_gpus;
        this->gpu_idx  = problem->gpu_idx;
        printf("-1: num_gpus = %d, this->num_gpus = %d\n", num_gpus, this->num_gpus);fflush(stdout);
        do {
            dones           = new volatile int*      [num_gpus];
            d_dones         = new          int*      [num_gpus];
            throttle_events = new cudaEvent_t        [num_gpus];
            retvals         = new cudaError_t        [num_gpus];
            total_runtimes  = new unsigned long long [num_gpus];
            total_lifetimes = new unsigned long long [num_gpus];
            total_queued    = new unsigned long long [num_gpus];
            iterations      = new          int       [num_gpus];
            edge_map_kernel_stats   = new util::KernelRuntimeStatsLifetime[num_gpus];
            vertex_map_kernel_stats = new util::KernelRuntimeStatsLifetime[num_gpus];
            work_progress           = new util::CtaWorkProgressLifetime   [num_gpus];

            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                //initialize the host-mapped "done"
                //if (!dones[gpu]) {
                    //if (num_gpus != 1) 
                        if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]), "BFSEnactor cudaSetDevice gpu failed", __FILE__, __LINE__)) break;
                    int flags = cudaHostAllocMapped;
                    work_progress[gpu].Setup();

                    // Allocate pinned memory for done
                    if (retval = util::GRError(cudaHostAlloc((void**)&(dones[gpu]), sizeof(int) * 1, flags),
                        "BFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                    // Map done into GPU space
                    if (retval = util::GRError(cudaHostGetDevicePointer((void**)&(d_dones[gpu]), (void*) dones[gpu], 0),
                        "BFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                    // Create throttle event
                    if (retval = util::GRError(cudaEventCreateWithFlags(&(throttle_events[gpu]), cudaEventDisableTiming),
                        "BFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
                //} // if !done

                //initialize runtime stats
                if (retval =   edge_map_kernel_stats[gpu].Setup(  edge_map_grid_size)) break;
                if (retval = vertex_map_kernel_stats[gpu].Setup(vertex_map_grid_size)) break;

                //Reset statistics
                iterations      [gpu]    = 0;
                total_runtimes  [gpu]    = 0;
                total_lifetimes [gpu]    = 0;
                total_queued    [gpu]    = 0;
                dones           [gpu][0] = -1;

                //graph slice
                //typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[gpu];

                // Bind row-offsets and column_indices texture
                cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    problem->graph_slices[gpu]->d_row_offsets,
                    row_offsets_desc,
                    (problem->graph_slices[gpu]->nodes + 1) * sizeof(SizeT)),
                        "BFSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            column_indices_desc,
                            graph_slice->edges * sizeof(VertexId)),
                        "BFSEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
            } // for gpu
            if (retval) break;
        } while (0);
        printf("BFSEnactor Setup end. \n"); fflush(stdout);
        return retval;
    }

    public:

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG)//,
        //iteration(0),
        //total_queued(0),
        //done(NULL),
        //d_done(NULL)
    {
        edge_map_kernel_stats   = NULL;
        vertex_map_kernel_stats = NULL;
        work_progress           = NULL;
        total_runtimes          = NULL;
        total_lifetimes         = NULL;
        total_queued            = NULL;
        dones                   = NULL;
        d_dones                 = NULL;
        throttle_events         = NULL;
        retvals                 = NULL;
        iterations              = NULL;
        gpu_idx                 = NULL;
        num_gpus                = 0;
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        printf("~BFSEnactor begin.\n"); fflush(stdout);
        if (All_Done(dones,retvals,num_gpus)) {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                if (num_gpus !=1)
                    util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "BFSEnactor cudaSetDevice gpu failed", __FILE__, __LINE__);

                util::GRError(cudaFreeHost((void*)(dones[gpu])),
                    "BFSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

                util::GRError(cudaEventDestroy(throttle_events[gpu]),
                    "BFSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
            }
            printf("dones & throttle_events on gpu freed\n"); fflush(stdout);
            delete[] dones;          dones           = NULL; printf("dones deleted.\n"          ); fflush(stdout);
            delete[] throttle_events;throttle_events = NULL; printf("throttle_events deleted.\n"); fflush(stdout);
            delete[] retvals;        retvals         = NULL; printf("retvals deleted.\n"        ); fflush(stdout);
            delete[] iterations;     iterations      = NULL; printf("iterations deleted.\n"     ); fflush(stdout);
            delete[] total_runtimes; total_runtimes  = NULL; printf("total_runtimes deleted.\n" ); fflush(stdout);
            delete[] total_lifetimes;total_lifetimes = NULL; printf("total_lifetimes deleted.\n"); fflush(stdout);
            delete[] total_queued;   total_queued    = NULL; printf("total_queued deleted.\n"   ); fflush(stdout);
            delete[] work_progress;  work_progress   = NULL; printf("work_progress deleted.\n"  ); fflush(stdout);
            delete[]   edge_map_kernel_stats;  edge_map_kernel_stats = NULL; printf("edge_map_kernel_stats deleted.\n"); fflush(stdout);
            delete[] vertex_map_kernel_stats;vertex_map_kernel_stats = NULL; printf("vertex_map_kernel_stats deleted.\n"); fflush(stdout);
            gpu_idx = NULL;
        }
        printf("~BFSEnactor end.\n"); fflush(stdout);
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] search_depth Search depth of BFS algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId &search_depth,
        double &avg_duty)
    {
        unsigned long long total_lifetimes=0;
        unsigned long long total_runtimes =0;
        total_queued = 0;
        search_depth = 0;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (num_gpus!=1) 
                util::GRError(cudaSetDevice(gpu_idx[gpu]),
                    "BFSEnactor cudaSetDevice gpu failed", __FILE__, __LINE__);
            cudaThreadSynchronize();

            total_queued += this->total_queued[gpu];
            if (this->iterations[gpu] > search_depth) search_depth = this->iterations[gpu];
            total_lifetimes += this->total_lifetimes[gpu];
            total_runtimes += this->total_runtimes[gpu];
        }
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */
    
   /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam BFSProblem BFS Problem type.
     *
     * @param[in] problem BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename BFSProblem>
    cudaError_t EnactBFS(
    BFSProblem                          *problem,
    typename BFSProblem::VertexId       src,
    int                                 max_grid_size = 0)
    {
        util::cpu_mt::CPUBarrier cpu_barrier[2];
        ThreadSlice *thread_slices;
        CUTThread   *thread_Ids;
        printf("EnactBFS begin.\n"); fflush(stdout);

        cudaError_t retval = cudaSuccess;
        printf("EnactBFS start.\n"); fflush(stdout);

        do {
            // Determine grid size(s)
            int edge_map_occupancy      = EdgeMapPolicy::CTA_OCCUPANCY;
            int edge_map_grid_size      = MaxGridSize(edge_map_occupancy, max_grid_size);

            int vertex_map_occupancy    = VertexMapPolicy::CTA_OCCUPANCY;
            int vertex_map_grid_size    = MaxGridSize(vertex_map_occupancy, max_grid_size);

            if (DEBUG) {
                printf("BFS edge map occupancy %d, level-grid size %d\n",
                        edge_map_occupancy, edge_map_grid_size);
                printf("BFS vertex map occupancy %d, level-grid size %d\n",
                        vertex_map_occupancy, vertex_map_grid_size);
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }

            // Lazy initialization
            if (retval = Setup<BFSProblem>(problem, edge_map_grid_size, vertex_map_grid_size)) break;
            thread_slices  
        //       = malloc (sizeof(ThreadSlice <INSTRUMENT, BFSProblem>)* num_gpus);
                 = new ThreadSlice [num_gpus];
            thread_Ids  = new CUTThread[num_gpus];
            //CUTBarrier  cpu_barrier[1];// = new CUTBarrier[1];
        
            cpu_barrier[0] = util::cpu_mt::CreateBarrier(num_gpus);
            cpu_barrier[1] = util::cpu_mt::CreateBarrier(num_gpus);
            printf("%d: num_gpus = %d\n", -1, num_gpus);
            printf("%d: cpu_barrier releaseCount=%d count=%d\n", -1, cpu_barrier->releaseCount, cpu_barrier->count);
            //for (int gpu=0;gpu<num_gpus;gpu++) thread_slices[gpu]=NULL;
            // Dummy call to the BFSThread function that will never be called, to force compile of the BFSThread
            //if (num_gpus < 0) BFSThread<INSTRUMENT,EdgeMapPolicy, VertexMapPolicy, BFSProblem> (&thread_slices[0]);

            printf("EnactBFS multithread begin.\n"); fflush(stdout);
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                //thread_slices[gpu] = (ThreadSlice*) malloc(sizeof(ThreadSlice));//new ThreadSlice;
                thread_slices[gpu].thread_num           = gpu;
                thread_slices[gpu].problem              = (void*)problem;
                thread_slices[gpu].enactor              = (void*)this;
                thread_slices[gpu].cpu_barrier          = cpu_barrier;
                thread_slices[gpu].max_grid_size        = max_grid_size;
                thread_slices[gpu].edge_map_grid_size   = edge_map_grid_size;
                thread_slices[gpu].vertex_map_grid_size = vertex_map_grid_size;
                if ((num_gpus == 1) || (gpu==problem->partition_tables[0][src])) 
                     thread_slices[gpu].init_size=1;
                else thread_slices[gpu].init_size=0;
                printf("EnactBFS thread %d begin.\n", gpu); fflush(stdout);
                //pthread_t thread;
                //printf("pthread_create= %d\n", pthread_create(&thread, NULL, &(BFSThread<INSTRUMENT, EdgeMapPolicy, VertexMapPolicy, BFSProblem>), &thread_slices[gpu]));
                thread_slices[gpu].thread_Id = cutStartThread((CUT_THREADROUTINE)&(BFSThread<INSTRUMENT,EdgeMapPolicy, VertexMapPolicy, BFSProblem>),(void*)&(thread_slices[gpu]));
                //BFSThread<INSTRUMENT,EdgeMapPolicy,VertexMapPolicy, BFSProblem> (&(thread_slices[gpu]));
                printf("EnactBFS thread %d running.\n", gpu); fflush(stdout);
                //thread_slices[gpu].thread_Id=thread;
                thread_Ids[gpu]=thread_slices[gpu].thread_Id;
                if (gpu!=num_gpus-1) util::cpu_mt::sleep_millisecs(1000);
            }

            cutWaitForThreads(thread_Ids,num_gpus);
            printf("BFSThreads finished.\n"); fflush(stdout);
            util::cpu_mt::DestoryBarrier(cpu_barrier);
            util::cpu_mt::DestoryBarrier(cpu_barrier+1);
            //delete[] cpu_barrier;cpu_barrier=NULL;

            for (int gpu=0;gpu<num_gpus;gpu++)
            if (this->retvals[gpu]!=cudaSuccess) {retval=this->retvals[gpu];break;}
        } while (0);
        if (retval) return retval;
        if (DEBUG) {printf("\nGPU BFS Done.\n");fflush(stdout);}
        for (int gpu=0; gpu<num_gpus;gpu++)
        {
            thread_slices[gpu].problem = NULL;
            thread_slices[gpu].enactor = NULL;
            thread_slices[gpu].cpu_barrier = NULL;
            //free( thread_slices[gpu]); thread_slices[gpu]=NULL;
        }
        //delete[] thread_slices; thread_slices=NULL;
        //printf("..\n");fflush(stdout);
        delete[] thread_Ids;   thread_Ids    = NULL; printf("thread_Ids freed.\n"); fflush(stdout);
        delete[] thread_slices;thread_slices = NULL; printf("thread_slices freed.\n"); fflush(stdout);
        printf("BFSEnact finished.\n"); fflush(stdout);
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename BFSProblem>
    cudaError_t Enact(
        BFSProblem                      *problem,
        typename BFSProblem::VertexId    src,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
                BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                VertexMapPolicy;

                typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
                BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7>                                  // LOG_SCHEDULE_GRANULARITY
                EdgeMapPolicy;

                return EnactBFS<EdgeMapPolicy, VertexMapPolicy, BFSProblem>(
                problem, src, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace bfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
