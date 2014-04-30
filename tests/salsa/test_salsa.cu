// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_salsa.cu
 *
 * @brief Simple test driver program for using SALSA algorithm to compute rank.
 */

#include <stdio.h> 
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <cstdlib>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BFS includes
#include <gunrock/app/salsa/salsa_enactor.cuh>
#include <gunrock/app/salsa/salsa_problem.cuh>
#include <gunrock/app/salsa/salsa_functor.cuh>

// Operator includes
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>

#include <moderngpu.cuh>

// boost includes
#include <boost/config.hpp>
#include <boost/utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/page_rank.hpp>


using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::salsa;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;

template <typename VertexId, typename Value>
struct RankPair {
    VertexId        vertex_id;
    Value           page_rank;

    RankPair(VertexId vertex_id, Value page_rank) : vertex_id(vertex_id), page_rank(page_rank) {}
};

template<typename RankPair>
bool SALSACompare(
    RankPair elem1,
    RankPair elem2)
{
    return elem1.page_rank > elem2.page_rank;
}

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
 void Usage()
 {
 printf("\ntest_salsa <graph type> <graph type args> [--device=<device_index>] "
        "[--undirected] [--instrumented] [--quick] "
        "[--v]\n"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
        "  --undirected If set then treat the graph as undirected.\n"
        "  --instrumented If set then kernels keep track of queue-search_depth\n"
        "  and barrier duty (a relative indicator of load imbalance.)\n"
        "  --quick If set will skip the CPU validation code.\n"
        );
 }

 /**
  * @brief Displays the BFS result (i.e., distance from source)
  *
  * @param[in] source_path Search depth from the source for each node.
  * @param[in] nodes Number of nodes in the graph.
  */
 template<typename Value, typename SizeT>
 void DisplaySolution(Value *hrank, Value *arank, SizeT nodes)
 { 
     //sort the top page ranks
     RankPair<SizeT, Value> *hr_list = (RankPair<SizeT, Value>*)malloc(sizeof(RankPair<SizeT, Value>) * nodes);
     RankPair<SizeT, Value> *ar_list = (RankPair<SizeT, Value>*)malloc(sizeof(RankPair<SizeT, Value>) * nodes);

     for (int i = 0; i < nodes; ++i)
     {
         hr_list[i].vertex_id = i;
         hr_list[i].page_rank = hrank[i];
         ar_list[i].vertex_id = i;
         ar_list[i].page_rank = arank[i];
     }
     std::stable_sort(hr_list, hr_list + nodes, SALSACompare<RankPair<SizeT, Value> >);
     std::stable_sort(ar_list, ar_list + nodes, SALSACompare<RankPair<SizeT, Value> >);

     // Print out at most top 10 largest components
     int top = (nodes < 10) ? nodes : 10;
     printf("Top %d Page Ranks:\n", top);
     for (int i = 0; i < top; ++i)
     {
         printf("Vertex ID: %d, Hub Rank: %5f\n", hr_list[i].vertex_id, hr_list[i].page_rank);
         printf("Vertex ID: %d, Authority Rank: %5f\n", ar_list[i].vertex_id, ar_list[i].page_rank);
     }

     free(hr_list);
     free(ar_list);
 }

 /**
  * Performance/Evaluation statistics
  */ 

struct Stats {
    char *name;
    Statistic rate;
    Statistic search_depth;
    Statistic redundant_work;
    Statistic duty;

    Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
    Stats(char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
};

/**
 * @brief Displays timing and correctness statistics
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * 
 * @param[in] stats Reference to the Stats object defined in RunTests
 * @param[in] h_rank Host-side vector stores computed page rank values for validation
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] elapsed Total elapsed kernel running time
 * @param[in] total_queued Total element queued in BFS kernel running process
 * @param[in] avg_duty Average duty of the BFS kernels
 */

void DisplayStats(
    Stats               &stats,
    double              elapsed,
    double              avg_duty)
{
    
    // Display test name
    printf("[%s] finished. ", stats.name);

    // Display the specific sample statistics
    printf(" elapsed: %.3f ms", elapsed);
    if (avg_duty != 0) {
        printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
    }
    printf("\n");
}




/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/

 /**
  * @brief A simple CPU-based reference Page Rank implementation.
  *
  * @tparam VertexId
  * @tparam Value
  * @tparam SizeT
  *
  * @param[in] graph Reference to the CSR graph we process on
  * @param[in] rank Host-side vector to store CPU computed labels for each node
  * @param[in] delta delta for computing SALSA rank
  * @param[in] error error threshold
  * @param[in] max_iter max iteration to go
  */
 template<
    typename VertexId,
    typename Value,
    typename SizeT>
void SimpleReferenceSALSA(
    const Csr<VertexId, Value, SizeT>       &graph,
    const Csr<VertexId, Value, SizeT>       &inv_graph,
    Value                                   *hrank,
    Value                                   *arank,
    SizeT                                   max_iter) 
{
    using namespace boost;

    //Preparation
    
    //
    //compute SALSA rank
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    printf("CPU BFS finished in %lf msec.\n", elapsed);
}

/**
 * @brief Run SALSA tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] delta Delta value for computing PageRank, usually set to .85
 * @param[in] error Error threshold value
 * @param[in] max_iter Max iteration for Page Rank computing
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] context CudaContext for moderngpu to use
 *
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT>
void RunTests(
    const Csr<VertexId, Value, SizeT> &graph,
    const Csr<VertexId, Value, SizeT> &inv_graph,
    SizeT max_iter,
    int max_grid_size,
    int num_gpus,
    CudaContext& context)
{
    
    typedef SALSAProblem<
        VertexId,
        SizeT,
        Value> Problem;

        // Allocate host-side label array (for both reference and gpu-computed results)
        Value    *reference_hrank       = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value    *reference_arank       = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value    *h_hrank               = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value    *h_arank               = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value    *reference_check_h     = (g_quick) ? NULL : reference_hrank;
        Value    *reference_check_a     = (g_quick) ? NULL : reference_arank;

        // Allocate BFS enactor map
        SALSAEnactor<INSTRUMENT> salsa_enactor(g_verbose);

        // Allocate problem on GPU
        Problem *csr_problem = new Problem;
        util::GRError(csr_problem->Init(
            g_stream_from_host,
            graph,
            inv_graph,
            num_gpus), "Problem SALSA Initialization Failed", __FILE__, __LINE__);

        //
        // Compute reference CPU SALSA solution for source-distance
        //
        if (reference_check_h != NULL)
        {
            printf("compute ref value\n");
            SimpleReferenceSALSA(
                    graph,
                    inv_graph,
                    reference_check_h,
                    reference_check_a,
                    max_iter);
            printf("\n");
        }

        Stats *stats = new Stats("GPU SALSA");

        long long           total_queued = 0;
        double              avg_duty = 0.0;

        // Perform BFS
        GpuTimer gpu_timer;

        util::GRError(csr_problem->Reset(salsa_enactor.GetFrontierType()), "SALSA Problem Data Reset Failed", __FILE__, __LINE__);
        gpu_timer.Start();
        util::GRError(salsa_enactor.template Enact<Problem>(context, csr_problem, max_iter, max_grid_size), "SALSA Problem Enact Failed", __FILE__, __LINE__);
        gpu_timer.Stop();

        salsa_enactor.GetStatistics(total_queued, avg_duty);

        double elapsed = gpu_timer.ElapsedMillis();

        // Copy out results
        util::GRError(csr_problem->Extract(h_hrank, h_arank), "SALSA Problem Data Extraction Failed", __FILE__, __LINE__);

        // Verify the result
        if (reference_check_a != NULL) {
            printf("Validity: ");
            CompareResults(h_hrank, reference_check_h, graph.nodes, true);
            CompareResults(h_arank, reference_check_a, graph.nodes, true);
        }
        printf("\nFirst 40 labels of the GPU result."); 
        // Display Solution
        DisplaySolution(h_hrank, h_arank, graph.nodes);

        DisplayStats(
            *stats,
            elapsed,
            avg_duty);


        // Cleanup
        delete stats;
        if (csr_problem) delete csr_problem;
        if (reference_check_h) free(reference_check_h);
        if (reference_check_a) free(reference_check_a);

        if (h_hrank) free(h_hrank);
        if (h_arank) free(h_arank);

        cudaDeviceSynchronize();
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
    Csr<VertexId, Value, SizeT> &inv_graph,
    CommandLineArgs &args,
    CudaContext& context)
{
    SizeT               max_iter            = 20;
    bool                instrumented        = false;        // Whether or not to collect instrumentation from kernels
    int                 max_grid_size       = 0;            // maximum grid size (0: leave it up to the enactor)
    int                 num_gpus            = 1;            // Number of GPUs for multi-gpu enactor to use

    instrumented = args.CheckCmdLineFlag("instrumented");
    args.GetCmdLineArgument("max-iter", max_iter);

    g_quick = args.CheckCmdLineFlag("quick");
    g_verbose = args.CheckCmdLineFlag("v");

    if (instrumented) {
        RunTests<VertexId, Value, SizeT, true>(
                        graph,
                        inv_graph,
                        max_iter,
                        max_grid_size,
                        num_gpus,
                        context);
    } else {
        RunTests<VertexId, Value, SizeT, false>(
                        graph,
                        inv_graph,
                        max_iter,
                        max_grid_size,
                        num_gpus,
                        context);
    }
}



/******************************************************************************
* Main
******************************************************************************/

int main( int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
		Usage();
		return 1;
	}

	//DeviceInit(args);
	//cudaSetDeviceFlags(cudaDeviceMapHost);
	int dev = 0;
    args.GetCmdLineArgument("device", dev);
    ContextPtr context = mgpu::CreateCudaDevice(dev);

	//srand(0);									// Presently deterministic
	//srand(time(NULL));

	// Parse graph-contruction params
	g_undirected = false;

	std::string graph_type = argv[1];
	int flags = args.ParsedArgc();
	int graph_args = argc - flags - 1;

	if (graph_args < 1) {
		Usage();
		return 1;
	}
	
	//
	// Construct graph and perform search(es)
	//

	if (graph_type == "market") {

		// Matrix-market coordinate-formatted graph file

		typedef int VertexId;							// Use as the node identifier type
		typedef float Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		Csr<VertexId, Value, SizeT> csr(false);         // default value for stream_from_host is false

		Csr<VertexId, Value, SizeT> inv_csr(false);

		if (graph_args < 1) { Usage(); return 1; }
		char *market_filename = (graph_args == 2) ? argv[2] : NULL;
		if (graphio::BuildMarketGraph<false>(
			market_filename, 
			csr, 
			g_undirected,
			false) != 0) 
		{
			return 1;
		}

        if (graphio::BuildMarketGraph<false>(
                    market_filename, 
                    inv_csr, 
                    g_undirected,
                    true) != 0) 
        {
            return 1;
        }

		csr.PrintHistogram();

		    // Run tests
		    RunTests(csr, inv_csr, args, *context);

	} else {

		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;

	}

	return 0;
} 
