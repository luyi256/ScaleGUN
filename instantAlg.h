#ifndef EMBEDDING_H
#define EMBEDDING_H
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace std;
namespace propagation
{
    class Instantgnn
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint num_threads; // Number of threads
        uint num_edges, num_nodes, dimension;
        string dataset;
        vector<vector<uint>> adj;
        uint *deg;
        vector<double> weights;
        double ***q;
        double ***residue;
        double *powdeg;
        vector<double> residue_sum;
        uint layer;
        double r;
        double init_graph(string path, string dataset, Eigen::Map<Eigen::MatrixXi> &edge_index,  uint _layer, double _r, const vector<double> &_weights, uint _num_threads);
        double init_push_graph(string path, string dataset, Eigen::Map<Eigen::MatrixXd> &X, Eigen::Map<Eigen::MatrixXi> &edge_index, uint _layer, double _r, const vector<double> &_weights, uint _num_threads, double rmax);
        int RandomWalkMethod(uint times, Eigen::Ref<Eigen::MatrixXd> result);
        void PushMethod(Eigen::Ref<Eigen::MatrixXd> result, double rmax);
        double PowerMethod(Eigen::Ref<Eigen::MatrixXd> result);
        double UpdateEdges(const vector<pair<uint, uint>> &edgepairs, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax);
        double UpdateStruct(const vector<pair<uint, uint>> &edgepairs);
        double UpdateNodes(const vector<uint> &nodes, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax);
        double UpdateFeatures(const vector<uint> &nodes, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax);
        double GetResidueSum(Eigen::Ref<Eigen::VectorXd> res);
        double UpdateStructNodes(const vector<uint> &nodes);

    private:
        // Eigen::MatrixXd MatrixMulMethod(Eigen::Ref<Eigen::MatrixXd> result);
        // void PushMethod(Eigen::Ref<Eigen::MatrixXd> result, double rmax);
        // void Push(uint times, uint start, uint ends, Eigen::Ref<Eigen::MatrixXd> result);

        void RandomWalk(uint times, uint start, uint ends, Eigen::Ref<Eigen::MatrixXd> result);

        void Push(uint start, uint ends, double rmax, Eigen::Ref<Eigen::MatrixXd> result);
        void UpdateEdgesOperator(uint start, uint ends, double rmax, vector<uint> &affectNodes, vector<vector<uint>> &affectNeighbors, Eigen::Ref<Eigen::MatrixXd> result);
        void UpdateFeatureOperator(uint start, uint ends, const vector<uint> &nodes, double rmax, Eigen::Ref<Eigen::MatrixXd> result);
        // void UpdateNodesOperator(uint start, uint ends, double rmax, vector<uint> &affectNodes, vector<vector<uint>> &affectNeighbors, Eigen::Ref<Eigen::MatrixXd> result);
        // void UpdateFeaturesOperator(uint start, uint ends, double rmax, vector<uint> &affectNodes, vector<vector<uint>> &affectNeighbors, Eigen::Ref<Eigen::MatrixXd> result);
        void Power(uint start, uint ends, Eigen::Ref<Eigen::MatrixXd> result);
        void PushOperator(uint dim, double rmax, Eigen::Ref<Eigen::MatrixXd> result, vector<uint> &candidates, vector<bool> &next_indicator_candidates, vector<uint> &next_candidates, uint k);
        uint GetDegree(uint node);
        uint GetNeighbor(uint node, uint index);
    };
    class Instantgnn_transpose
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint num_threads; // Number of threads
        uint num_edges, num_nodes, dimension;
        string dataset;
        vector<vector<uint>> adj;
        uint *deg;
        vector<double> weights;
        vector<double> **q;
        vector<double> **residue;
        double *powdeg;
        uint layer;
        vector<double> residue_sum;
        double r;
        double init_graph(string path, string dataset, Eigen::Map<Eigen::MatrixXi> &edge_index, uint _layer, double _r, const vector<double> &_weights, uint _num_threads);
        double init_push_graph(string path, string dataset, Eigen::Map<Eigen::MatrixXd> &X, Eigen::Map<Eigen::MatrixXi> &edge_index, uint _layer, double _r, const vector<double> &_weights, uint _num_threads, double rmax);
        int RandomWalkMethod(uint times, Eigen::Ref<Eigen::MatrixXd> result);
        void PushMethod(Eigen::Ref<Eigen::MatrixXd> result, double rmax);
        double PowerMethod(Eigen::Ref<Eigen::MatrixXd> result);
        double UpdateEdges(const vector<pair<uint, uint>> &edgepairs, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax);
        double UpdateStruct(const vector<pair<uint, uint>> &edgepairs);
        double UpdateNodes(const vector<uint> &nodes, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax);
        double UpdateFeatures(const vector<uint> &nodes, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax);
        double GetResidueSum(Eigen::Ref<Eigen::VectorXd> res);
        double UpdateStructNodes(const vector<uint> &nodes);

    private:
        // Eigen::MatrixXd MatrixMulMethod(Eigen::Ref<Eigen::MatrixXd> result);
        // void PushMethod(Eigen::Ref<Eigen::MatrixXd> result, double rmax);
        // void Push(uint times, uint start, uint ends, Eigen::Ref<Eigen::MatrixXd> result);

        void RandomWalk(uint times, uint start, uint ends, Eigen::Ref<Eigen::MatrixXd> result);

        void Push(uint start, uint ends, double rmax, Eigen::Ref<Eigen::MatrixXd> result);
        void UpdateEdgesOperator(uint start, uint ends, double rmax, vector<uint> &affectNodes, vector<vector<uint>> &affectNeighbors, Eigen::Ref<Eigen::MatrixXd> result);
        void UpdateFeatureOperator(uint start, uint ends, const vector<uint> &nodes, double rmax, Eigen::Ref<Eigen::MatrixXd> result);
        void Power(uint start, uint ends, Eigen::Ref<Eigen::MatrixXd> result);
        void PushOperator(uint dim, double rmax, Eigen::Ref<Eigen::MatrixXd> result, vector<uint> &candidates, vector<bool> &next_indicator_candidates, vector<uint> &next_candidates, uint k);
        uint GetDegree(uint node);
        uint GetNeighbor(uint node, uint index);
    };
}

#endif // EMBEDDING_H