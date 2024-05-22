#include "common.h"
#include "instantAlg.h"
using namespace std;

namespace propagation
{
    double Instantgnn_transpose::GetResidueSum(Eigen::Ref<Eigen::VectorXd> res)
    {
        res=Eigen::Map<Eigen::VectorXd,Eigen::Unaligned> (residue_sum.data(),residue_sum.size());
        return 0;
    }

    double Instantgnn_transpose::init_push_graph(string path, string dataset, Eigen::Map<Eigen::MatrixXd> &X, Eigen::Map<Eigen::MatrixXi> &edge_index, uint _layer, double _r, const vector<double> &_weights, uint _num_threads, double rmax)
    {
        layer = _layer; // from 0 to _layer
        r = _r;
        num_threads = _num_threads;
        weights = _weights;
        string data_attr = path + dataset + "/" + dataset + ".attr";
        ifstream fin(data_attr.c_str(), ios::in);
        if (!fin)
        {
            cout << "Error opening file:" << data_attr << endl;
            exit(1);
        }
        fin >> num_nodes >> num_edges >> dimension;
        fin.close();
        cout << "---graph info---" << endl;
        cout << "num_nodes=" << num_nodes << endl;
        cout << "num_edges=" << num_edges << endl;
        cout << "dimension=" << dimension << endl;
        cout << "X:" << X.rows() << " " << X.cols() << endl;
        if (X.rows() != dimension || X.cols() != num_nodes)
        {
            cout << "X.rows()!=dimension || X.cols()!=num_nodes" << endl;
        }
        Eigen::MatrixXd tmp_result;
        if (X.rows() > dimension)
            tmp_result = X.transpose();
        else
            tmp_result = X;
        cout << "tmp_result:" << tmp_result.rows() << " " << tmp_result.cols() << endl;
        adj.resize(num_nodes);
        deg = new uint[num_nodes];
        residue_sum.resize(dimension, 0.0);
        memset(deg, 0, sizeof(uint) * num_nodes);
        // undirected already
        cout << "edge_index:" << edge_index.rows() << " " << edge_index.cols() << endl;
        if (edge_index.cols() != 2)
        {
            cout << "edge_index.cols()!=2" << endl;
            exit(-1);
        }

        for (uint i = 0; i < edge_index.rows(); i++)
        {
            uint u = edge_index(i, 0);
            uint v = edge_index(i, 1);
            adj[u].push_back(v);
            deg[u]++;
        }
        cout << "load adj matrix done!" << endl;

        double sum = 0.0;
        for (uint i = 0; i < num_nodes; i++)
            sum += tmp_result(0, i);
        cout << "sum:" << sum << endl;
        cout << "load feature matrix done!" << endl;

        vector<thread> threads;
        powdeg = new double[num_nodes];
        for (uint i = 0; i < num_nodes; i++)
            powdeg[i] = pow(GetDegree(i), -r);
        q = new vector<double> *[dimension];
        residue = new vector<double> *[dimension];
        for (uint i = 0; i < dimension; i++)
        {
            q[i] = new vector<double>[layer];
            residue[i] = new vector<double>[layer];
            for (uint j = 0; j < layer; j++)
            {
                q[i][j].resize(num_nodes, 0.0);
                residue[i][j].resize(num_nodes, 0.0);
            }
        }

        Timer queryTimer;
        double time;
        time = queryTimer.get_operation_time();
        uint start;
        uint ends = 0;

        uint ti;
        for (ti = 1; ti <= dimension % num_threads; ti++)
        {
            start = ends;
            ends += ceil((double)dimension / num_threads);
            threads.push_back(thread(&Instantgnn_transpose::Push, this, start, ends, rmax, Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }
        for (; ti <= num_threads; ti++)
        {
            start = ends;
            ends += dimension / num_threads;
            threads.push_back(thread(&Instantgnn_transpose::Push, this, start, ends, rmax, Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }

        for (uint t = 0; t < num_threads; t++)
            threads[t].join();
        vector<thread>().swap(threads);
        if (X.rows() > dimension)
            X = tmp_result.transpose();
        else
            X = tmp_result;
        time = queryTimer.get_operation_time();
        double pkm = peak_mem() / 1024.0 / 1024.0;
        cout << "Total graph: peak memory: " << pkm << " G" << endl;
        return time;
    }

    double Instantgnn_transpose::UpdateStruct(const vector<pair<uint, uint>> &edgepairs)
    {
        Timer queryTimer;
        double time;
        time = queryTimer.get_operation_time();
        for (auto &edgepair : edgepairs)
        {
            auto u = edgepair.first;
            auto v = edgepair.second;
            auto u_index = find(adj[v].begin(), adj[v].end(), u);
            if (u_index != adj[v].end())
            {
                adj[v].erase(u_index);
                deg[v]--;
                powdeg[v] = pow(deg[v], -r);
            }
            else
                continue;
            auto v_index = find(adj[u].begin(), adj[u].end(), v);
            if (v_index != adj[u].end())
            {
                adj[u].erase(v_index);
                deg[u]--;
                powdeg[u] = pow(deg[u], -r);
            }
            else
                continue;
        }
        time = queryTimer.get_operation_time();
        return time;
    }

    double Instantgnn_transpose::init_graph(string path, string dataset, Eigen::Map<Eigen::MatrixXi> &edge_index, uint _layer, double _r, const vector<double> &_weights, uint _num_threads)
    {
        layer = _layer; // from 0 to _layer
        r = _r;
        num_threads = _num_threads;
        weights = _weights;
        string data_attr = path + dataset + "/" + dataset + ".attr";
        ifstream fin(data_attr.c_str(), ios::in);
        if (!fin)
        {
            cout << "Error opening file:" << data_attr << endl;
            exit(1);
        }
        fin >> num_nodes >> num_edges >> dimension;
        fin.close();
        cout << "---graph info---" << endl;
        cout << "num_nodes=" << num_nodes << endl;
        cout << "num_edges=" << num_edges << endl;
        cout << "dimension=" << dimension << endl;
        adj.resize(num_nodes);
        deg = new uint[num_nodes];
        memset(deg, 0, sizeof(uint) * num_nodes);
        // undirected already
        cout << "edge_index:" << edge_index.rows() << " " << edge_index.cols() << endl;
        for (uint i = 0; i < edge_index.rows(); i++)
        {
            uint u = edge_index(i, 0);
            uint v = edge_index(i, 1);
            adj[u].push_back(v);
            deg[u]++;
        }
        cout << "load adj matrix done!" << endl;
        powdeg = new double[num_nodes];
        for (uint i = 0; i < num_nodes; i++)
        {
            powdeg[i] = pow(GetDegree(i), -r);
        }
        return 0;
    }

    void Instantgnn_transpose::PushMethod(Eigen::Ref<Eigen::MatrixXd> result, double rmax)
    {
        // cout << feat << endl;
        cout << "result:" << result.rows() << " " << result.cols() << endl;
        Eigen::MatrixXd tmp_result;
        if (result.rows() > dimension)
            tmp_result = result.transpose();
        else
            tmp_result = result;
        cout << "tmp_result:" << tmp_result.rows() << " " << tmp_result.cols() << endl;
        vector<thread> threads;
        q = new vector<double> *[dimension];
        residue = new vector<double> *[dimension];
        for (uint i = 0; i < dimension; i++)
        {
            q[i] = new vector<double>[layer];
            residue[i] = new vector<double>[layer];
            for (uint j = 0; j < layer; j++)
            {
                q[i][j].resize(num_nodes, 0.0);
                residue[i][j].resize(num_nodes, 0.0);
            }
        }
        uint start;
        uint ends = 0;
        uint ti;
        for (ti = 1; ti <= dimension % num_threads; ti++)
        {
            start = ends;
            ends += ceil((double)dimension / num_threads);
            threads.push_back(thread(&Instantgnn_transpose::Push, this, start, ends, rmax, Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }
        for (; ti <= num_threads; ti++)
        {
            start = ends;
            ends += dimension / num_threads;
            threads.push_back(thread(&Instantgnn_transpose::Push, this, start, ends, rmax, Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }

        for (uint t = 0; t < num_threads; t++)
            threads[t].join();
        vector<thread>().swap(threads);
        if (result.rows() > dimension)
            result = tmp_result.transpose();
        else
            result = tmp_result;
    }

    double Instantgnn_transpose::UpdateEdges(const vector<pair<uint, uint>> &edgepairs, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax)
    {
        Timer queryTimer;
        double time;
        time = queryTimer.get_operation_time();
        Eigen::MatrixXd tmp_result;
        if (result.rows() > dimension)
        {
            // cout << "result is transposed" << endl;
            tmp_result = result.transpose();
        }
        else
            tmp_result = result;
        vector<thread> threads;
        // update graph, delete the edges in edgepairs
        bool *isAffectNodes = new bool[num_nodes];
        memset(isAffectNodes, 0, sizeof(bool) * num_nodes);
        vector<uint> affectNodes;
        vector<vector<uint>> affectNeighbors(num_nodes);
        for (auto &edgepair : edgepairs)
        {
            auto u = edgepair.first;
            auto v = edgepair.second;
            if (u == v)
                continue;
            auto u_index = find(adj[v].begin(), adj[v].end(), u);
            if (u_index != adj[v].end())
            {
                adj[v].erase(u_index);
                deg[v]--;
                powdeg[v] = pow(deg[v], -r);
                if (!isAffectNodes[v])
                {
                    isAffectNodes[v] = true;
                    affectNodes.push_back(v);
                }
                affectNeighbors[v].push_back(u);
            }
            else
                continue;
            auto v_index = find(adj[u].begin(), adj[u].end(), v);
            if (v_index != adj[u].end())
            {
                adj[u].erase(v_index);
                deg[u]--;
                powdeg[u] = pow(deg[u], -r);
                if (!isAffectNodes[u])
                {
                    isAffectNodes[u] = true;
                    affectNodes.push_back(u);
                }
                affectNeighbors[u].push_back(v);
            }
            else
                continue;
        }

        // update q and residue
        uint start;
        uint ends = 0;
        // vector<double> updatetimes(update_num_threads, 0.0);
        vector<vector<uint>> candidates(dimension);

        uint ti;
        for (ti = 1; ti <= dimension % update_num_threads; ti++)
        {
            start = ends;
            ends += ceil((double)dimension / update_num_threads);
            threads.push_back(thread(&Instantgnn_transpose::UpdateEdgesOperator, this, start, ends, rmax, ref(affectNodes), ref(affectNeighbors), Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }
        for (; ti <= update_num_threads; ti++)
        {
            start = ends;
            ends += dimension / update_num_threads;
            threads.push_back(thread(&Instantgnn_transpose::UpdateEdgesOperator, this, start, ends, rmax, ref(affectNodes), ref(affectNeighbors), Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }

        for (uint t = 0; t < update_num_threads; t++)
            threads[t].join();
        vector<thread>().swap(threads);
        if (result.rows() > dimension)
            result = tmp_result.transpose();
        else
            result = tmp_result;
        time = queryTimer.get_operation_time();
        // double threadtime = 0.0;
        // for (uint i = 0; i < update_num_threads; i++)
        //     threadtime += updatetimes[i];
        // threadtime /= update_num_threads;
        // cout << "avg thread push update time:" << threadtime << endl;
        return time;
    }

    double Instantgnn_transpose::UpdateStructNodes(const vector<uint> &nodes)
    {
        Timer queryTimer;
        double time;
        time = queryTimer.get_operation_time();
        for (auto &u : nodes)
        {
            for (auto nei:adj[u])
            {
                auto u_index = find(adj[nei].begin(), adj[nei].end(), u);
                if (u_index != adj[nei].end())
                    adj[nei].erase(u_index);
                else continue;
                deg[nei]--;
                powdeg[nei]=pow(deg[nei],-r);
            }
            deg[u]=1;
            powdeg[u]=pow(deg[u],-r);
            adj[u].clear();
            adj[u].push_back(u);
        }
        time = queryTimer.get_operation_time();
        return time;
    }


    double Instantgnn_transpose::UpdateNodes(const vector<uint> &nodes, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax)
    {
        Timer queryTimer;
        double time;
        time = queryTimer.get_operation_time();
        Eigen::MatrixXd tmp_result;
        if (result.rows() > dimension)
        {
            // cout << "result is transposed" << endl;
            tmp_result = result.transpose();
        }
        else
            tmp_result = result;
        vector<thread> threads;
        // update graph, delete the edges in edgepairs
        bool *isAffectNodes = new bool[num_nodes];
        memset(isAffectNodes, 0, sizeof(bool) * num_nodes);
        vector<uint> affectNodes;
        vector<vector<uint>> affectNeighbors(num_nodes);
        for (auto &node : nodes)
        {
            if (!isAffectNodes[node])
            {
                isAffectNodes[node] = true;
                affectNodes.push_back(node);
            }
            for (auto nei : adj[node])
            {
                if (nei == node)
                    continue;
                auto u_index = find(adj[nei].begin(), adj[nei].end(), node);
                if (u_index != adj[nei].end())
                {
                    adj[nei].erase(u_index);
                    deg[nei]--;
                    powdeg[nei] = pow(deg[nei], -r);
                    if (!isAffectNodes[nei])
                    {
                        isAffectNodes[nei] = true;
                        affectNodes.push_back(nei);
                    }
                    affectNeighbors[nei].push_back(node);
                    affectNeighbors[node].push_back(nei);
                }
                else
                    continue;
            }
            deg[node] = 1;
            powdeg[node] = pow(deg[node], -r);
            adj[node].clear();
            adj[node].push_back(node);
        }

        // update q and residue
        uint start;
        uint ends = 0;

        vector<vector<uint>> candidates(dimension);
        uint ti;
        for (ti = 1; ti <= dimension % update_num_threads; ti++)
        {
            start = ends;
            ends += ceil((double)dimension / update_num_threads);
            threads.push_back(thread(&Instantgnn_transpose::UpdateEdgesOperator, this, start, ends, rmax, ref(affectNodes), ref(affectNeighbors), Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }
        for (; ti <= update_num_threads; ti++)
        {
            start = ends;
            ends += dimension / update_num_threads;
            threads.push_back(thread(&Instantgnn_transpose::UpdateEdgesOperator, this, start, ends, rmax, ref(affectNodes), ref(affectNeighbors), Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }
        for (uint t = 0; t < update_num_threads; t++)
            threads[t].join();
        vector<thread>().swap(threads);
        if (result.rows() > dimension)
            result = tmp_result.transpose();
        else
            result = tmp_result;
        time = queryTimer.get_operation_time();
        return time;
    }

    double Instantgnn_transpose::UpdateFeatures(const vector<uint> &nodes, Eigen::Ref<Eigen::MatrixXd> result, uint update_num_threads, double rmax)
    {
        Timer queryTimer;
        double time;
        time = queryTimer.get_operation_time();
        Eigen::MatrixXd tmp_result;
        if (result.rows() > dimension)
        {
            // cout << "result is transposed" << endl;
            tmp_result = result.transpose();
        }
        else
            tmp_result = result;
        uint start;
        uint ends = 0;
        vector<vector<uint>> candidates(dimension);
        vector<thread> threads;
        uint ti;
        for (ti = 1; ti <= dimension % update_num_threads; ti++)
        {
            start = ends;
            ends += ceil((double)dimension / update_num_threads);
            threads.push_back(thread(&Instantgnn_transpose::UpdateFeatureOperator, this, start, ends, nodes, rmax, Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }
        for (; ti <= update_num_threads; ti++)
        {
            start = ends;
            ends += dimension / update_num_threads;
            threads.push_back(thread(&Instantgnn_transpose::UpdateFeatureOperator, this, start, ends, nodes, rmax, Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }
        for (uint t = 0; t < update_num_threads; t++)
            threads[t].join();
        vector<thread>().swap(threads);
        if (result.rows() > dimension)
            result = tmp_result.transpose();
        else
            result = tmp_result;
        time = queryTimer.get_operation_time();
        return time;
    }

    void Instantgnn_transpose::UpdateFeatureOperator(uint start, uint ends, const vector<uint> &nodes, double rmax, Eigen::Ref<Eigen::MatrixXd> result)
    {
        for (uint i = start; i < ends; i++)
        {
            vector<uint> candidates;
            vector<bool> indicator_candidates(num_nodes, false);
            uint k = 0;
            for (auto node : nodes)
            {
                residue_sum[i]-=residue[i][k][node];
                residue[i][k][node] = -q[i][k][node];
                residue_sum[i]+=residue[i][k][node];
                if (fabs(residue[i][k][node]) > rmax)
                {
                    candidates.push_back(node);
                    indicator_candidates[node] = true;
                }
            }
            vector<bool>().swap(indicator_candidates);

            for (k = 1; k < layer; k++)
            {
                vector<bool> next_indicator_candidates(num_nodes, false);
                vector<uint> next_candidates;
                PushOperator(i, rmax, Eigen::Ref<Eigen::MatrixXd>(result), candidates, next_indicator_candidates, next_candidates, k - 1);
                // after Push Operator, candidates are the k-th layer candidates, next_candidates are the (k+1)-th layer candidates
            }
        }
    }

    void Instantgnn_transpose::UpdateEdgesOperator(uint start, uint ends, double rmax, vector<uint> &affectNodes, vector<vector<uint>> &affectNeighbors, Eigen::Ref<Eigen::MatrixXd> result)
    {
        for (uint i = start; i < ends; i++)
        {
            vector<uint> candidates;
            vector<bool> indicator_candidates(num_nodes, false);
            uint k = 0;
            for (auto node : affectNodes)
            {
                double feat = (q[i][k][node] + residue[i][k][node]) / pow(GetDegree(node) + affectNeighbors[node].size(), r);
                q[i][k][node] = q[i][k][node] * GetDegree(node) / (GetDegree(node) + affectNeighbors[node].size());
                residue_sum[i]-=residue[i][k][node];
                residue[i][k][node] = feat * pow(GetDegree(node), r) - q[i][k][node];
                residue_sum[i]+=residue[i][k][node];
                result(i, node) = weights[k] * feat;
                if (fabs(residue[i][k][node]) > rmax)
                {
                    candidates.push_back(node);
                    indicator_candidates[node] = true;
                }
            }
            vector<bool>().swap(indicator_candidates);

            for (k = 1; k < layer; k++)
            {
                vector<bool> next_indicator_candidates(num_nodes, false);
                vector<uint> next_candidates;
                for (auto node : affectNodes)
                {
                    // cout << "before update result(" << i << "," << node << ")=" << result(i, node) << endl;
                    // cout << "update result(" << i << "," << node << ")=" << result(i, node) << endl;
                    q[i][k][node] = q[i][k][node] * GetDegree(node) / (GetDegree(node) + affectNeighbors[node].size());
                    result(i, node) += weights[k] * powdeg[node] * q[i][k][node];
                    // if (isnan(result(i, node)))
                    // {
                    //     cout << "q[" << i << "][" << k << "][" << node << "]=" << q[i][k][node] << endl;
                    //     cout << "residue[" << i << "][" << k << "][" << node << "]=" << residue[i][k][node] << endl;
                    //     cout << "deg[" << node << "]=" << deg[node] << endl;
                    //     cout << "feat(" << i << "," << node << ")=" << feat(i, node) << endl;
                    //     cout << "result(" << i << "," << node << ")=" << result(i, node) << endl;
                    //     cout << "error" << endl;
                    //     exit(-1);
                    // }
                    double tmp=q[i][k][node] / GetDegree(node) * affectNeighbors[node].size();
                    residue[i][k][node] += tmp;
                    residue_sum[i]+=tmp;
                    for (auto nei : affectNeighbors[node])
                    {
                        residue[i][k][node] -= q[i][k - 1][nei] / GetDegree(nei);
                        residue_sum[i]-=q[i][k - 1][nei] / GetDegree(nei);
                    }
                    if (fabs(residue[i][k][node]) > rmax && k < layer - 1)
                    {
                        next_indicator_candidates[node] = true;
                        next_candidates.push_back(node);
                    }
                    else if (k == layer - 1)
                    {
                        q[i][k][node] += residue[i][k][node];
                        result(i, node) += weights[k] * powdeg[node] * residue[i][k][node];
                        // if (isnan(result(i, node)))
                        // {
                        //     cout << "q[" << i << "][" << k << "][" << node << "]=" << q[i][k][node] << endl;
                        //     cout << "residue[" << i << "][" << k << "][" << node << "]=" << residue[i][k][node] << endl;
                        //     cout << "deg[" << node << "]=" << deg[node] << endl;
                        //     cout << "feat(" << i << "," << node << ")=" << feat(i, node) << endl;
                        //     cout << "result(" << i << "," << node << ")=" << result(i, node) << endl;
                        //     cout << "error" << endl;
                        //     exit(-1);
                        // }
                        residue_sum[i]-=residue[i][k][node];
                        residue[i][k][node] = 0.0;
                    }
                }

                PushOperator(i, rmax, Eigen::Ref<Eigen::MatrixXd>(result), candidates, next_indicator_candidates, next_candidates, k - 1);
                // after Push Operator, candidates are the k-th layer candidates, next_candidates are the (k+1)-th layer candidates
            }
        }
    }

    // PushOperator(), k-th layer, one dimension
    void Instantgnn_transpose::PushOperator(uint dim, double rmax, Eigen::Ref<Eigen::MatrixXd> result, vector<uint> &candidates, vector<bool> &next_indicator_candidates, vector<uint> &next_candidates, uint k)
    {
        while (!candidates.empty())
        {
            auto source = candidates.back();
            candidates.pop_back();
            auto num_neighbor = GetDegree(source);
            if (num_neighbor == 0)
                continue;
            for (uint l = 0; l < num_neighbor; l++)
            {
                auto w = GetNeighbor(source, l);
                // if (w > num_nodes)
                // {
                //     cout << "source=" << source << " l=" << l << " num_neighbor=" << num_neighbor << endl;
                //     cout << "w=" << w << " num_nodes=" << num_nodes << endl;
                //     cout << "error" << endl;
                //     exit(-1);
                // }
                if (k == layer - 2)
                {
                    result(dim, w) += weights[k + 1] * residue[dim][k][source] / num_neighbor * powdeg[w];
                    q[dim][k + 1][w] += residue[dim][k][source] / num_neighbor;
                    // if (isnan(result(dim, w)))
                    // {
                    //     cout << "q[" << dim << "][" << k + 1 << "][" << w << "]=" << q[dim][k + 1][w] << endl;
                    //     cout << "residue[" << dim << "][" << k << "][" << source << "]=" << residue[dim][k][source] << endl;
                    //     cout << "deg[" << w << "]=" << deg[w] << endl;
                    //     cout << "feat(" << dim << "," << w << ")=" << feat(dim, w) << endl;
                    //     cout << "result(" << dim << "," << w << ")=" << result(dim, w) << endl;
                    //     cout << "error" << endl;
                    //     exit(-1);
                    // }
                    // cout << "***result(" << dim << "," << w << ")=" << result(dim, w) << endl;
                    // cout << (q[dim][0][w] * 0.1 + q[dim][1][w] * 0.09) * pow(deg[w], -r) << endl;
                    // cout << "***" << endl;
                }
                else
                {
                    residue[dim][k + 1][w] += residue[dim][k][source] / num_neighbor;
                    residue_sum[dim]+=residue[dim][k][source] / num_neighbor;
                    if (!next_indicator_candidates[w] && fabs(residue[dim][k + 1][w]) > rmax)
                    {
                        next_indicator_candidates[w] = true;
                        next_candidates.push_back(w);
                    }
                }
            }
            q[dim][k][source] += residue[dim][k][source];
            result(dim, source) += weights[k] * residue[dim][k][source] * powdeg[source];
            // if (isnan(result(dim, source)))
            // {
            //     cout << "q[" << dim << "][" << k << "][" << source << "]=" << q[dim][k][source] << endl;
            //     cout << "residue[" << dim << "][" << k << "][" << source << "]=" << residue[dim][k][source] << endl;
            //     cout << "deg[" << source << "]=" << deg[source] << endl;
            //     cout << "feat(" << dim << "," << source << ")=" << feat(dim, source) << endl;
            //     cout << "result(" << dim << "," << source << ")=" << result(dim, source) << endl;
            //     cout << "error" << endl;
            //     exit(-1);
            // }
            // cout << "***result(" << dim << "," << source << ")=" << result(dim, source) << endl;
            // cout << (q[dim][0][source] * 0.1 + q[dim][1][source] * 0.09) * pow(deg[source], -r) << endl;
            // cout << "***" << endl;
            residue_sum[dim]-=residue[dim][k][source];
            residue[dim][k][source] = 0.0;
        }
        candidates = next_candidates;
    }

    void Instantgnn_transpose::Push(uint start, uint ends, double rmax, Eigen::Ref<Eigen::MatrixXd> result)
    {
        for (uint i = start; i < ends; i++) // dimension
        {
            vector<uint> candidates;
            vector<bool> indicator_candidates(num_nodes, false);
            for (uint j = 0; j < num_nodes; j++) // nodes
            {
                if (GetDegree(j) == 0) // actually, no deg-0 nodes since self-loop
                {
                    residue[i][0][j] = 0.0;                   // never push, since deg[j]=0
                    result(i, j) = weights[0] * result(i, j); // so calculate the layer 0 by hand
                }
                else
                {
                    // no candidate has 0 degree
                    residue[i][0][j] = result(i, j) * pow(GetDegree(j), r);
                    residue_sum[i]+=residue[i][0][j];
                    result(i, j) = 0.0;
                    if (fabs(residue[i][0][j]) > rmax)
                    {
                        candidates.push_back(j);
                        indicator_candidates[j] = true;
                    }
                }
            }
            vector<bool>().swap(indicator_candidates);
            for (uint k = 0; k < layer - 1; k++)
            {
                vector<bool> next_indicator_candidates(num_nodes, false);
                vector<uint> next_candidates;
                PushOperator(i, rmax, Eigen::Ref<Eigen::MatrixXd>(result), candidates, next_indicator_candidates, next_candidates, k);
            }
            vector<uint>().swap(candidates);
            // uint k = 0;
            // for (uint j = 0; j < num_nodes; j++)
            // {
            //     cout << "=====check node " << j << "=====" << endl;
            //     cout << "residue[" << i << "][" << k << "][" << j << "]=" << residue[i][k][j] << endl;
            //     cout << "q[" << i << "][" << k << "][" << j << "]=" << q[i][k][j] << endl;
            //     cout << "deg[" << j << "]=" << deg[j] << endl;
            //     cout << "feat(" << i << "," << j << ")=" << feat(i, j) << endl;
            //     cout << "left hand side:" << residue[i][k][j] + q[i][k][j] << endl;
            //     cout << "right hand side:" << feat(i, j) * pow(deg[j], r) << endl;

            //     double sum = 0;
            //     for (uint l = 0; l < adj[j].size(); l++)
            //     {
            //         cout << "neighbor " << l << " =" << adj[j][l] << endl;
            //         cout << "q[" << i << "][" << k << "][" << adj[j][l] << "]=" << q[i][k][adj[j][l]] << endl;
            //         cout << "deg[" << adj[j][l] << "]=" << deg[adj[j][l]] << endl;
            //         cout << "add " << q[i][k][adj[j][l]] / deg[adj[j][l]] << endl;
            //         sum += q[i][k][adj[j][l]] / deg[adj[j][l]];
            //     }
            //     cout << "residue[" << i << "][" << k + 1 << "][" << j << "]=" << residue[i][k + 1][j] << endl;
            //     cout << "q[" << i << "][" << k + 1 << "][" << j << "]=" << q[i][k + 1][j] << endl;
            //     cout << "left hand side: " << residue[i][k + 1][j] + q[i][k + 1][j] << endl;
            //     cout << "right hand side:" << sum << endl;
            // }
        }
    }

    double Instantgnn_transpose::PowerMethod(Eigen::Ref<Eigen::MatrixXd> result)
    {
        Timer queryTimer;
        double time;
        time = queryTimer.get_operation_time();
        // cout << "result:" << result.rows() << " " << result.cols() << endl;
        Eigen::MatrixXd tmp_result;
        if (result.rows() > dimension)
            tmp_result = result.transpose();
        else
            tmp_result = result;
        // cout << "tmp_result:" << tmp_result.rows() << " " << tmp_result.cols() << endl;
        vector<thread> threads;
        uint start;
        uint ends = 0;

        uint ti;
        for (ti = 1; ti <= dimension % num_threads; ti++)
        {
            start = ends;
            ends += ceil((double)dimension / num_threads);
            threads.push_back(thread(&Instantgnn_transpose::Power, this, start, ends, Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }
        for (; ti <= num_threads; ti++)
        {
            start = ends;
            ends += dimension / num_threads;
            threads.push_back(thread(&Instantgnn_transpose::Power, this, start, ends, Eigen::Ref<Eigen::MatrixXd>(tmp_result)));
        }

        for (uint t = 0; t < num_threads; t++)
            threads[t].join();
        vector<thread>().swap(threads);
        if (result.rows() > dimension)
            result = tmp_result.transpose();
        else
            result = tmp_result;
        time = queryTimer.get_operation_time();
        return time;
    }

    void Instantgnn_transpose::Power(uint start, uint ends, Eigen::Ref<Eigen::MatrixXd> result)
    {
        auto feat = result;
        for (uint i = start; i < ends; i++)
        {
            vector<double> tmp(num_nodes, 0.0);
            for (uint j = 0; j < num_nodes; j++)
            {
                if (GetDegree(j) == 0)
                    tmp[j] = 0;
                else
                    tmp[j] = feat(i, j) * pow(GetDegree(j), r);
                result(i, j) = weights[0] * feat(i, j);
            }
            for (uint k = 1; k < layer; k++)
            {
                vector<double> next_tmp(num_nodes, 0.0);
                for (uint j = 0; j < num_nodes; j++)
                {
                    auto num_neighbor = GetDegree(j);
                    for (uint l = 0; l < num_neighbor; l++)
                    {
                        auto w = GetNeighbor(j, l);
                        next_tmp[w] += tmp[j] / num_neighbor;
                    }
                }
                for (uint j = 0; j < num_nodes; j++)
                {
                    if (GetDegree(j) == 0)
                        continue;
                    result(i, j) += weights[k] * next_tmp[j] * powdeg[j];
                }
                tmp = next_tmp;
            }
        }
    }

    uint Instantgnn_transpose::GetDegree(uint node)
    {
        return deg[node];
    }

    uint Instantgnn_transpose::GetNeighbor(uint node, uint index)
    {
        return adj[node][index];
    }
}