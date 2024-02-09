#ifndef GRAPH_H
#define GRAPH_H

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
using namespace std;

class Graph
{
public:
	uint n; // number of nodes
	uint m; // number of edges

	vector<vector<uint>> inAdj;
	vector<vector<uint>> outAdj;
	uint *indegree;
	uint *outdegree;
	vector<uint> indices;
	vector<uint> indptr;
	Graph()
	{
	}
	~Graph()
	{
	}

	void insertEdge(uint from, uint to)
	{
		outAdj[from].push_back(to);
		inAdj[to].push_back(from);
		outdegree[from]++;
		indegree[to]++;
	}

	void deleteEdge(uint from, uint to)
	{
		uint j;
		for (j = 0; j < indegree[to]; j++)
		{
			if (inAdj[to][j] == from)
			{
				break;
			}
		}
		inAdj[to].erase(inAdj[to].begin() + j);
		indegree[to]--;

		for (j = 0; j < outdegree[from]; j++)
		{
			if (outAdj[from][j] == to)
			{
				break;
			}
		}

		outAdj[from].erase(outAdj[from].begin() + j);
		outdegree[from]--;
	}

	int isEdgeExist(uint u, uint v)
	{
		for (uint j = 0; j < outdegree[u]; j++)
		{
			if (outAdj[u][j] == v)
			{
				return -1;
			}
		}
		return 1;
	}

	void inputGraph(string path, string dataset, uint nodenum, uint edgenum)
	{
		n = nodenum;
		m = edgenum;
		indices = vector<uint>(m);
		indptr = vector<uint>(n + 1);
		// string dataset_el="data/"+dataset+"_adj_el.txt";
		string dataset_el = path + dataset + "_adj_el.txt";
		const char *p1 = dataset_el.c_str();
		if (FILE *f1 = fopen(p1, "rb"))
		{
			size_t rtn = fread(indices.data(), sizeof indices[0], indices.size(), f1);
			if (rtn != m)
				cout << "Error! " << dataset_el << " Incorrect read!" << endl;
			fclose(f1);
		}
		else
		{
			cout << dataset_el << " Not Exists." << endl;
			exit(1);
		}
		string dataset_pl = path + dataset + "_adj_pl.txt";
		const char *p2 = dataset_pl.c_str();

		if (FILE *f2 = fopen(p2, "rb"))
		{
			size_t rtn = fread(indptr.data(), sizeof indptr[0], indptr.size(), f2);
			if (rtn != n + 1)
				cout << "Error! " << dataset_pl << " Incorrect read!" << endl;
			fclose(f2);
		}
		else
		{
			cout << dataset_pl << " Not Exists." << endl;
			exit(1);
		}
		indegree = new uint[n];
		outdegree = new uint[n];
		clock_t t1 = clock();
		for (uint i = 0; i < n; i++)
		{
			indegree[i] = indptr[i + 1] - indptr[i];
			outdegree[i] = indptr[i + 1] - indptr[i];
			vector<uint> templst(indices.begin() + indptr[i], indices.begin() + indptr[i + 1]);
			outAdj.push_back(templst);
			inAdj.push_back(templst);
		}

		clock_t t2 = clock();
		cout << "m=" << m << endl;
		cout << "reading in graph takes " << (t2 - t1) / (1.0 * CLOCKS_PER_SEC) << " s." << endl;
	}

	void inputGraph_fromedgelist(string dataset, uint nodenum, uint edgenum)
	{
		m = edgenum;
		n = nodenum;
		string filename = "data/" + dataset + "_adj.txt";
		ifstream infile(filename.c_str());

		indegree = new uint[n];
		outdegree = new uint[n];
		for (uint i = 0; i < n; i++)
		{
			indegree[i] = 0;
			outdegree[i] = 0;
		}
		// read graph and get degree info
		uint from;
		uint to;
		while (infile >> from >> to)
		{
			outdegree[from]++;
			indegree[to]++;
		}

		cout << "..." << endl;

		for (uint i = 0; i < n; i++)
		{
			vector<uint> templst;
			inAdj.push_back(templst);
			outAdj.push_back(templst);
		}

		infile.clear();
		infile.seekg(0);

		clock_t t1 = clock();

		while (infile >> from >> to)
		{
			outAdj[from].push_back(to);
			inAdj[to].push_back(from);
		}
		infile.close();
		clock_t t2 = clock();
		cout << "m=" << m << endl;
		cout << "reading in graph takes " << (t2 - t1) / (1.0 * CLOCKS_PER_SEC) << " s." << endl;
	}

	uint getInSize(uint vert)
	{
		return indegree[vert];
	}
	uint getInVert(uint vert, uint pos)
	{
		return inAdj[vert][pos];
	}
	uint getOutSize(uint vert)
	{
		return outdegree[vert];
	}
	uint getOutVert(uint vert, uint pos)
	{
		return outAdj[vert][pos];
	}
	vector<uint> getOutAdjs(uint vert)
	{
		return outAdj[vert];
	}
};

#endif
