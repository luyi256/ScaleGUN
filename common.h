#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <vector>
using namespace std;
typedef uint32_t uint;
#include <fstream>
#include <random>
#include <thread>
#include "SFMT/dSFMT/dSFMT.h"
#include "SFMT/dSFMT/dSFMT.c"
#include <math.h>
#include "timer.h"
thread_local dsfmt_t dsfmt_tl;
void init_dsfmt()
{
    dsfmt_init_gen_rand(&dsfmt_tl, std::random_device{}());
}
double thread_local_genrand()
{
    return dsfmt_genrand_close_open(&dsfmt_tl);
}

#endif // COMMON_H
