#include "Numcpp.hpp"
#include <math.h>
#include <vector>

#define nc_t double

#define OWN(x, a, b) (a / b) * sqrt(a *x) * sqrt(a *x)
nc_t OWN_A = 1, OWN_K = 1;

void gIN(Numcpp<nc_t> &nc)
{
    for (size_t j = 0; j < nc.row; j++)
    {
        nc.matrix[j][0] = 20 + 50 * rand() * 1.0f / RAND_MAX;  // age
        nc.matrix[j][1] = 50 + 10 * rand() * 1.0f / RAND_MAX;  // weight
        nc.matrix[j][2] = 170 + 30 * rand() * 1.0f / RAND_MAX; // high

        double fam = rand() * 1.0f / RAND_MAX;
        if (fam > 0.5)
        {
            // boy
            nc.matrix[j][3] = 1;
            nc.matrix[j][1] *= 1.05;
            nc.matrix[j][2] *= 1.05;
        }
        else
        {
            // girl
            nc.matrix[j][3] = 0;
            nc.matrix[j][1] *= 0.95;
            nc.matrix[j][2] *= 0.95;
        }
    }
}

void average(Numcpp<nc_t> &nc)
{
    auto asum = 0, wsum = 0, hsum = 0;
    for (size_t i = 0; i < nc.row; i++)
    {
        asum += nc.matrix[i][0];
        wsum += nc.matrix[i][1];
        hsum += nc.matrix[i][2];
    }
    asum /= nc.row;
    wsum /= nc.row;
    hsum /= nc.row;
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][0] -= asum;
        nc.matrix[i][1] -= wsum;
        nc.matrix[i][2] -= hsum;
    }
}

void gWei(Numcpp<nc_t> &nc, nc_t age, nc_t weight, nc_t high)
{
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][0] = age;
    }
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][1] = weight;
    }
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][2] = high;
    }
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][3] = 0;
    }
}

void gCost(Numcpp<nc_t> &Inputs, Numcpp<nc_t> &Cost)
{
    for (size_t i = 0; i < Inputs.row; i++)
    {
        Cost.matrix[i][0] = Inputs.matrix[i][3];
    }
}

nc_t sigmoid(nc_t x, nc_t y)
{
    return 1 / (1 + exp(-x));
}
nc_t d_sigmoid(nc_t x, nc_t y)
{
    return sigmoid(x, y) * (1 - sigmoid(x, y));
}
nc_t Squdiff(nc_t x, nc_t y)
{
    return x * x;
}
nc_t eta = 0.01;

int main(int argc, char const *argv[])
{
    // generate random data with random values
    Numcpp<nc_t> Inputs(64, 4);
    gIN(Inputs);
    // std::cout << "RAW: " << Inputs << "\n";
    Numcpp<nc_t> Cost(Inputs.row, 1);
    gCost(Inputs, Cost);
    // std::cout << "COST: " << Cost << "\n";

    // set weight
    Numcpp<nc_t> Wei(Inputs.col, Inputs.col);
    gWei(Wei, 1, 1, 1);
    Numcpp<nc_t> Aver = Inputs * Wei;
    average(Aver);
    Aver.col -= 1;
    // std::cout << "Averaged:" << Aver << "\n";

    // Weight & BaisOut_Wei
    Numcpp<nc_t> Hider_Wei(Aver.col, 2);                 // 3*2
    Numcpp<nc_t> Hider_Bais(Aver.row, Hider_Wei.col, 0); // 16*2
    Numcpp<nc_t> Out_Wei(Hider_Wei.col, 1);              // 2*1
    Numcpp<nc_t> Out_Bais(Aver.row, Out_Wei.col, 0);     // 16*1
    //10w次
    for (size_t i = 0; i < 1000; i++)
    // while (1)
    {
        // std::cout << "############################################################################\n";
        //   Hide layer computation
        Numcpp<nc_t> z1 = Aver * Hider_Wei + Hider_Bais;
        Numcpp<nc_t> Hider = z1<sigmoid> NULL;
        // std::cout << "Hider: " << Hider << "\n";
        //   Out layer computation
        Numcpp<nc_t> z2 = Hider * Out_Wei + Out_Bais; // 16 * 2
        Numcpp<nc_t> Out = z2<sigmoid> NULL;
        // std::cout << "Out" << Out << "\n";
        //    loss computation
        Numcpp<nc_t> L = Out - Cost;
        Numcpp<nc_t> s_L = Numcpp<nc_t>(1, Inputs.row) * (L<Squdiff> NULL) / Inputs.row;
        std::cout << "Loss: " << s_L[0][0] << "\n";

        // updata wei & bais computation
        // Out
        /*
        std::cout << "Out_Wei: " << Out_Wei << "\n";
        std::cout << "Out_Bais: " << Out_Bais << "\n";
        */
        Numcpp<nc_t> dow = Hider.transpose() * (L.Hadamard(z2<d_sigmoid> NULL)) * 2;
        Out_Wei = Out_Wei - dow * eta;
        Numcpp<nc_t> dob = (L.Hadamard(z2<d_sigmoid> NULL)) * 2;
        Out_Bais = Out_Bais - dob * eta;
        /*
        std::cout << "dow: " << dow << "\n";
        std::cout << "dob: " << dob << "\n";
        std::cout << "updata Out_Wei: " << Out_Wei << "\n";
        std::cout << "updata Out_Bais: " << Out_Bais << "\n";
        */
        //  Hider
        /*
        std::cout << "Hider_Wei: " << Hider_Wei << "\n";
        std::cout << "Hider_Bais: " << Hider_Bais << "\n";
        */
        Numcpp<nc_t> dhw = (Aver.transpose() * (z1<d_sigmoid> NULL).Hadamard(L.Hadamard(z2<d_sigmoid> NULL) * Out_Wei.transpose())) * 2;
        Hider_Wei = Hider_Wei - dhw * eta;
        Numcpp<nc_t> dhb = (z1<d_sigmoid> NULL).Hadamard(L.Hadamard(z2<d_sigmoid> NULL) * Out_Wei.transpose()) * 2;
        Hider_Bais = Hider_Bais - dhb * eta;
        /*
        std::cout << "dhw: " << dhw << "\n";
        std::cout << "dhb: " << dhb << "\n";
        std::cout << "updata Hider_Wei: " << Hider_Wei << "\n";
        std::cout << "updata Hider_Bais: " << Hider_Bais << "\n";
        std::cout << "############################################################################\n";
        */
        //_sleep(500);
    }
    std::cout << "Train done.\n";
    // testing
    /*
    //  Age weight high
    Numcpp<nc_t> T(Aver.row, 3, 0);
    while (1)
    {
        printf("scan: Age && Weight && High\n");
        std::cin >> T.matrix[0][0];
        std::cin >> T.matrix[0][1];
        std::cin >> T.matrix[0][2];
        std::cout << "scan: " << (T.matrix[0][0]) << "&&" << (T.matrix[0][1]) << "&&" << (T.matrix[0][2]) << "\n";

        Numcpp<nc_t> T_Hider = (T * Hider_Wei + Hider_Bais)<sigmoid> NULL;
        Numcpp<nc_t> T_Out = (T_Hider * Out_Wei + Out_Bais)<sigmoid> NULL;
        T_Out.row = 1;
        std::cout << "Out" << T_Out << "\n";
    }
    */
    Numcpp<nc_t> T(Aver.row, 3);
    gIN(T);
    Numcpp<nc_t> T_Hider = (T * Hider_Wei + Hider_Bais)<sigmoid> NULL;
    Numcpp<nc_t> T_Out = (T_Hider * Out_Wei + Out_Bais)<sigmoid> NULL;
    std::cout << "Out" << T_Out << "\n";
    return 0;
}
