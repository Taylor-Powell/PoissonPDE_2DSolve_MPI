/* -----------------------------------------------------------
 Taylor Powell                              April 18, 2022
 Elliptic PDE Solver
 MPI and Eigen libraries required
 -------------------------------------------------------------
 Input:


 Output:


 Comments:


 -----------------------------------------------------------*/

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <functional>
#include <Eigen/Dense>
#include <mpi.h>
#include "int_funcs.h"

using namespace std;
using namespace Eigen;

const double pi = 3.14159265358979323846; // 20 digits
const double e = 2.71828182845904523536; // 20 digits

const bool output = 1;

/// Bounds
const double xmin = 0.0;
const double xmax = 1.0;
const double ymin = 0.0;
const double ymax = 1.0;

/// Iteration Params
const int N = 10000;
const double eps = 1.0e-5;
const double omega = 1.2;

/// Grid params
const int Nx = 51;
const int Ny = 51;
const double dx = (xmax - xmin) / (Nx - 1);
const double dy = (ymax - ymin) / (Ny - 1);

int main(int argc, char** argv)
{
    int rank, procs;
    double R = (double)Nx / Ny;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Status status;

    MPI_params p;
    MPI_Comm cart_comm;
    myMPI::create_2D_cart(p, cart_comm, procs, rank, R);
    
    // Create the mesh on each node. Interior values are intialized to zero
    MatrixXd grid, F;
    VectorXd xVals, yVals;
    params param = elliptic::fill_params(xmin, ymin, dx, dy, eps, omega, N, Nx, Ny);
    myMPI::createGrid(p, rank, param, grid, xVals, yVals);

    // For Poisson's equation, external force (set to zero)
    F = grid;
    F.setZero();
    
    // Call the solver
    myMPI::Jacobi_MPI_Solve(p, cart_comm, param, grid, F);
    //cout << "Rank " << rank << ", solution:\n" << grid << "\n";

    if ((param.flag == false) && (rank == 0))
        cout << "Solution failed to converge\n";
    else {
        string filename = "Poisson_Sol_Jacobi";
        filename += "_Nx_"
            + to_string(Nx) + "_Ny_"
            + to_string(Ny) + ".dat";
        if (rank == 0) cout << "Solution saved to '" << filename << "\n\n";
        if (output) {
            myMPI::output_2D_MPI(filename, grid, xVals, yVals, p, param, rank);
        }
    }
    MPI_Finalize();


    return 0;
}

