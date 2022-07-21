#ifndef INT_FUNCS_H_INCLUDED
#define INT_FUNCS_H_INCLUDED

#include <functional>
#include <Eigen/Dense>
#include <mpi.h>

struct params
{
	double xmin, ymin, dx, dy, eps, omega;
	int N, Nx, Ny;
	bool flag = true;
};

struct MPI_params
{
	int up, down, left, right;
	int xmax, ymax, Cx, Cy;
	int NxL, NyL;
};

namespace elliptic
{
	params fill_params(double xmin, double ymin, double dx, double dy, double eps, double omega, int N, int Nx, int Ny);
	void SOR_Solve(params& p, Eigen::MatrixXd& grid, Eigen::MatrixXd& F);
	void Jacobi_Solve(params& p, Eigen::MatrixXd& grid, Eigen::MatrixXd& F);
}

namespace diffusion
{
	void FTCS_Solve(Eigen::MatrixXd& grid, double dx, double dt, int Nx, int Nt, double alpha);
	void BTCS_Solve(Eigen::MatrixXd& grid, double dx, double dt, int Nx, int Nt, double alpha);
}

namespace myMPI
{
	void create_2D_cart(MPI_params& p, MPI_Comm& comm, int nproc, int rank, double R);
	void createGrid(MPI_params& pMPI, int rank, params& p, Eigen::MatrixXd& grid, Eigen::VectorXd& xVals, Eigen::VectorXd& yVals);
	void boundaryConditions(MPI_params& p, Eigen::MatrixXd& grid, Eigen::VectorXd& xVals, Eigen::VectorXd& yVals);
	void Jacobi_MPI_Solve(MPI_params& pMPI, MPI_Comm& cart_comm, params& p, Eigen::MatrixXd& grid, Eigen::MatrixXd& F);
	void output_2D_MPI(std::string f, Eigen::MatrixXd & grid, Eigen::VectorXd& xVals, Eigen::VectorXd& yVals, MPI_params & pMPI, params & p, int rank);
}


void findClosestFactors(int N, int factors[2]);
void findFactorsInRatio(int N, double R, int factors[2]);
void output_2D(params& p, Eigen::MatrixXd& f, std::fstream& fout);
Eigen::VectorXd ThomasSolve(Eigen::MatrixXd c, Eigen::VectorXd b);

#endif