#include <iostream>
#include <iomanip>
#include <fstream>

#include <functional>
#include <Eigen/Dense>
#include <mpi.h>
#include "int_funcs.h"

const double pi = 3.14159265358979323846; // 20 digits

void myMPI::boundaryConditions(MPI_params& p, Eigen::MatrixXd& grid, Eigen::VectorXd& xVals, Eigen::VectorXd& yVals)
{
	// Left and right
	if (p.Cx == 0)
		for (int i = 0; i < grid.rows(); i++)
			grid(i, 0) = 2.0 * yVals(i) * (1.0 - yVals(i));
	if (p.Cx == p.xmax)
		for (int i = 0; i < grid.rows(); i++)
			grid(i, grid.cols() - 1) = 2.0 * yVals(i) * (1.0 - yVals(i));
	// Top and bottom
	if (p.Cy == 0)
		for (int i = 0; i < grid.cols(); i++)
			grid(0, i) = xVals(i) * (1.0 - xVals(i));
	if (p.Cy == p.ymax)
		for (int i = 0; i < grid.cols(); i++)
			grid(grid.rows() - 1, i) = xVals(i) * (1.0 - xVals(i));
}

void myMPI::create_2D_cart(MPI_params& p, MPI_Comm& comm_cart, int nproc, int rank, double R)
{
	int dim[2];
	// Split 2D region using nproc into as close of a 1:1 ratio as possible
	if (abs(R - 1.0) < 0.1) findClosestFactors(nproc, dim);
	// Split 2D region using nproc into as close of a Nx:Ny ratio as possible
	//		Gives a +/-10 percent buffer
	//      R = Nx / Ny
	else findFactorsInRatio(nproc, R, dim);
	int periodic[2] = { false, false };
	int coord[2], id;

	// Create the Cartesian communicator and find coordinates
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periodic, 0, &comm_cart);
	MPI_Cart_coords(comm_cart, rank, 2, coord);
	p.Cx = coord[0];
	p.Cy = coord[1];

	// Find coordinates of largest element and broadcast to all subprocs
	MPI_Bcast(coord, 2, MPI_INT, nproc - 1, MPI_COMM_WORLD);
	p.xmax = coord[0];
	p.ymax = coord[1];

	// Identify the nearest neighbors on the Cartesian communciator
	MPI_Cart_shift(comm_cart, 0, 1, &p.left, &p.right);
	MPI_Cart_shift(comm_cart, 1, 1, &p.up, &p.down);
}

// Function to find the two factors of a number which are as 
//		close to sqrt(N) as possible
void findClosestFactors(int N, int factors[2])
{
	int candidate = sqrt(N);
	while (N % candidate != 0) candidate--;
	factors[1] = candidate;
	factors[0] = N / candidate;
}

// Function to find the two factors of a number which are as 
//		close to a specified ratio as possible
void findFactorsInRatio(int N, double R, int factors[2])
{
	int Cmin = (sqrt(N * R) < N ? sqrt(N * R) : N);
	int Cmax = Cmin;
	while (N % Cmin != 0) Cmin--;
	while (N % Cmax != 0) Cmax++;
	int Rmin = static_cast<double>(round((Cmin * Cmin) / N));
	int Rmax = static_cast<double>(round((Cmax * Cmax) / N));
	factors[0] = (abs(Rmin - R) / R < abs(Rmax - R) / R ? Cmin : Cmax);
	factors[1] = N / factors[0];
}

void myMPI::createGrid(MPI_params& pMPI, int rank, params& p, Eigen::MatrixXd& grid, Eigen::VectorXd& xVals, Eigen::VectorXd& yVals)
{
	// Remander from evenly dividing mesh points over number of x and y coords
	int remain[2] = { p.Nx % (pMPI.xmax + 1), p.Ny % (pMPI.ymax + 1) };

	// Number of grid points including ghost cells
	//		not accounting for remainder from division or edges of grid
	int mesh[2] = { 2 + p.Nx / (pMPI.xmax + 1), 2 + p.Ny / (pMPI.ymax + 1) };
	
	// If there are remainders, add one mesh points each to lower corner of grid
	if ((pMPI.Cx < remain[0]) && (pMPI.Cy < remain[1])) {
		mesh[0]++;
		mesh[1]++;
	}
	// Remainders for only x or y directions, add to appropriate mesh
	else if (pMPI.Cx < remain[0]) mesh[0]++;
	else if (pMPI.Cy < remain[1]) mesh[1]++;

	// Num of calculated mesh points on subarray
	pMPI.NxL = mesh[0] - 2;
	pMPI.NyL = mesh[1] - 2;

	// Remove outer bounds
	if (pMPI.Cx == 0) mesh[0]--;
	if (pMPI.Cx == pMPI.xmax) mesh[0]--;
	if (pMPI.Cy == 0) mesh[1]--;
	if (pMPI.Cy == pMPI.ymax) mesh[1]--;

	// Create empty grid of correct size for subarray
	grid.resize(mesh[1], mesh[0]);
	grid.setZero();

	// Create arrays of corresponding x and y values
	xVals.resize(mesh[0]);
	xVals(0) = p.xmin + p.dx * pMPI.Cx * (p.Nx / (pMPI.xmax + 1) + (p.Nx % (pMPI.xmax + 1) != 0));
	if (pMPI.Cx != 0) xVals(0) -= p.dx;
	if ((remain[0] != 0) && (pMPI.Cx > remain[0])) xVals(0) -= p.dx * (pMPI.Cx - remain[0]);
	for (int i = 1; i < mesh[0]; i++)
		xVals(i) = xVals(i - 1) + p.dx;
	yVals.resize(mesh[1]);
	yVals(0) = p.ymin + p.dy * pMPI.Cy * (p.Ny / (pMPI.ymax + 1) + (p.Ny % (pMPI.ymax + 1) != 0));
	if (pMPI.Cy != 0) yVals(0) -= p.dy;
	if ((remain[1] != 0) && (pMPI.Cy > remain[1])) yVals(0) -= p.dy * (pMPI.Cy - remain[1]);
	for (int i = 1; i < mesh[1]; i++)
		yVals(i) = yVals(i - 1) + p.dy;

	// Impose BCs
	myMPI::boundaryConditions(pMPI, grid, xVals, yVals);
}

void myMPI::Jacobi_MPI_Solve(MPI_params& pMPI, MPI_Comm& cart_comm, params& p, Eigen::MatrixXd& grid, Eigen::MatrixXd& F)
{
	// Prefactors & constants
	double beta_sq = pow(p.dx / p.dy, 2);
	double denom = 2.0 * (1.0 + beta_sq);
	double deltaf_max, deltaf_maxL, deltaf;

	// Pointer arrays for the buffers to send/receive edges
	//    Maximum lengths per node to ensure buffers are large enough
	const int maxX = 3 + p.Nx / (pMPI.xmax + 1), maxY = 3 + p.Ny / (pMPI.ymax + 1);
	double* tempDeltaX = new double[maxX]; // For top/bottom edges
	double* buffDeltaX = new double[maxX];
	double* tempDeltaY = new double[maxY]; // For left/right edges
	double* buffDeltaY = new double[maxY];

	for (int t = 0; t < p.N; t++) {
		deltaf_maxL = 0.0;
		// Jacobi iteration on all interior points of each grid 
		//      (ignore edges and ghost cells)
		for (int j = 1; j < grid.cols() - 1; j++) {
			for (int k = 1; k < grid.rows() - 1; k++) {
				deltaf = (grid(k + 1, j) + beta_sq * grid(k, j + 1) + grid(k - 1, j) + beta_sq * grid(k, j - 1) - pow(p.dx, 2) * F(k, j)) / denom - grid(k, j);
				if (abs(deltaf) > deltaf_maxL) deltaf_maxL = abs(deltaf);
				grid(k, j) += deltaf;
			}
		}
		
		// Send left
		for (int i = 1; i < grid.rows() - 1; i++) // Left edge into buffer
			tempDeltaY[i - 1] = grid(i, 1);
		MPI_Sendrecv(tempDeltaY, grid.rows() - 2, MPI_DOUBLE, pMPI.left, 0, buffDeltaY, grid.rows() - 2, MPI_DOUBLE, pMPI.right, 0, cart_comm, MPI_STATUS_IGNORE);
		if (pMPI.Cx != pMPI.xmax) // Right side does not receive
			for (int i = 1; i < grid.rows() - 1; i++) // Write new right edge into grid
				grid(i, grid.cols() - 1) = buffDeltaY[i - 1];

		// Send right
		for (int i = 1; i < grid.rows() - 1; i++)  // Right edge into buffer
			tempDeltaY[i - 1] = grid(i, grid.cols() - 2);
		MPI_Sendrecv(tempDeltaY, grid.rows() - 2, MPI_DOUBLE, pMPI.right, 0, buffDeltaY, grid.rows() - 2, MPI_DOUBLE, pMPI.left, 0, cart_comm, MPI_STATUS_IGNORE);
		if (pMPI.Cx != 0) // Left side does not receive
			for (int i = 1; i < grid.rows() - 1; i++) // Write new left edge into grid
				grid(i, 0) = buffDeltaY[i - 1];

		// Send down... Same ideas as above
		for (int i = 1; i < grid.cols() - 1; i++) 
			tempDeltaX[i - 1] = grid(grid.rows() - 2, i);
		MPI_Sendrecv(tempDeltaX, grid.cols() - 2, MPI_DOUBLE, pMPI.down, 0, buffDeltaX, grid.cols() - 2, MPI_DOUBLE, pMPI.up, 0, cart_comm, MPI_STATUS_IGNORE);
		if (pMPI.Cy != 0)
			for (int i = 1; i < grid.cols() - 1; i++)
				grid(0, i) = buffDeltaX[i - 1];

		// Send up
		for (int i = 1; i < grid.cols() - 1; i++) 
			tempDeltaX[i - 1] = grid(1, i);
		MPI_Sendrecv(tempDeltaX, grid.cols() - 2, MPI_DOUBLE, pMPI.up, 0, buffDeltaX, grid.cols() - 2, MPI_DOUBLE, pMPI.down, 0, cart_comm, MPI_STATUS_IGNORE);
		if (pMPI.Cy != pMPI.ymax)
			for (int i = 1; i < grid.cols() - 1; i++)
				grid(grid.rows() - 1, i) = buffDeltaX[i - 1];

		// Get the maximum deltaf from all processes
		MPI_Allreduce(&deltaf_maxL, &deltaf_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		if (deltaf_max <= p.eps) break;
		if (t == p.N - 1) p.flag = false; // Non-convergent
	}
	delete[] tempDeltaX;
	delete[] tempDeltaY;
	delete[] buffDeltaX;
	delete[] buffDeltaY;
}

params elliptic::fill_params(double xmin, double ymin, double dx, double dy, double eps, double omega, int N, int Nx, int Ny)
{
	params p;
	p.xmin = xmin;
	p.ymin = ymin;
	p.dx = dx;
	p.dy = dy;
	p.eps = eps;
	p.omega = omega;
	p.N = N;
	p.Nx = Nx;
	p.Ny = Ny;
	return p;
}

void myMPI::output_2D_MPI(std::string f, Eigen::MatrixXd& grid, Eigen::VectorXd& xVals, Eigen::VectorXd& yVals, MPI_params& pMPI, params& p, int rank)
{
	// Output vector of all calculated points (including edges)
	double* outVals = new double[3 * pMPI.NxL * pMPI.NyL];
	// If left edge, include boundary, otherwise exclude (overlap)
	int imin = (xVals(0) == p.xmin ? 0 : 1); 
	// If bottom edge, include boundary, otherwise exclude (overlap)
	int jmin = (yVals(0) == p.ymin ? 0 : 1);
	int c = 0, imax = imin + pMPI.NxL, jmax = jmin + pMPI.NyL;
	for (int i = imin; i < imax; i++)
		for (int j = jmin; j < jmax; j++) {
			outVals[c] = xVals(i);
			outVals[c + 1] = yVals(j);
			outVals[c + 2] = grid(j, i);
			c += 3;
		}
	
	// Convert filename from string to char array
	int n = f.length();
	char* file = new char[n + 1];
	strcpy(file, f.c_str());

	// Delete old file (if exists), then open new version
	MPI_File fh;
	MPI_File_delete(file, MPI_INFO_NULL);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_File_open(MPI_COMM_WORLD, file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

	// Offset (in bytes) from start of the file
	MPI_Offset offset = 0;
	// If not on the left edge, add size of all columns to left to offset
	if (pMPI.Cx != 0) offset += 3.0 * p.Ny * sizeof(double) * (xVals(1) - p.xmin) / p.dx;
	// If not on the bottom edge, add size of the values from rows above (in same column) to offset
	if (pMPI.Cy != 0) offset += 3.0 * pMPI.NxL * sizeof(double) * (yVals(1) - p.ymin) / p.dy;

	// Parallel I/O for all processes, given access to their portion of the output file
	MPI_File_write_at_all(fh, offset, outVals, 3 * pMPI.NxL * pMPI.NyL, MPI_DOUBLE, MPI_STATUS_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);

	// Cleanup
	delete[] outVals;
	MPI_File_close(&fh);
}

void elliptic::SOR_Solve(params& p, Eigen::MatrixXd& f, Eigen::MatrixXd& F)
{
	double beta_sq = pow(p.dx / p.dy, 2);
	double denom = 2.0 * (1.0 + beta_sq);
	double deltaf_max, deltaf;

	for (int i = 0; i < p.N; i++) {
		deltaf_max = 0.0;
		for (int j = 1; j < p.Ny - 1; j++) {
			for (int k = 1; k < p.Nx - 1; k++) {
				deltaf = (f(k + 1, j) + beta_sq * f(k, j + 1) + f(k - 1, j) + beta_sq * f(k, j - 1) - pow(p.dx, 2) * F(k, j) - denom * f(k, j)) / denom;
				if (abs(deltaf) > deltaf_max) deltaf_max = abs(deltaf);
				f(k, j) += p.omega * deltaf;
			}
		}
		if (deltaf_max <= p.eps) break;
		if (i == p.N - 1) p.flag = false;
	}
}

void elliptic::Jacobi_Solve(params& p, Eigen::MatrixXd& f, Eigen::MatrixXd& F)
{
	double beta_sq = pow(p.dx / p.dy, 2);
	double denom = 2.0 * (1.0 + beta_sq);
	double deltaf_max, deltaf;

	for (int i = 0; i < p.N; i++) {
		deltaf_max = 0.0;
		for (int j = 1; j < p.Ny - 1; j++) {
			for (int k = 1; k < p.Nx - 1; k++) {
				deltaf = (f(k + 1, j) + beta_sq * f(k, j + 1) + f(k - 1, j) + beta_sq * f(k, j - 1) - pow(p.dx, 2) * F(k, j)) / denom - f(k, j);
				if (abs(deltaf) > deltaf_max) deltaf_max = abs(deltaf);
				f(k, j) += deltaf;
			}
		}
		if (deltaf_max <= p.eps) break;
		if (i == p.N - 1) p.flag = false;
	}
}

void diffusion::FTCS_Solve(Eigen::MatrixXd& grid, double dx, double dt, int Nx, int Nt, double alpha)
{
	double denom = alpha * dt / pow(dx, 2);
	if (denom > 0.5) {
		std::cout << "NOPE";
		return;
	}
	for (int j = 0; j < Nt; j++) {
		for (int i = 1; i < Nx - 1; i++) {
			grid(i, j + 1) = grid(i, j) + denom * (grid(i + 1, j) - 2.0 * grid(i, j) + grid(i - 1, j));
		}
	}
}

void diffusion::BTCS_Solve(Eigen::MatrixXd& grid, double dx, double dt, int Nx, int Nt, double alpha)
{
	double denom = alpha * dt / pow(dx, 2);
	Eigen::MatrixXd c(Nx - 2, 3);
	c.setZero();
	for (int i = 1; i < Nx - 1; i++)
	{
		c(i - 1, 0) = -denom;
		c(i - 1, 1) = 1.0 + 2.0 * denom;
		c(i - 1, 2) = -denom;
	}
	c(0, 0) = 0.0;
	c(Nx - 3, 2) = 0.0;

	Eigen::VectorXd b(Nx - 2);
	b.setZero();

	for (int t = 1; t < Nt; t++)
	{
		for (int i = 1; i < Nx - 1; i++)
		{
			b(i - 1) = grid(i, t - 1);
		}
		c(0, 0) = 0.0;
		b(0) += denom * grid(0, t);
		c(Nx - 3, 2) = 0.0;
		b(Nx - 3) += denom * grid(Nx - 1, t);

	}
}

void output_2D(params& p, Eigen::MatrixXd& f, std::fstream& fout)
{
	double x, y;
	for (int i = 0; i < p.Nx; i++) {
		x = p.xmin + i * p.dx;
		for (int j = 0; j < p.Ny; j++) {
			y = p.ymin + j * p.dy;
			fout << x << "," << y << "," << f(i, j) << "\n";
		}
	}
}

/*
	Solution of a tridiagonal matrix using the Thomas algorithm for
		C*x=b
	The solution is destructive of the original arrays,
		therefore they are passed by value instead of by reference
		in case the calling function still needs them.
	c(N, 3) is a formatted matrix of coeffs from C(N, N)
		and b(N) are the RHS coeff
*/
Eigen::VectorXd ThomasSolve(Eigen::MatrixXd c, Eigen::VectorXd b)
{
	int N = c.rows();
	Eigen::VectorXd x(N);
	double W;
	for (int i = 1; i < N; i++)
	{
		W = c(i, 0) / c(i - 1, 1);
		b(i) -= W * b(i - 1);
		c(i, 1) -= W * c(i - 1, 2);
	}
	x(N - 1) = b(N - 1) / c(N - 1, 1);
	for (int i = N - 2; i >= 0; i--)
		x(i) = (b(i) - c(i, 2) * x(i + 1)) / c(i, 1);

	return x;
}
