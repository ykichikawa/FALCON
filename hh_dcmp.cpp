//-O3 -march=native -fopenmp -isystem Eigen/ -std=c++11
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include <search.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <vector>
#include <omp.h>
#include <time.h>
#include <queue>
#include <random>
#include <limits>
#define BUFSIZE 1000000
using namespace std;

Eigen::MatrixXd readTable(ifstream &stream);
Eigen::VectorXi wccDecompose(Eigen::SparseMatrix<double, Eigen::RowMajor> &adj);

int main(int argc, char *argv[])
{

  int nodeNum, i, j, k, l, temp_i, temp_i2, temp_i3, linkNum;
  double temp_d, temp_d2;
  Eigen::VectorXd pot, b, temp_v;
  Eigen::MatrixXd in;
  Eigen::SparseMatrix<double, Eigen::RowMajor> adj, flow, A, B;
  typedef Eigen::Triplet<double> T;
  vector<T> trip;
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::DiagonalPreconditioner<double> /**/> solver;
  double tol = 1e-10;

  if (argc >= 3)
  {
    tol = pow(10, atof(argv[2]));
  }

  std::numeric_limits<double>::quiet_NaN();
  std::mt19937 mt((int)time(NULL));
  Eigen::initParallel();
  cout << "NbThreads = " << Eigen::nbThreads() << ", tol: " << tol << endl;

  if (argc < 2)
  {
    cout << "<program> <input file>" << endl;
    return 1;
  }

  string input_file(argv[1]);
  ifstream fin(input_file);
  if (!fin)
  {
    cout << "can not open input file" << endl;
    return 1;
  }
  in = readTable(fin);
  std::size_t found = input_file.rfind(".");
  input_file.erase(found, input_file.length() - found);
  string output_file;
  /*if (argc > 2)
  {
    output_file = argv[2];
    found = output_file.rfind(".");
    output_file.erase(found, output_file.length() - found);
  }
  else
  {*/
  output_file = input_file;
  //}

  if (in.rows() == 0 | in.cols() < 2)
  {
    cout << "Input file error" << endl;
    return 1;
  }

  vector<int> unode(in.data(), in.data() + in.rows() * 2);
  sort(unode.begin(), unode.end());
  unode.erase(unique(unode.begin(), unode.end()), unode.end());
  vector<int> nodeIndex(unode.begin(), unode.end());
  nodeNum = unode.size();

  decltype(nodeIndex)::iterator first = nodeIndex.begin(), last = nodeIndex.end(), it;
  for (j = 0; j < 2; j++)
  {
    for (i = 0, k = in.rows(); i < k; i++)
    {
      it = lower_bound(first, last, in.coeff(i, j));
      in.coeffRef(i, j) = it - first;
    }
  }

  cout << "Input file \"" << argv[1] << "\" [" << in.rows() << "x" << in.cols() << "]" << endl;
  cout << "Input graph: " << nodeNum << "nodes, " << in.rows() << " links" << endl;
  /*vector<int> rind(in.rows());
  for (i = 0, j = rind.size(); i < j; ++i)
  {
    rind[i] = i;
  }
*/
  if (in.cols() >= 3)
  {
    for (i = 0, k = in.rows(); i < k; i++)
    {
      trip.push_back(T(in(i, 0), in(i, 1), in(i, 2)));
    }
  }
  else if (in.cols() == 2)
  {
    for (i = 0, k = in.rows(); i < k; i++)
    {
      trip.push_back(T(in(i, 0), in(i, 1), 1));
    }
  }
  adj.resize(nodeNum, nodeNum);
  adj.setFromTriplets(trip.begin(), trip.end());
  adj = (adj - Eigen::SparseMatrix<double, Eigen::RowMajor>(adj.diagonal().asDiagonal())).pruned();
  adj.makeCompressed();

  Eigen::VectorXi wcc_index = wccDecompose(adj);
  bool induced = (wcc_index.array() > 0).any();
  int wccs = wcc_index.maxCoeff();
  cout << "WCC clusters: " << wccs + 1 << endl;
  /*if (wccs > 0)
  {
    cout << "warning: The Helmholtz-Hodge decomposition applies only to the largest connected component. Potentials other than the largest connected component of the output file are filled with NaN." << endl;
  }*/
  std::vector<int> sub_indices;
  for (i = 0; i < wcc_index.size(); i++)
  {
    if (wcc_index.coeff(i) == 0)
      sub_indices.push_back(i);
  }
  Eigen::SparseMatrix<double, Eigen::RowMajor> sub_adj(sub_indices.size(), sub_indices.size());
  if (induced)
  {
    for (i = 0, j = 0; i < wcc_index.size(); i++)
    {
      if (wcc_index.coeff(i) == 0)
      {
        wcc_index.coeffRef(i) = ++j;
      }
      else
      {
        wcc_index.coeffRef(i) = 0;
      }
    }
    std::vector<int> sub_innerSize;
    for (i = 0; i < sub_indices.size(); i++)
    {
      sub_innerSize.push_back(adj.row(sub_indices[i]).nonZeros() + 1);
    }
    sub_adj.reserve(sub_innerSize);
    for (i = 0; i < sub_indices.size(); i++)
    {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(adj, sub_indices[i]); it; ++it)
      {
        if (wcc_index.coeff(it.index()))
        {
          sub_adj.insert(i, wcc_index.coeff(it.index()) - 1) = it.value();
        }
      }
    }
    sub_adj.makeCompressed();
  }
  else
  {
    sub_adj = adj;
  }
  int sub_nodeNum = sub_adj.rows();
  flow = (sub_adj - Eigen::SparseMatrix<double, Eigen::RowMajor>(sub_adj.transpose()));
  flow = (flow - Eigen::SparseMatrix<double, Eigen::RowMajor>(flow.diagonal().asDiagonal()));

  A = -sub_adj.cwiseAbs() - Eigen::SparseMatrix<double, Eigen::RowMajor>(sub_adj.transpose().cwiseAbs());
  Eigen::VectorXd A_rowWise_sum(A.rows());
  for (i = 0; i < A.rows(); i++)
  {
    A_rowWise_sum.coeffRef(i) = A.row(i).sum();
  }
  A = (A - Eigen::SparseMatrix<double, Eigen::RowMajor>(A_rowWise_sum.asDiagonal())).pruned();
  b.resize(sub_nodeNum);
  for (i = 0; i < sub_nodeNum; i++)
  {
    b.coeffRef(i) = flow.row(i).sum();
  }
  cout << "HH decompose starts" << endl;
  solver.setTolerance(tol);
  solver.compute(A);
  if (solver.info() != Eigen::Success)
  {
    cout << "decomposition failed : " << solver.error() << endl;
    return 1;
  }
  pot = solver.solve(b);
  if (solver.info() != Eigen::Success)
  {
    cout << "solving failed : " << solver.error() << endl;
    return 1;
  }
  cout << "Complete" << endl;
  cout << "#iterations: " << solver.iterations() << endl;
  cout << "estimated error: " << solver.error() << endl;
  pot = pot.array() - pot.mean();
  Eigen::MatrixXd pot_out = Eigen::MatrixXd::Constant(nodeNum, 2, NAN);
  pot_out.col(0) = (Eigen::Map<Eigen::VectorXi>(nodeIndex.data(), nodeIndex.size())).cast<double>();
  if (induced)
  {
    for (i = 0; i < sub_indices.size(); i++)
    {
      pot_out.coeffRef(sub_indices[i], 1) = pot.coeff(i);
    }
  }
  else
  {
    pot_out.col(1) = pot;
  }
  string pot_outfile = output_file + "_potential.dat";
  FILE *fpot;
  if ((fpot = fopen(pot_outfile.c_str(), "w")) == NULL)
  {
    cout << "output file error" << endl;
    exit(1);
  }
  for (i = 0; i < pot_out.rows(); i++)
  {
    fprintf(fpot, "%d\t%.10f\n", nodeIndex[i], pot_out.coeffRef(i, 1));
  }
  fclose(fpot);
  //cout << "output \"" << output_file + pot_outfile << "\"" << endl;
  cout << "output \"" << pot_outfile << "\"" << endl;
  B = A.triangularView<Eigen::StrictlyUpper>();
  trip.clear();
  trip.reserve(B.nonZeros() * 2);
  Eigen::SparseMatrix<double, Eigen::RowMajor> p_flow(sub_nodeNum, sub_nodeNum);
  Eigen::MatrixXd p_out = Eigen::MatrixXd::Zero(B.nonZeros(), 3), l_out = Eigen::MatrixXd::Zero(B.nonZeros(), 3);

  for (i = j = 0; i < sub_nodeNum; i++)
  {
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(B, i); it; ++it)
    {
      temp_d = -it.value() * (pot.coeff(i) - pot.coeff(it.index()));
      trip.push_back(T(i, it.index(), temp_d));
      trip.push_back(T(it.index(), i, -temp_d));
      if (temp_d > 0)
      {
        p_out.coeffRef(j, 0) = (double)nodeIndex[sub_indices[i]];
        p_out.coeffRef(j, 1) = (double)nodeIndex[sub_indices[it.index()]];
        p_out.coeffRef(j, 2) = temp_d;
        ++j;
      }
      else if (temp_d < 0)
      {
        p_out.coeffRef(j, 0) = (double)nodeIndex[sub_indices[it.index()]];
        p_out.coeffRef(j, 1) = (double)nodeIndex[sub_indices[i]];
        p_out.coeffRef(j, 2) = -temp_d;
        ++j;
      }
    }
  }
  p_out.conservativeResize(j, Eigen::NoChange);

  string p_outfile = output_file + "_potential_flow.dat";
  //ofstream fp_out(output_file + p_outfile);
  FILE *fp;
  if ((fp = fopen(p_outfile.c_str(), "w")) == NULL)
  {
    cout << "output file error" << endl;
    exit(1);
  }
  Eigen::MatrixXi temp_mt(p_out.rows(), 2);
  temp_mt.col(0) = p_out.col(0).cast<int>();
  temp_mt.col(1) = p_out.col(1).cast<int>();
  for (i = 0; i < p_out.rows(); i++)
  {
    fprintf(fp, "%d\t%d\t%.10f\n", temp_mt.coeffRef(i, 0), temp_mt.coeffRef(i, 1), p_out.coeffRef(i, 2));
  }
  //fp_out.precision(10);
  //fp_out << fixed << setprecision(10) << p_out << endl;
  //cout << "output \"" << output_file + p_outfile << "\"" << endl;
  cout << "output \"" << p_outfile << "\"" << endl;

  p_flow.setFromTriplets(trip.begin(), trip.end());
  Eigen::SparseMatrix<double, Eigen::RowMajor>
      l_flow = (flow - p_flow).pruned(-10);
  for (i = j = 0; i < sub_nodeNum; i++)
  {
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(l_flow, i); it; ++it)
    {
      if (it.value() > 0)
      {
        l_out.coeffRef(j, 0) = (double)nodeIndex[sub_indices[i]];
        l_out.coeffRef(j, 1) = (double)nodeIndex[sub_indices[it.index()]];
        l_out.coeffRef(j, 2) = it.value();
        j++;
      }
    }
  }
  l_out.conservativeResize(j, Eigen::NoChange);

  string l_outfile = output_file + "_loop_flow.dat";
  //ofstream fp_out(output_file + p_outfile);
  FILE *lp;
  if ((lp = fopen(l_outfile.c_str(), "w")) == NULL)
  {
    cout << "output file error" << endl;
    exit(1);
  }
  temp_mt.resize(l_out.rows(), 2);
  temp_mt.col(0) = l_out.col(0).cast<int>();
  temp_mt.col(1) = l_out.col(1).cast<int>();
  for (i = 0; i < l_out.rows(); i++)
  {
    fprintf(lp, "%d\t%d\t%.10f\n", temp_mt.coeffRef(i, 0), temp_mt.coeffRef(i, 1), l_out.coeffRef(i, 2));
  }
  //ofstream fl_out(output_file + l_outfile);
  //fl_out.precision(10);
  //fl_out << fixed << setprecision(10) << l_out << endl;
  cout << "output \"" << l_outfile << "\"" << endl;
  //cout << "output \"" << output_file + l_outfile << "\"" << endl;

  double gr, lr, bal2eta;
  Eigen::SparseMatrix<double, Eigen::RowMajor> eta(nodeNum, nodeNum);
  eta = (-A + Eigen::SparseMatrix<double, Eigen::RowMajor>(A.diagonal().asDiagonal())).pruned();
  eta.makeCompressed();
  for (i = 0; i < eta.nonZeros(); i++)
  {
    *(eta.valuePtr() + i) = 1 / *(eta.valuePtr() + i);
  }
  bal2eta = flow.cwiseProduct(flow).cwiseProduct(eta).sum();
  gr = p_flow.cwiseProduct(p_flow).cwiseProduct(eta).sum() / bal2eta;
  lr = l_flow.cwiseProduct(l_flow).cwiseProduct(eta).sum() / bal2eta;

  string f_log("_log.dat");
  ofstream of_log(output_file + f_log);
  of_log << "Input file name: " << argv[1] << endl;
  of_log << "Input file: " << in.rows() << "x" << in.cols() << endl;
  of_log << "Input graph: " << nodeNum << " nodes, " << adj.nonZeros() << " links" << endl;
  of_log << "Potential flow ratio: " << fixed << std::setprecision(10) << gr << endl;
  of_log << "Loop flow ratio: " << fixed << std::setprecision(10) << lr << endl;
  cout << "output \"" << output_file + f_log << "\"" << endl;
  return 0;
}

Eigen::MatrixXd readTable(ifstream &fin)
{
  ios_base::sync_with_stdio(false);
  vector<double> el;
  el.reserve(1e6);
  char *buf = new char[BUFSIZE];
  char temp_c[64];
  temp_c[63] = '\0';
  int bi = 0;
  int ncol = 0;
  int temp = 0;
  int flag = 0;
  while (int gcount = fin.read(buf, BUFSIZE).gcount())
  {
    for (int i = 0; i < gcount; ++i)
    {
      switch (buf[i])
      {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case '+':
      case '-':
      case '.':
      case 'e':
      case 'E':
        temp_c[bi] = buf[i];
        ++bi;
        break;
      case '\n':
        if (bi != 0)
        {
          temp_c[bi] = '\0';
          el.push_back(atof(temp_c));
          bi = 0;
          ++temp;
        }
        if (temp != ncol)
        {
          if (flag)
          {
            cout << "number of cols is not constant: " << el.size() / ncol << "line" << endl;
            exit(0);
          }
          ++flag;
        }
        ncol = temp;
        temp = 0;
        break;
      default:
        if (bi != 0)
        {
          temp_c[bi] = '\0';
          el.push_back(atof(temp_c));
          bi = 0;
          ++temp;
        }
      }
    }
  }
  if (bi != 0)
  {
    temp_c[bi] = '\0';
    el.push_back(atof(temp_c));
    bi = 0;
    ++temp;
    if (temp != ncol)
    {
      if (flag)
      {
        cout << "number of cols is not constant: " << el.size() / ncol << "line" << endl;
        exit(0);
      }
      ++flag;
    }
  }
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> /**/> m(el.data(), el.size() / ncol, ncol);
  return m;
}

Eigen::VectorXi wccDecompose(Eigen::SparseMatrix<double, Eigen::RowMajor> &adj)
{
  int nrow = adj.rows();
  queue<int> search_n;
  vector<int> cl_size, temp_v;
  Eigen::VectorXi wcc_index = Eigen::VectorXi::Zero(nrow);
  Eigen::SparseMatrix<double, Eigen::RowMajor> A = adj.cwiseAbs() + Eigen::SparseMatrix<double, Eigen::RowMajor>(adj.transpose().cwiseAbs());
  int i = 0, j = 1, count = 0;
  while ((wcc_index.array() == 0).any())
  {
    while (wcc_index.coeff(i++) != 0)
      ;
    wcc_index.coeffRef(i - 1) = j;
    search_n.push(i - 1);
    count = 1;
    while (!search_n.empty())
    {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, search_n.front()); it; ++it)
      {
        if (wcc_index.coeff(it.index()) == 0)
        {
          search_n.push(it.index());
          wcc_index.coeffRef(it.index()) = j;
          count++;
        }
      }
      search_n.pop();
    }
    cl_size.push_back(count);
    j++;
  }
  temp_v = cl_size;
  sort(temp_v.begin(), temp_v.end(), std::greater<int>());
  for (i = 0; i < temp_v.size(); i++)
  {
    for (j = 0; j < cl_size.size(); j++)
    {
      if (temp_v[i] == cl_size[j])
      {
        cl_size[j] = -i;
        break;
      }
    }
  }
  for (j = 0; j < cl_size.size(); j++)
  {
    for (i = 0; i < wcc_index.size(); i++)
    {
      if (wcc_index.coeff(i) == j + 1)
      {
        wcc_index.coeffRef(i) = -cl_size[j];
      }
    }
  }
  return (wcc_index);
}
