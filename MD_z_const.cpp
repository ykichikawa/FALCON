//-isystem Eigen/ -std=c++11 --march=native
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <omp.h>
#include <queue>
#include <random>
using namespace std;
#define Kbt_init 1000
#define dt_init 0.0001
//#define Theta2 2.309401
#define Theta2 1.333333
#define kij_init 30.0
#define g_cent 1e-16
#define BUFSIZE 1000000
//#define Eps_inv_init 2e3 //1 / 4pi / epsilon_0
#define Eps_inv_init 1.0 //1 / 4pi / epsilon_0
#pragma omp declare reduction(matplus0          \
                              : Eigen::MatrixXd \
                              : omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
struct Node
{
    double qsum;
    double l2;
    int inc_begin;
    int inc_end;
    int num_leaf;
    int num_child;
    int leaf[8];
    int child[8];
    Eigen::Vector3d com;
    Eigen::Vector3d lf;
    Eigen::Vector3d rr;
};

Eigen::MatrixXd readTable(ifstream &fin);
inline int calc_node(vector<struct Node> &node, vector<pair<int, int> /**/> &ppart, Eigen::MatrixXd &r);

int main(int argc, char *argv[])
{
    int num_thread = 1;
    random_device seed_gen;
    mt19937 mt(seed_gen());
    normal_distribution<> nd(0, 1);
#ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
    num_thread = omp_get_max_threads();
#endif

    if (argc < 3)
    {
        cout << "<program> <edgelist> <initial coordinate>" << endl;
        return 1;
    }
    ifstream fin_el(argv[1]);
    if (!fin_el)
    {
        cout << "can not open edgelist file" << endl;
        return 1;
    }
    ifstream fin_ic(argv[2]);
    if (!fin_ic)
    {
        cout << "can not open coordinate file" << endl;
        return 1;
    }
    Eigen::MatrixXd el = readTable(fin_el);
    Eigen::MatrixXd ic = readTable(fin_ic);
    cout << "edgelist: " << el.rows() << "x" << el.cols() << endl;
    cout << "coordinate: " << ic.rows() << "x" << ic.cols() << endl;
    int num_part = ic.rows();
    int cols = el.cols();
    vector<pair<int, int> /**/> id_order;
    id_order.reserve(num_part);
    id_order.clear();
    for (int i = 0; i < num_part; ++i)
    {
        id_order.push_back(make_pair(ic.coeff(i, 0), i));
    }
    sort(id_order.begin(), id_order.end());
    Eigen::MatrixXd r(3, num_part);
    Eigen::MatrixXd r0(3, num_part);
    Eigen::MatrixXd v(3, num_part);

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(3, num_part);
    Eigen::MatrixXd F_k = Eigen::MatrixXd::Zero(3, num_part);
    Eigen::MatrixXd F_q = Eigen::MatrixXd::Zero(3, num_part);
    Eigen::VectorXd id;
    id.resize(num_part);
    for (int i = 0; i < num_part; ++i)
    {
        r.col(i) = ic.row(id_order[i].second).segment(1, 3).transpose();
        id.coeffRef(i) = id_order[i].first;
    }
    r0 = r;
    Eigen::MatrixXd edge(3, el.rows());
    if (el.cols() > 2)
    {
        for (int i = 0, k = el.rows(); i < k; ++i)
        {
            edge.coeffRef(0, i) = lower_bound(id_order.begin(), id_order.end(), el.coeff(i, 0), [](pair<int, int> &a, double b) { return a.first < int(b); }) - id_order.begin();
            edge.coeffRef(1, i) = lower_bound(id_order.begin(), id_order.end(), el.coeff(i, 1), [](pair<int, int> &a, double b) { return a.first < int(b); }) - id_order.begin();
            edge.coeffRef(2, i) = el.coeff(i, 2);
        }
    }
    else
    {
        for (int i = 0, k = el.rows(); i < k; ++i)
        {
            edge.coeffRef(0, i) = lower_bound(id_order.begin(), id_order.end(), el.coeff(i, 0), [](pair<int, int> &a, double b) { return a.first < int(b); }) - id_order.begin();
            edge.coeffRef(1, i) = lower_bound(id_order.begin(), id_order.end(), el.coeff(i, 1), [](pair<int, int> &a, double b) { return a.first < int(b); }) - id_order.begin();
            edge.coeffRef(2, i) = 1;
        }
    }
    int edge_num = edge.cols();
    double range = (r.rowwise().maxCoeff() - r.rowwise().minCoeff()).maxCoeff();
    double kij = kij_init;
    //  *num_part *sqrt(range) / edge_num;
    double eps_inv = Eps_inv_init * range * range * range / num_part;
    cout << "eps: " << eps_inv << ", k: " << kij << ", numThread: " << num_thread << endl;
    vector<struct Node>
        node;
    vector<struct Node> root8;
    node.reserve(num_part);
    queue<int> que;
    vector<pair<int, int> /**/> ppart;
    ppart.reserve(num_part);
    for (int i = 0; i < num_part; ++i)
    {
        ppart.push_back(make_pair(0, i));
    }
    //double kbt = Kbt_init;
    double kbt = range;
    struct Node root, newNode;
    for (auto it = v.data(), end = v.data() + 3 * num_part; it != end; ++it)
    {
        *it = nd(mt) * range / 10;
    }
    v.row(2).setZero();
    double dt = dt_init;
    double dt2 = dt * dt;
    double lim_dt = range / dt / dt / 2;
    double lim_dt2 = lim_dt * lim_dt;
#pragma omp parallel
    for (int ite = 0; ite < 200001; ++ite) /////////////////////////main loop ////////////////
    {
        /*#pragma omp single nowait
        {
            cout << "                                      \rite: " << ite << flush;
        }*/
#pragma omp sections nowait
        {
#pragma omp section //initialize
            {
                if (ite % 10 == 0)
                {
                    root.lf = r.rowwise().minCoeff();
                    root.rr = r.rowwise().maxCoeff();
                    Eigen::Vector3d border = Eigen::Vector3d::Zero();
                    double l_all = (root.rr - root.lf).array().maxCoeff();
                    root.l2 = l_all * l_all;
                    Eigen::VectorXi area = Eigen::VectorXi::Zero(num_part);
                    area = (r.row(0).array() >= 0).cast<int>();
                    area.array() += (r.row(1).array() >= 0).cast<int>() * 2;
                    area.array() += (r.row(2).array() >= 0).cast<int>() * 4;
                    for (int i = 0; i < num_part; ++i)
                    {
                        ppart[i].first = area.coeff(ppart[i].second);
                    }
                    sort(ppart.begin(), ppart.end());
                    root8.clear();
                    node.clear();
                    newNode.l2 = l_all * l_all * 0.25;
                    root.lf = border.array() - l_all;
                    root.rr = border.array() + l_all;
                    for (int i = 0; i < 8; ++i)
                    {
                        switch (i)
                        {
                        case 0:
                            newNode.lf = root.lf;
                            newNode.rr = border;
                            break;
                        case 1:
                            newNode.lf.coeffRef(0) = border.coeff(0);
                            newNode.lf.coeffRef(1) = root.lf.coeff(1);
                            newNode.lf.coeffRef(2) = root.lf.coeff(2);
                            newNode.rr.coeffRef(0) = root.rr.coeff(0);
                            newNode.rr.coeffRef(1) = border.coeff(1);
                            newNode.rr.coeffRef(2) = border.coeff(2);
                            break;
                        case 2:
                            newNode.lf.coeffRef(0) = root.lf.coeff(0);
                            newNode.lf.coeffRef(1) = border.coeff(1);
                            newNode.lf.coeffRef(2) = root.lf.coeff(2);
                            newNode.rr.coeffRef(0) = border.coeff(0);
                            newNode.rr.coeffRef(1) = root.rr.coeff(1);
                            newNode.rr.coeffRef(2) = border.coeff(2);
                            break;
                        case 3:
                            newNode.lf.coeffRef(0) = border.coeff(0);
                            newNode.lf.coeffRef(1) = border.coeff(1);
                            newNode.lf.coeffRef(2) = root.lf.coeff(2);
                            newNode.rr.coeffRef(0) = root.rr.coeff(0);
                            newNode.rr.coeffRef(1) = root.rr.coeff(1);
                            newNode.rr.coeffRef(2) = border.coeff(2);
                            break;
                        case 4:
                            newNode.lf.coeffRef(0) = root.lf.coeff(0);
                            newNode.lf.coeffRef(1) = root.lf.coeff(1);
                            newNode.lf.coeffRef(2) = border.coeff(2);
                            newNode.rr.coeffRef(0) = border.coeff(0);
                            newNode.rr.coeffRef(1) = border.coeff(1);
                            newNode.rr.coeffRef(2) = root.rr.coeff(2);
                            break;
                        case 5:
                            newNode.lf.coeffRef(0) = border.coeff(0);
                            newNode.lf.coeffRef(1) = root.lf.coeff(1);
                            newNode.lf.coeffRef(2) = border.coeff(2);
                            newNode.rr.coeffRef(0) = root.rr.coeff(0);
                            newNode.rr.coeffRef(1) = border.coeff(1);
                            newNode.rr.coeffRef(2) = root.rr.coeff(2);
                            break;
                        case 6:
                            newNode.lf.coeffRef(0) = root.lf.coeff(0);
                            newNode.lf.coeffRef(1) = border.coeff(1);
                            newNode.lf.coeffRef(2) = border.coeff(2);
                            newNode.rr.coeffRef(0) = border.coeff(0);
                            newNode.rr.coeffRef(1) = root.rr.coeff(1);
                            newNode.rr.coeffRef(2) = root.rr.coeff(2);
                            break;
                        case 7:
                            newNode.lf = border;
                            newNode.rr = root.rr;
                        }
                        newNode.inc_begin = lower_bound(ppart.begin(), ppart.end(), i, [](pair<int, int> &a, int b) { return a.first < b; }) - ppart.begin();
                        newNode.inc_end = lower_bound(ppart.begin(), ppart.end(), i + 1, [](pair<int, int> &a, int b) { return a.first < b; }) - ppart.begin();
                        root8.push_back(newNode);
                        node.clear();
                    }
                }
                queue<int>().swap(que);
            }

#pragma omp section //update Tmperature
            {
                if (ite % 1000 == 0)
                {
                    double v2sum = v.colwise().squaredNorm().mean() / 2;
#pragma omp critical(cout)
                    {
                        cout << "ite: " << ite << ", vmean*dt: " << sqrt(v2sum) * dt << ", dRMS: " << F.colwise().squaredNorm().mean() << ", D: " << (r - r0).colwise().squaredNorm().sum() << endl;
                        r0 = r;
                    }
                }
            }
        }

#pragma omp for num_threads(num_thread - 1) reduction(matplus0 \
                                                      : F_k) nowait //elasitc force
        for (int i = 0; i < edge_num; ++i)
        {
            int orig = edge.coeff(0, i);
            int dest = edge.coeff(1, i);
            Eigen::Vector3d &&rij = kij * edge.coeff(2, i) * (r.col(orig) - r.col(dest));
            F_k.col(orig) -= rij;
            F_k.col(dest) += rij;
        }

#pragma omp for firstprivate(ppart)
        for (int i = 0; i < 8; ++i)
        {
            if (ite % 10 == 0)
            {
                vector<struct Node> temp;
                temp.reserve(num_part);
                temp.clear();
                temp.push_back(root8[i]);
                calc_node(temp, ppart, r);
                for (auto it = temp.rbegin(), end = temp.rend(); it != end; ++it)
                {
                    it->qsum = it->num_leaf;
                    it->com.setZero();
                    for (int j = 0; j < it->num_leaf; ++j)
                    {
                        it->com += r.col(it->leaf[j]);
                    }
                    for (int j = 0; j < it->num_child; ++j)
                    {
                        it->qsum += (it - it->child[j])->qsum;
                        it->com += (it - it->child[j])->qsum * (it - it->child[j])->com;
                    }
                    it->com /= it->qsum;
                }
#pragma omp critical(linking)
                {
                    que.push(node.size());
                    node.insert(node.end(), temp.begin(), temp.end());
                }
            }
        }

#pragma omp for reduction(matplus0 \
                          : F_q) // qoulomb force
        for (int i = 0; i < num_part; ++i)
        {
            queue<int> search_n = que;
            Eigen::Vector3d r_i = r.col(i);
            Eigen::Vector3d F_i = -g_cent * r_i / r_i.norm();
            while (!search_n.empty())
            {
                int foc = search_n.front();
                struct Node fnode = node[foc];
                if (fnode.l2 > (fnode.com - r_i).squaredNorm() * Theta2)
                {
                    for (int j = 0; j < fnode.num_leaf; ++j)
                    {
                        if (fnode.leaf[j] != i)
                        {
                            Eigen::Vector3d rij = r_i - r.col(fnode.leaf[j]);
                            double d_inv = 1 / rij.norm();
                            F_i += eps_inv * rij * d_inv * d_inv * d_inv;
                        }
                    }
                    for (int j = 0; j < fnode.num_child; ++j)
                    {
                        search_n.push(fnode.child[j] + foc);
                    }
                }
                else
                {
                    Eigen::Vector3d rij = r_i - fnode.com;
                    double d_inv = 1 / rij.norm();
                    F_i += eps_inv * fnode.qsum * rij * d_inv * d_inv * d_inv;
                }
                search_n.pop();
            }
            F_q.col(i) += F_i;
        }

#pragma omp single
        {
            F = F_k + F_q;
            v.topRows(2) += (0.5 * dt) * F.topRows(2);
            r.topRows(2) += v.topRows(2) * dt + F.topRows(2) * (dt2 * 0.5);
            v.topRows(2) *= 0.999;
            v.topRows(2) += (0.5 * dt) * F.topRows(2);
            F_k.setZero();
            F_q.setZero();
            if (ite % 10000 == 0)
            {
                Eigen::MatrixXd co_res(num_part, 4);
                co_res.col(0) = id;
                co_res.block(0, 1, num_part, 3) = r.transpose();
                ofstream fout("ss" + to_string(ite) + ".dat");
                fout << co_res;
            }
        }
    }
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
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> /**/> m(el.data(), el.size() / ncol, ncol);
    return m;
}

inline int calc_node(vector<struct Node> &node, vector<pair<int, int> /**/> &ppart, Eigen::MatrixXd &r)
{
    Eigen::Vector3d border;
    Eigen::MatrixXd temp(3, node[0].inc_end - node[0].inc_begin);
    int test = 0;
    for (int i = 0; i < node.size(); ++i)
    {

        int count[8];
        for (int j = 0; j < 8; ++j)
        {
            count[j] = 0;
        }
        struct Node fnode = node[i];
        vector<pair<int, int> /**/>::iterator beg = ppart.begin() + fnode.inc_begin;
        vector<pair<int, int> /**/>::iterator end = ppart.begin() + fnode.inc_end;
        border = (fnode.rr + fnode.lf) / 2;
        for (auto it = beg; it != end; ++it)
        {
            Eigen::Vector3d r_temp = r.col(it->second);
            int pos_box = 0;
            if (r_temp.coeff(0) >= border.coeff(0))
            {
                pos_box = 1;
            }
            if (r_temp.coeff(1) >= border.coeff(1))
            {
                pos_box += 2;
            }
            if (r_temp.coeff(2) >= border.coeff(2))
            {
                pos_box += 4;
            }
            it->first = pos_box;
            ++count[pos_box];
        }
        /*if (i == 1000)
        {
            cout << "[" << count[0] << " " << count[1] << " " << count[2] << " " << count[3] << " " << count[4] << " " << count[5] << " " << count[6] << " " << count[7] << "]: " << Eigen::Map<Eigen::VectorXi>(count, 8).array().sum() << " " << end - beg << endl;
            cout << r.col(beg - ppart.begin()).transpose() << endl;
            cout << r.col(end - 1 - ppart.begin()).transpose() << endl;
            cout << border.transpose() << endl;
        }*/
        sort(beg, end);
        fnode.num_leaf = 0;
        fnode.num_child = 0;
        struct Node newNode;
        for (int j = 0; j < 8; ++j)
        {
            if (count[j] == 1)
            {
                fnode.leaf[fnode.num_leaf] = lower_bound(beg, end, j, [](pair<int, int> &a, int b) { return a.first < b; })->second;
                ++fnode.num_leaf;
            }
            else if (count[j] > 1)
            {
                switch (j)
                {
                case 0:
                    newNode.lf = fnode.lf;
                    newNode.rr = border;
                    break;
                case 1:
                    newNode.lf.coeffRef(0) = border.coeff(0);
                    newNode.lf.coeffRef(1) = fnode.lf.coeff(1);
                    newNode.lf.coeffRef(2) = fnode.lf.coeff(2);
                    newNode.rr.coeffRef(0) = fnode.rr.coeff(0);
                    newNode.rr.coeffRef(1) = border.coeff(1);
                    newNode.rr.coeffRef(2) = border.coeff(2);
                    break;
                case 2:
                    newNode.lf.coeffRef(0) = fnode.lf.coeff(0);
                    newNode.lf.coeffRef(1) = border.coeff(1);
                    newNode.lf.coeffRef(2) = fnode.lf.coeff(2);
                    newNode.rr.coeffRef(0) = border.coeff(0);
                    newNode.rr.coeffRef(1) = fnode.rr.coeff(1);
                    newNode.rr.coeffRef(2) = border.coeff(2);
                    break;
                case 3:
                    newNode.lf.coeffRef(0) = border.coeff(0);
                    newNode.lf.coeffRef(1) = border.coeff(1);
                    newNode.lf.coeffRef(2) = fnode.lf.coeff(2);
                    newNode.rr.coeffRef(0) = fnode.rr.coeff(0);
                    newNode.rr.coeffRef(1) = fnode.rr.coeff(1);
                    newNode.rr.coeffRef(2) = border.coeff(2);
                    break;
                case 4:
                    newNode.lf.coeffRef(0) = fnode.lf.coeff(0);
                    newNode.lf.coeffRef(1) = fnode.lf.coeff(1);
                    newNode.lf.coeffRef(2) = border.coeff(2);
                    newNode.rr.coeffRef(0) = border.coeff(0);
                    newNode.rr.coeffRef(1) = border.coeff(1);
                    newNode.rr.coeffRef(2) = fnode.rr.coeff(2);
                    break;
                case 5:
                    newNode.lf.coeffRef(0) = border.coeff(0);
                    newNode.lf.coeffRef(1) = fnode.lf.coeff(1);
                    newNode.lf.coeffRef(2) = border.coeff(2);
                    newNode.rr.coeffRef(0) = fnode.rr.coeff(0);
                    newNode.rr.coeffRef(1) = border.coeff(1);
                    newNode.rr.coeffRef(2) = fnode.rr.coeff(2);
                    break;
                case 6:
                    newNode.lf.coeffRef(0) = fnode.lf.coeff(0);
                    newNode.lf.coeffRef(1) = border.coeff(1);
                    newNode.lf.coeffRef(2) = border.coeff(2);
                    newNode.rr.coeffRef(0) = border.coeff(0);
                    newNode.rr.coeffRef(1) = fnode.rr.coeff(1);
                    newNode.rr.coeffRef(2) = fnode.rr.coeff(2);
                    break;
                case 7:
                    newNode.lf = border;
                    newNode.rr = fnode.rr;
                }
                newNode.l2 = (newNode.rr.coeff(0) - newNode.lf.coeff(0));
                newNode.l2 *= newNode.l2;
                newNode.inc_begin = lower_bound(beg, end, j, [](pair<int, int> &a, int b) { return a.first < b; }) - ppart.begin();
                newNode.inc_end = lower_bound(beg, end, j + 1, [](pair<int, int> &a, int b) { return a.first < b; }) - ppart.begin();
                node.push_back(newNode);
                fnode.child[fnode.num_child] = node.size() - i - 1;
                ++fnode.num_child;
            }
        }
        if (fnode.num_child == 1 && fnode.num_leaf == 0)
        {
            int n = fnode.inc_end - fnode.inc_begin;
            for (int j = 0; j < n; ++j)
            {
                temp.col(j) = r.col(ppart[fnode.inc_begin + j].second);
            }
            newNode.lf = temp.leftCols(n).rowwise().minCoeff();
            newNode.rr = temp.leftCols(n).rowwise().maxCoeff();
            border = (newNode.rr + newNode.lf) / 2;
            double l = (newNode.rr - newNode.lf).array().maxCoeff();
            newNode.l2 = l * l;
            newNode.lf = border.array() - l;
            newNode.rr = border.array() + l;
            node[i] = newNode;
            --i;
            node.pop_back();
        }
        else
        {
            node[i] = fnode;
        }
    }
    return 0;
}