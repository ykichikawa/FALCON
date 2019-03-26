//-isystem Eigen/ -std=c++11 -ftemplate-depth=8192
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <omp.h>
#include <random>
#include <time.h>
using namespace std;

Eigen::MatrixXd readTable(ifstream &stream);
Eigen::Vector3d max_space(struct Particle *begin, struct Particle *end);
Eigen::Vector3d min_space(struct Particle *begin, struct Particle *end);
int createChildNode(vector<struct Node> *pnode, struct Node &my);
struct Node *createNode(vector<struct Node> *pnode, Eigen::Vector3d lf, Eigen::Vector3d rr, pair<int, Particle *> *ptr_begin, pair<int, Particle *> *ptr_end);
int calcNode(struct Node &my);

struct Particle
{
    double m;
    double q;
    Eigen::Vector3d r;
    Eigen::Vector3d v;
    Eigen::Vector3d F;
};

struct Node
{
    double l;
    double qsum;
    int count[8];
    Node *child[8];
    pair<int, Particle *> *ptr_sep[9];
    Eigen::Vector3d com;
    Eigen::Vector3d lf; // [
    Eigen::Vector3d rr; // )
};

struct Link
{
    Particle *orig;
    Particle *dest;
    double w;
};

int main(int argc, char *argv[])
{
    ios_base::sync_with_stdio(false);
    std::mt19937 mt((int)time(NULL));

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

    int npart = ic.rows(), nlink = el.rows();

    struct Particle part[npart];
    for (int i = 0; i < npart; i++)
    {
        part[i].r.coeffRef(0) = ic.coeff(i, 1);
        part[i].r.coeffRef(1) = ic.coeff(i, 2);
        part[i].r.coeffRef(2) = ic.coeff(i, 3);
        part[i].m = 1;
        part[i].q = 1;
    }

    struct Link links[nlink];
    vector<int> id(ic.data(), ic.data() + npart);
    for (int i = 0; i < nlink; i++)
    {
        links[i].orig = part + (find(id.begin(), id.end(), el.coeff(i, 0)) - id.begin());
        links[i].dest = part + (find(id.begin(), id.end(), el.coeff(i, 1)) - id.begin());
        links[i].w = el.coeff(i, 2);
    }

    vector<pair<int, Particle *> /**/> ppart;
    for (int i = 0; i < npart; i++)
    {
        ppart.push_back(make_pair(0, part + i));
    }
    vector<Node> emp(8);
    vector<vector<Node>> pntop(8);
    for (int i = 0; i < 8; i++)
    {
        pntop[i].reserve(npart);
    }
    //////////////////////////////////////////////////////////////////////////
    struct Node root;
    root.lf = min_space(part, part + npart);
    root.rr = max_space(part, part + npart);
    Eigen::Vector3d border = (root.lf + root.rr) / 2;
    root.l = (root.rr - root.lf).maxCoeff();
    root.lf = (border.array() - root.l / 2).matrix();
    root.rr = (border.array() + root.l / 2).matrix();
    root.ptr_sep[0] = ppart.data();
    root.ptr_sep[8] = ppart.data() + ppart.size();
    emp.clear();
    createChildNode(&emp, root);
    for (int i = 0; i < 8; i++)
    {
        pntop[i].clear();
        pntop[i].push_back(emp[i]);
    }
    for (int i = 0; i < 8; i++)
    { // pallarel
        for (int j = 0; j < pntop[i].size(); j++)
        {
            createChildNode(&pntop[i], pntop[i][j]);
        }
        for (auto it = pntop[i].rbegin(); it != pntop[i].rend(); ++it)
        {
            calcNode(*it);
        }
    }

    /*for (int i = 0; i < 8; i++) ///////////////////////////////////////////////////
    {
        cout << "pntop[" << i << "]: " << pntop[i].size() << endl;
    }*/

    return 0;
}

int createChildNode(vector<struct Node> *pnode, struct Node &my)
{
    //cout << "*" << flush;
    Eigen::Vector3d border = (my.lf + my.rr) / 2, temp_lf, temp_rr;
    for (auto it = my.ptr_sep[0]; it != my.ptr_sep[8]; ++it)
    {
        it->first = 0;
        if (it->second->r.coeff(0) >= border.coeff(0))
        {
            it->first += 1;
        }
        if (it->second->r.coeff(1) >= border.coeff(1))
        {
            it->first += 2;
        }
        if (it->second->r.coeff(2) >= border.coeff(2))
        {
            it->first += 4;
        }
    }
    sort(my.ptr_sep[0], my.ptr_sep[8]);
    for (int i = 0; i < 8; i++)
    {
        my.count[i] = 0;
    }
    for (auto it = my.ptr_sep[0]; it != my.ptr_sep[8]; it++)
    {
        my.count[it->first] += 1;
    }
    for (int i = 0; i < 7; i++)
    {
        my.ptr_sep[i + 1] = my.ptr_sep[i] + my.count[i];
    }
    for (int i = 0; i < 8; i++)
    {
        if (my.count[i] > 1)
        {
            switch (i)
            {
            case 0:
                temp_lf = my.lf;
                temp_rr = border;
                break;
            case 1:
                temp_lf.coeffRef(0) = border.coeff(0);
                temp_lf.coeffRef(1) = my.lf.coeff(1);
                temp_lf.coeffRef(2) = my.lf.coeff(2);
                temp_rr.coeffRef(0) = my.rr.coeff(0);
                temp_rr.coeffRef(1) = border.coeff(1);
                temp_rr.coeffRef(2) = border.coeff(2);
                break;
            case 2:
                temp_lf.coeffRef(0) = my.lf.coeff(0);
                temp_lf.coeffRef(1) = border.coeff(1);
                temp_lf.coeffRef(2) = my.lf.coeff(2);
                temp_rr.coeffRef(0) = border.coeff(0);
                temp_rr.coeffRef(1) = my.rr.coeff(1);
                temp_rr.coeffRef(2) = border.coeff(2);
                break;
            case 3:
                temp_lf.coeffRef(0) = border.coeff(0);
                temp_lf.coeffRef(1) = border.coeff(1);
                temp_lf.coeffRef(2) = my.lf.coeff(2);
                temp_rr.coeffRef(0) = my.rr.coeff(0);
                temp_rr.coeffRef(1) = my.rr.coeff(1);
                temp_rr.coeffRef(2) = border.coeff(2);
                break;
            case 4:
                temp_lf.coeffRef(0) = my.lf.coeff(0);
                temp_lf.coeffRef(1) = my.lf.coeff(1);
                temp_lf.coeffRef(2) = border.coeff(2);
                temp_rr.coeffRef(0) = border.coeff(0);
                temp_rr.coeffRef(1) = border.coeff(1);
                temp_rr.coeffRef(2) = my.rr.coeff(2);
                break;
            case 5:
                temp_lf.coeffRef(0) = border.coeff(0);
                temp_lf.coeffRef(1) = my.lf.coeff(1);
                temp_lf.coeffRef(2) = border.coeff(2);
                temp_rr.coeffRef(0) = my.rr.coeff(0);
                temp_rr.coeffRef(1) = border.coeff(1);
                temp_rr.coeffRef(2) = my.rr.coeff(2);
                break;
            case 6:
                temp_lf.coeffRef(0) = my.lf.coeff(0);
                temp_lf.coeffRef(1) = border.coeff(1);
                temp_lf.coeffRef(2) = border.coeff(2);
                temp_rr.coeffRef(0) = border.coeff(0);
                temp_rr.coeffRef(1) = my.rr.coeff(1);
                temp_rr.coeffRef(2) = my.rr.coeff(2);
                break;
            case 7:
                temp_lf = border;
                temp_rr = my.rr;
            }
            my.child[i] = createNode(pnode, temp_lf, temp_rr, my.ptr_sep[i], my.ptr_sep[i + 1]);
        }
        else
        {
            my.child[i] = NULL;
            //cout << "-";
        }
    }
    //cout << "/" << endl;
    return 0;
}

struct Node *createNode(vector<struct Node> *pnode, Eigen::Vector3d lf, Eigen::Vector3d rr, pair<int, Particle *> *ptr_begin, pair<int, Particle *> *ptr_end)
{
    //cout << "+" << flush;
    struct Node my;
    my.lf = lf;
    my.rr = rr;
    my.ptr_sep[0] = ptr_begin;
    my.ptr_sep[8] = ptr_end;
    my.l = rr.coeff(0) - lf.coeff(0);
    pnode->push_back(my);
    return &pnode->back();
}

int calcNode(struct Node &my)
{
    my.qsum = 0;
    my.com.setZero();
    for (int i = 0; i < 8; i++)
    {
        if (my.count[i] == 0)
        {
        }
        else if (my.count[i] == 1)
        {
            my.qsum += my.ptr_sep[i]->second->q;
            my.com += my.ptr_sep[i]->second->r * my.ptr_sep[i]->second->q;
        }
        else if (my.count[i] > 1)
        {
            my.qsum += my.child[i]->qsum;
            my.com += my.child[i]->com * my.child[i]->qsum;
        }
    }
    my.com = my.com / my.qsum;
    //cout << "qsum: " << my.qsum << " com: " << my.com(0) << " " << my.com(1) << " " << my.com(2) << endl;
    return 0;
}
Eigen::Vector3d max_space(struct Particle *begin, struct Particle *end)
{
    double x[3];
    x[0] = max_element(begin, end, [](struct Particle a, struct Particle b) { return a.r.coeff(0) < b.r.coeff(0); })->r.coeff(0);
    x[1] = max_element(begin, end, [](struct Particle a, struct Particle b) { return a.r.coeff(1) < b.r.coeff(1); })->r.coeff(1);
    x[2] = max_element(begin, end, [](struct Particle a, struct Particle b) { return a.r.coeff(2) < b.r.coeff(2); })->r.coeff(2);
    Eigen::Vector3d spacerr(x[0], x[1], x[2]);
    return spacerr;
}

Eigen::Vector3d min_space(struct Particle *begin, struct Particle *end)
{
    double x[3];
    x[0] = min_element(begin, end, [](struct Particle a, struct Particle b) { return a.r.coeff(0) < b.r.coeff(0); })->r.coeff(0);
    x[1] = min_element(begin, end, [](struct Particle a, struct Particle b) { return a.r.coeff(1) < b.r.coeff(1); })->r.coeff(1);
    x[2] = min_element(begin, end, [](struct Particle a, struct Particle b) { return a.r.coeff(2) < b.r.coeff(2); })->r.coeff(2);
    Eigen::Vector3d spacelf(x[0], x[1], x[2]);
    return spacelf;
}

Eigen::MatrixXd readTable(ifstream &stream)
{
    ios_base::sync_with_stdio(false);
    int rows, i, cols;
    double temp;
    std::vector<double> vec;
    string str;

    //getline(stream, str);
    cols = 0;
    while (!stream.eof())
    {
        stream >> temp;
        vec.push_back(temp);
        cols++;
        if (stream.peek() == '\n')
        {
            break;
        }
    }

    if (cols == 0)
    {
        exit(1);
    }

    //stream.seekg(0, ios_base::beg);
    i = 0;
    while (!stream.eof())
    {
        stream >> temp;
        vec.push_back(temp);
        i++;
        if (stream.peek() == '\n')
        {
            if (i != cols)
            {
                cout << "Input file error" << endl;
                exit(1);
            }
            i = 0;
        }
    }

    std::vector<double>::iterator it = vec.begin();
    Eigen::MatrixXd mat((int)(vec.size() / cols), cols);
    for (i = 0; i < mat.rows(); i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mat.coeffRef(i, j) = *it;
            it++;
        }
    }
    return (mat);
}