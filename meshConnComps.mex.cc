///////////////////////////////////////////////////////////////////////////////
//
//  Name:        meshConnComps.mex.cc
//  Purpose:     Find connected components in triangle meshes.
//  Author:      Daeyun Shin <dshin11@illinois.edu>
//  Created:     03.17.2015
//  Modified:    03.18.2015
//  Version:     0.1
//
//  This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
///////////////////////////////////////////////////////////////////////////////

#include "mex.h"
#include "mexutil.h"

#include <iostream>
#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Example usage:
 *
 * mesh(1).f = ...;  mesh(1).v = ...;
 * mesh(2).f = ...;  mesh(2).v = ...;
 * connComps = meshConnComps(mesh);
 *
 * % output:
 * connComps(1).surfaceArea
 * connComps(1).isClosed
 * connComps(1).indices
 * connComps(2).surfaceArea
 * connComps(2).isClosed
 * connComps(2).indices
 */

namespace mesh_util {
namespace conn_comps {
using namespace std;

struct Face {
  int v[3];
  set<int> adj_faces;
};

struct Vertex {
  float coord[3];
  set<int> adj_faces;
};

void _find_conn_comps(const vector<Face> &faces, vector<int> &conn_comp) {
  conn_comp.resize(faces.size());

  fill(conn_comp.begin(), conn_comp.end(), -1);

  int comp_ind = 0;

  for (int face_ind = 0; face_ind < (int)faces.size(); face_ind++) {
    if (conn_comp[face_ind] > -1) {
      continue;
    }

    queue<int> q;
    q.push(face_ind);

    while (!q.empty()) {
      int f = q.front();
      q.pop();
      for (int neighbor_ind : faces[f].adj_faces) {
        if (conn_comp[neighbor_ind] == -1) {
          q.push(neighbor_ind);
          conn_comp[neighbor_ind] = comp_ind;
        }
      }
    }
    comp_ind++;
  }
}

void _is_closed(const vector<Face> &faces, const vector<int> &conn_comp,
                vector<int> &is_closed) {
  int max_comp = *max_element(conn_comp.begin(), conn_comp.end());
  is_closed.resize(max_comp + 1);
  fill(is_closed.begin(), is_closed.end(), true);

  for (int i = 0; i < (int)faces.size(); i++) {
    if (faces[i].adj_faces.size() != 3) {
      is_closed[conn_comp[i]] = false;
    }
  }
}

void _surface_area(const vector<Face> &faces, const vector<Vertex> &vertices,
                   const vector<int> &conn_comp, vector<double> &surface_area) {
  int max_comp = *max_element(conn_comp.begin(), conn_comp.end());
  surface_area.resize(max_comp + 1);
  fill(surface_area.begin(), surface_area.end(), 0);

  vector<double> triangle_area(faces.size());

#pragma omp parallel for
  for (int i = 0; i < (int)faces.size(); i++) {
    double ax =
        vertices[faces[i].v[1]].coord[0] - vertices[faces[i].v[0]].coord[0];
    double ay =
        vertices[faces[i].v[1]].coord[1] - vertices[faces[i].v[0]].coord[1];
    double az =
        vertices[faces[i].v[1]].coord[2] - vertices[faces[i].v[0]].coord[2];
    double bx =
        vertices[faces[i].v[2]].coord[0] - vertices[faces[i].v[0]].coord[0];
    double by =
        vertices[faces[i].v[2]].coord[1] - vertices[faces[i].v[0]].coord[1];
    double bz =
        vertices[faces[i].v[2]].coord[2] - vertices[faces[i].v[0]].coord[2];
    double cx = ay * bz - az * by;
    double cy = az * bx - ax * bz;
    double cz = ax * by - ay * bx;

    triangle_area[i] = 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
  }

  for (int i = 0; i < (int)faces.size(); i++) {
    surface_area[conn_comp[i]] += triangle_area[i];
  }
}

void init(const vector<int> &faces_in, const vector<double> &vertices_in,
          vector<Face> &faces, vector<Vertex> &vertices) {
  vertices.resize(vertices_in.size() / 3);
  faces.resize(faces_in.size() / 3);

  for (size_t i = 0; i < vertices_in.size(); i++) {
    vertices[i / 3].coord[i % 3] = vertices_in[i];
  }

  for (size_t i = 0; i < faces_in.size(); i++) {
    faces[i / 3].v[i % 3] = faces_in[i];
  }

  for (size_t i = 0; i < faces.size(); i++) {
    vertices[faces[i].v[0]].adj_faces.insert(i);
    vertices[faces[i].v[1]].adj_faces.insert(i);
    vertices[faces[i].v[2]].adj_faces.insert(i);
  }

  for (size_t i = 0; i < faces.size(); i++) {
    vertices[faces[i].v[0]].adj_faces.insert(i);
    vertices[faces[i].v[1]].adj_faces.insert(i);
    vertices[faces[i].v[2]].adj_faces.insert(i);

    set<int> adj;
    set_intersection(vertices[faces[i].v[0]].adj_faces.begin(),
                     vertices[faces[i].v[0]].adj_faces.end(),
                     vertices[faces[i].v[1]].adj_faces.begin(),
                     vertices[faces[i].v[1]].adj_faces.end(),
                     inserter(adj, adj.begin()));

    set_intersection(vertices[faces[i].v[1]].adj_faces.begin(),
                     vertices[faces[i].v[1]].adj_faces.end(),
                     vertices[faces[i].v[2]].adj_faces.begin(),
                     vertices[faces[i].v[2]].adj_faces.end(),
                     inserter(adj, adj.begin()));

    set_intersection(vertices[faces[i].v[0]].adj_faces.begin(),
                     vertices[faces[i].v[0]].adj_faces.end(),
                     vertices[faces[i].v[2]].adj_faces.begin(),
                     vertices[faces[i].v[2]].adj_faces.end(),
                     inserter(adj, adj.begin()));
    adj.erase(i);

    faces[i].adj_faces = adj;
  }
}

void find_conn_comps(const vector<int> &faces, const vector<double> &vertices,
                     vector<int> &conn_comps, vector<int> &is_closed,
                     vector<double> &surface_area) {
  assert(vertices.size() > 0);
  assert(vertices.size() % 3 == 0);
  assert(faces.size() > 0);
  assert(faces.size() % 3 == 0);

  vector<Vertex> v;
  vector<Face> f;

  init(faces, vertices, f, v);

  _find_conn_comps(f, conn_comps);
  _is_closed(f, conn_comps, is_closed);
  _surface_area(f, v, conn_comps, surface_area);
}
};
};

void mexFunction(int nargout, mxArray *out[], int nargin, const mxArray *in[]) {
  using namespace mexutil;

  N_IN(1);
  N_OUT(1);

  M_ASSERT(mxIsStruct(in[0]));

  size_t num_struct_rows = mxGetM(in[0]);
  size_t num_struct_cols = mxGetN(in[0]);

  M_ASSERT(num_struct_cols == 1 || num_struct_rows == 1);
  size_t num_structs =
      (num_struct_cols > num_struct_rows) ? num_struct_cols : num_struct_rows;

  std::vector<MatlabStruct> conn_comps(num_structs);

  std::vector<std::vector<int>> indices(num_structs);
  std::vector<std::vector<int>> is_closed(num_structs);
  std::vector<std::vector<double>> surface_area(num_structs);

#pragma omp parallel for
  for (size_t k = 0; k < num_structs; k++) {
    mxArray *v_field = mxGetField(in[0], k, "v");
    if (v_field == 0) {
      v_field = mxGetField(in[0], k, "vertices");
    }
    mxArray *f_field = mxGetField(in[0], k, "f");
    if (f_field == 0) {
      f_field = mxGetField(in[0], k, "faces");
    }
    if (v_field == 0 || f_field == 0) {
      ERR_EXIT("InvalidStruct", "Input should be struct('f', ..., 'v', ...)");
    }

    M_ASSERT(mxIsDouble(v_field));
    size_t num_v_rows = mxGetM(v_field);
    size_t num_v_cols = mxGetN(v_field);
    M_ASSERT(num_v_rows == 3, "size(mesh.v, 1) must be 3.");
    M_ASSERT(num_v_cols > 3, "size(mesh.v, 2) must be greater than 3.");
    double *vertices = (double *)mxGetPr(v_field);

    M_ASSERT(mxIsInt32(f_field));
    size_t num_f_rows = mxGetM(f_field);
    size_t num_f_cols = mxGetN(f_field);
    M_ASSERT(num_f_rows == 3, "size(mesh.f, 1) must be 3.");
    M_ASSERT(num_f_cols > 3, "size(mesh.f, 2) must be greater than 3.");
    int *faces = (int *)mxGetData(f_field);

    std::vector<double> v;
    v.assign(vertices, vertices + num_v_rows * num_v_cols);
    std::vector<int> f(faces, faces + num_f_rows * num_f_cols);
    for (int &val : f) {
      val--;
    }

    try {
      mesh_util::conn_comps::find_conn_comps(f, v, indices[k], is_closed[k],
                                             surface_area[k]);
    } catch (const std::exception &ex) {
      ERR_EXIT("Exception", ex.what());
    }

    for (int &val : indices[k]) {
      val++;
    }

    conn_comps[k] = {{
        {"indices", kInt32, {indices[k].size(), 1, 1}, &indices[k][0]},
        {"isClosed", kLogical, {is_closed[k].size(), 1, 1}, &is_closed[k][0]},
        {"surfaceArea",
         kDouble,
         {surface_area[k].size(), 1, 1},
         &surface_area[k][0]},
    }};
  }

  createStructArray(conn_comps, out[0]);
}
