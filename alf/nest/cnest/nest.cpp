// Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

template <typename... Args>
std::string py_format(const std::string& format, Args... args) {
  // Use python's string format
  static py::object py_format = py::module::import("format").attr("format");
  return py::cast<py::str>(py_format(format, args...));
}

py::int_ GetPyInt(unsigned int i) {
  // cache py::int_ to avoid creating new objects for repeating i
  static std::vector<py::int_> table;
  if (table.size() <= i) {
    for (unsigned int j = table.size(); j <= i; j++) {
      table.push_back(py::int_(j));
    }
  }
  return table[i];
}

bool IsNested(const py::object& value) {
  return (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value) ||
          py::isinstance<py::dict>(value));
}

bool IsNamedtuple(const py::object& value) {
  return (py::isinstance<py::tuple>(value) && py::hasattr(value, "_fields"));
}

bool IsUnnamedtuple(const py::object& value) {
  return (py::isinstance<py::tuple>(value) && !py::hasattr(value, "_fields"));
}

void AssertSameType(const py::object& value1, const py::object& value2) {
  bool type_equal = value1.get_type().equal(value2.get_type());
  bool both_dict =
      (py::isinstance<py::dict>(value1) && py::isinstance<py::dict>(value2));
  if (!(type_equal || both_dict)) {
    throw std::runtime_error(
        py_format("Different types! {} <-> {}", value1, value2));
  }
}

void AssertSameLength(const py::object& seq1, const py::object& seq2) {
  bool both_iterable = (py::isinstance<py::iterable>(seq1) &&
                        py::isinstance<py::iterable>(seq2));
  if (!both_iterable) {
    throw std::runtime_error(py_format(
        "The arguments should be iterable! seq1: {}, seq2: {}", seq1, seq2));
  }
  if (py::len(seq1) != py::len(seq2)) {
    throw std::runtime_error(
        py_format("Different lengths! {} <-> {}", seq1, seq2));
  }
}

typedef std::vector<std::pair<std::string, py::object>> field_list;

field_list ExtractFieldsFromNest(const py::object& nest) {
  bool is_dict = py::isinstance<py::dict>(nest);
  if (!(is_dict || IsNamedtuple(nest))) {
    throw std::runtime_error(
        py_format("Nest {} must be a dict or namedtuple!", nest));
  }

  field_list ret;
  if (is_dict) {
    auto nest_dict = py::cast<py::dict>(nest);
    for (const auto& item : nest_dict) {
      if (!py::isinstance<py::str>(item.first)) {
        throw std::runtime_error(py_format(
            "Only support string keys in a dictionary!! Wrong key: {}",
            item.first));
      }
      ret.push_back(std::make_pair(py::cast<py::str>(item.first),
                                   py::cast<py::object>(item.second)));
    }
  } else {
    for (const auto& field : nest.attr("_fields")) {
      std::string key = py::cast<py::str>(field);
      ret.push_back(
          std::make_pair(key, py::cast<py::object>(nest.attr(key.c_str()))));
    }
  }
  std::sort(ret.begin(),
            ret.end(),
            [](const std::pair<std::string, py::object>& p1,
               const std::pair<std::string, py::object>& p2) {
              return p1.first < p2.first;
            });
  return ret;
}

std::vector<py::object> Flatten(const py::object& nest) {
  std::vector<py::object> flat;
  if (!IsNested(nest)) {
    flat.push_back(nest);
    return flat;
  }

  if (py::isinstance<py::list>(nest) || IsUnnamedtuple(nest)) {
    for (const auto& value : nest) {
      auto sub_flat = Flatten(py::cast<py::object>(value));
      flat.insert(flat.end(), sub_flat.begin(), sub_flat.end());
    }
  } else {
    for (const auto& item : ExtractFieldsFromNest(nest)) {
      auto sub_flat = Flatten(item.second);
      flat.insert(flat.end(), sub_flat.begin(), sub_flat.end());
    }
  }
  return flat;
}

std::vector<py::object> FlattenUpTo(const py::object& shallow_nest,
                                    const py::object& nest) {
  std::vector<py::object> ret;
  if (!IsNested(shallow_nest)) {
    ret.push_back(nest);
    return ret;
  }
  AssertSameType(shallow_nest, nest);
  AssertSameLength(shallow_nest, nest);

  if (py::isinstance<py::list>(shallow_nest) || IsUnnamedtuple(shallow_nest)) {
    for (unsigned int i = 0; i < py::len(shallow_nest); i++) {
      auto flat = FlattenUpTo(py::cast<py::object>(shallow_nest[GetPyInt(i)]),
                              py::cast<py::object>(nest[GetPyInt(i)]));
      ret.insert(ret.end(), flat.begin(), flat.end());
    }
  } else {
    auto fields_and_values1 = ExtractFieldsFromNest(shallow_nest);
    auto fields_and_values2 = ExtractFieldsFromNest(nest);
    for (unsigned int i = 0; i < fields_and_values1.size(); i++) {
      auto fv1 = fields_and_values1[i];
      auto fv2 = fields_and_values2[i];
      if (fv1.first != fv2.first) {
        throw std::runtime_error(py_format("Keys are different! {} <-> {}",
                                           fields_and_values1,
                                           fields_and_values2));
      }
      auto flat = FlattenUpTo(fv1.second, fv2.second);
      ret.insert(ret.end(), flat.begin(), flat.end());
    }
  }
  return ret;
}

void AssertSameStructure(const py::object& nest1, const py::object& nest2) {
  if (IsNested(nest1) || IsNested(nest2)) {
    AssertSameType(nest1, nest2);
    AssertSameLength(nest1, nest2);

    if (py::isinstance<py::list>(nest1) || IsUnnamedtuple(nest1)) {
      int i = 0;
      for (const auto& value1 : nest1) {
        auto value2 = nest2[GetPyInt(i)];
        AssertSameStructure(py::cast<py::object>(value1),
                            py::cast<py::object>(value2));
        i++;
      }
    } else {
      auto fields_values1 = ExtractFieldsFromNest(nest1);
      auto fields_values2 = ExtractFieldsFromNest(nest2);
      int i = 0;
      for (const auto& fv1 : fields_values1) {
        auto fv2 = fields_values2[i];
        if (fv1.first != fv2.first) {
          throw std::runtime_error(py_format(
              "Keys are different! {} <-> {}", fields_values1, fields_values2));
        }
        AssertSameStructure(fv1.second, fv2.second);
        i++;
      }
    }
  }
}

py::object _MapStructure(const py::function& func, const py::args& nests) {
  std::vector<py::object> nests_;
  for (const auto& nest : nests) {
    nests_.push_back(py::cast<py::object>(nest));
  }
  auto first_nest = nests_[0];
  auto is_first_nested = IsNested(first_nest);
  for (unsigned int i = 1; i < nests_.size(); i++) {
    if (is_first_nested || IsNested(nests_[i])) {
      AssertSameType(first_nest, nests_[i]);
      AssertSameLength(first_nest, nests_[i]);
    }
  }

  if (!is_first_nested) {
    return func(*nests);
  }
  auto first_nest_type = first_nest.get_type();
  py::object ret;
  if (py::isinstance<py::list>(first_nest) || IsUnnamedtuple(first_nest)) {
    py::list results;
    for (unsigned int i = 0; i < py::len(first_nest); i++) {
      py::list args;
      for (const auto& nest : nests_) {
        args.append(py::cast<py::object>(nest[GetPyInt(i)]));
      }
      results.append(_MapStructure(func, args));
    }
    ret = first_nest_type(results);
  } else {
    py::dict results;
    std::vector<field_list> fields_and_values;
    fields_and_values.reserve(nests_.size());
    for (const auto& nest : nests_) {
      fields_and_values.push_back(ExtractFieldsFromNest(nest));
    }
    for (unsigned int i = 0; i < fields_and_values[0].size(); i++) {
      auto field = fields_and_values[0][i].first;
      py::list values;
      for (const auto& fv : fields_and_values) {
        if (fv[i].first != field) {
          throw std::runtime_error(py_format(
              "Keys are different! {} <-> {}", fields_and_values[0], fv));
        }
        values.append(fv[i].second);
      }
      results[py::str(field)] = _MapStructure(func, values);
    }
    ret = first_nest_type(**results);
  }
  return ret;
}

py::object MapStructure(const py::function& func, const py::args& nests) {
  if (py::len(nests) == 0) {
    throw std::runtime_error("There should be at least one input nest!");
  }
  return _MapStructure(func, nests);
}

py::object _MapStructureUpTo(const py::object& shallow_nest,
                             const py::function& func,
                             const py::args& nests) {
  if (!IsNested(shallow_nest)) {
    return func(*nests);
  }

  std::vector<py::object> nests_;
  nests_.reserve(nests.size());
  for (const auto& nest : nests) {
    nests_.push_back(py::cast<py::object>(nest));
  }

  for (const auto& nest : nests_) {
    AssertSameType(shallow_nest, nest);
    AssertSameLength(shallow_nest, nest);
  }

  py::object ret;
  if (py::isinstance<py::list>(shallow_nest) || IsUnnamedtuple(shallow_nest)) {
    py::list results;
    for (unsigned int i = 0; i < py::len(shallow_nest); i++) {
      py::list args;
      for (const auto& nest : nests_) {
        args.append(py::cast<py::object>(nest[GetPyInt(i)]));
      }
      results.append(_MapStructureUpTo(
          py::cast<py::object>(shallow_nest[GetPyInt(i)]), func, args));
    }
    ret = (shallow_nest.get_type())(results);
  } else {
    py::dict results;
    auto shallow_fields_and_values = ExtractFieldsFromNest(shallow_nest);
    std::vector<field_list> fields_and_values;
    fields_and_values.reserve(nests_.size());
    for (const auto& nest : nests_) {
      fields_and_values.push_back(ExtractFieldsFromNest(nest));
    }
    for (unsigned int i = 0; i < shallow_fields_and_values.size(); i++) {
      auto shallow_fv = shallow_fields_and_values[i];
      auto shallow_field = shallow_fv.first;
      auto shallow_value = shallow_fv.second;
      py::list values;
      for (const auto& fvs : fields_and_values) {
        auto fv = fvs[i];
        if (shallow_field != fv.first) {
          throw std::runtime_error(
              py_format("Fields are not all the same: {} <-> {}",
                        shallow_fields_and_values,
                        fvs));
        }
        values.append(fv.second);
      }
      results[py::str(shallow_field)] =
          _MapStructureUpTo(shallow_value, func, values);
    }
    ret = (shallow_nest.get_type())(**results);
  }
  return ret;
}

py::object MapStructureUpTo(const py::object& shallow_nest,
                            const py::function& func,
                            const py::args& nests) {
  if (py::len(nests) == 0) {
    throw std::runtime_error("There should be at least one input nest!");
  }
  return _MapStructureUpTo(shallow_nest, func, nests);
}

py::object _PackSequenceAs(const py::object& nest,
                           const py::list& flat_seq,
                           int* i) {
  if (!IsNested(nest)) {
    auto ret = flat_seq[GetPyInt(*i)];
    (*i)++;
    return ret;
  }
  py::object ret;
  if (py::isinstance<py::list>(nest) || IsUnnamedtuple(nest)) {
    py::list results;
    for (const auto& value : nest) {
      results.append(_PackSequenceAs(py::cast<py::object>(value), flat_seq, i));
    }
    ret = (nest.get_type())(results);
  } else {
    py::dict results;
    for (const auto& fv : ExtractFieldsFromNest(nest)) {
      results[py::str(fv.first)] = _PackSequenceAs(fv.second, flat_seq, i);
    }
    ret = (nest.get_type())(**results);
  }
  return ret;
}

py::object PackSequenceAs(const py::object& nest, const py::object& flat_seq) {
  auto flat_nest = Flatten(nest);
  if (flat_nest.size() != py::len(flat_seq)) {
    throw std::runtime_error(
        py_format("Different lengths! {} <-> {}", nest, flat_seq));
  }
  int i = 0;
  return _PackSequenceAs(nest, py::cast<py::list>(flat_seq), &i);
}

py::object PruneNestLike(const py::object& nest,
                         const py::object& slim_nest,
                         const py::object& value_to_match = py::none()) {
  if (IsNested(nest) || IsNested(slim_nest)) {
    AssertSameType(nest, slim_nest);
    py::object ret;
    if (py::isinstance<py::list>(nest) || IsUnnamedtuple(nest)) {
      AssertSameLength(nest, slim_nest);
      py::list results;
      for (unsigned int i = 0; i < py::len(nest); i++) {
        auto n = py::cast<py::object>(nest[GetPyInt(i)]);
        auto sn = py::cast<py::object>(slim_nest[GetPyInt(i)]);
        if (sn.equal(value_to_match)) {
          results.append(sn);
        } else {
          results.append(PruneNestLike(n, sn, value_to_match));
        }
      }
      ret = (nest.get_type())(results);
    } else {
      py::dict results;

      std::map<std::string, py::object> fields_and_values;
      for (const auto& fv : ExtractFieldsFromNest(nest)) {
        fields_and_values[fv.first] = fv.second;
      }
      for (const auto& slim_fv : ExtractFieldsFromNest(slim_nest)) {
        auto field = slim_fv.first;
        auto slim_nest_value = slim_fv.second;
        if (fields_and_values.count(field) == 0) {
          throw std::runtime_error(
              py_format("Slim field {} not in nest {}!", field, nest));
        }
        auto nest_value = fields_and_values[field];
        if (slim_nest_value.equal(value_to_match)) {
          results[py::str(field)] = slim_nest_value;
        } else {
          results[py::str(field)] =
              PruneNestLike(nest_value, slim_nest_value, value_to_match);
        }
      }
      ret = (nest.get_type())(**results);
    }
    return ret;
  } else {
    return nest;
  }
}

PYBIND11_MODULE(cnest, m) {
  m.doc() = R"pbdoc(
            C++ implementation of several key nest functions that are performance
            critical.
        )pbdoc";

  // Private usage for testing purpose
  m.def("_extract_fields_from_nest", &ExtractFieldsFromNest);
  m.def("_is_nested", &IsNested);
  m.def("_is_namedtuple", &IsNamedtuple);
  m.def("_is_unnamedtuple", &IsUnnamedtuple);
  m.def("_assert_same_type", &AssertSameType);
  m.def("_assert_same_length", &AssertSameLength);

  // Exposed APIs
  m.def("flatten",
        &Flatten,
        R"pbdoc(
            Returns a flat list from a given nested structure.
          )pbdoc");
  m.def("assert_same_structure",
        &AssertSameStructure,
        R"pbdoc(
            Asserts that two structures are nested in the same way.
          )pbdoc");
  m.def("map_structure",
        &MapStructure,
        R"pbdoc(
            Applies func to each entry in structure and returns a new structure.
          )pbdoc",
        py::arg("func"));
  m.def("pack_sequence_as",
        &PackSequenceAs,
        R"pbdoc(
            Returns a given flattened sequence packed into a given structure.
          )pbdoc",
        py::arg("nest"),
        py::arg("flat_seq"));
  m.def("flatten_up_to",
        &FlattenUpTo,
        R"pbdoc(
            Flatten ``nests`` up to the depths of ``shallow_nest``. Every sub-nest
            of each of ``nests`` beyond the depth of the corresponding sub-nest in
            ``shallow_nest`` will be treated as a leaf that stops flattening downwards.
          )pbdoc",
        py::arg("shallow_nest"),
        py::arg("nest"));
  m.def("map_structure_up_to",
        &MapStructureUpTo,
        R"pbdoc(
            Applies a function to ``nests`` up to the depths of ``shallow_nest``.
            Every sub-nest of each of ``nests`` beyond the depth of the corresponding
            sub-nest in ``shallow_nest`` will be treated as a leaf and input to
            ``func``.
          )pbdoc",
        py::arg("shallow_nest"),
        py::arg("func"));
  m.def("prune_nest_like",
        &PruneNestLike,
        R"pbdoc(
            Prune a nested structure referring to another slim nest. Generally,
            for every corrsponding node, we only keep the fields that're contained
            in ``slim_nest``. In addition, if a field of ``slim_nest`` contains
            a value of ``value_to_match``, then the corresponding field of ``nest``
            will also be updated to this value.
          )pbdoc",
        py::arg("nest"),
        py::arg("slim_nest"),
        py::arg("value_to_match") = py::none());

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
