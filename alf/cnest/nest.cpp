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

#include <algorithm>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

template <typename... Args>
std::string _StrFormat(const std::string& format, Args... args) {
  // Extra space for '\0'
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  // We don't want the '\0' inside
  return std::string(buf.get(), buf.get() + size - 1);
}

bool IsNested(py::object value) {
  return (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value) ||
          py::isinstance<py::dict>(value));
}

bool IsNamedtuple(py::object value) {
  return (py::isinstance<py::tuple>(value) && py::hasattr(value, "_fields"));
}

bool IsUnnamedtuple(py::object value) {
  return (py::isinstance<py::tuple>(value) && !py::hasattr(value, "_fields"));
}

py::list ExtractFieldsFromNest(py::object nest) {
  bool is_dict = py::isinstance<py::dict>(nest);
  if (!(is_dict || IsNamedtuple(nest))) {
    throw std::invalid_argument(
        _StrFormat("Nest %s must be a dict or namedtuple!", py::repr(nest)));
  }

  std::vector<std::string> fields;
  py::dict nest_dict;
  if (is_dict) {
    nest_dict = py::cast<py::dict>(nest);
    for (const auto& item : nest_dict) {
      fields.push_back(py::str(item.first));
    }
  } else {
    for (const auto& field : nest.attr("_fields")) {
      fields.push_back(py::str(field));
    }
  }
  std::sort(fields.begin(), fields.end());

  py::list pyret;
  for (const auto& field : fields) {
    if (is_dict) {
      pyret.append(py::make_tuple(field, nest_dict[py::str(field)]));
    } else {
      pyret.append(py::make_tuple(field, nest.attr(field.c_str())));
    }
  }
  return pyret;
}

std::vector<py::object> _Flatten(py::object nest) {
  if (!IsNested(nest)) {
    std::vector<py::object> flat = {nest};
    return flat;
  }
  std::vector<py::object> flat;
  if (py::isinstance<py::list>(nest) || IsUnnamedtuple(nest)) {
    for (auto value : py::cast<py::list>(nest)) {
      auto sub_flat = _Flatten(py::cast<py::object>(value));
      flat.insert(flat.end(), sub_flat.begin(), sub_flat.end());
    }
  } else {
    for (auto item : ExtractFieldsFromNest(nest)) {
      auto sub_flat = _Flatten(item[py::int_(1)]);
      flat.insert(flat.end(), sub_flat.begin(), sub_flat.end());
    }
  }
  return flat;
}

py::list Flatten(py::object nest) {
  auto flat = _Flatten(nest);
  py::list ret;
  for (auto x : flat) {
    ret.append(x);
  }
  return ret;
}

PYBIND11_MODULE(cnest, m) {
  m.doc() = R"pbdoc(
            C++ implementation of several key nest functions:
            * ``extract_fields_from_nest``
            * ``flatten``
            * ``assert_same_structure``
        )pbdoc";

  m.def("extract_fields_from_nest",
        &ExtractFieldsFromNest,
        R"pbdoc(
            Extract fields and the corresponding values from a nest if it's
            either a ``namedtuple`` or ``dict``.

            Args:
                nest (nest): a nested structure

            Returns:
                Iterable: an iterator that generates ``(field, value)`` pairs.
                    The fields are sorted before being returned.

            Raises:
                ValueError: if the nest is neither ``namedtuple`` nor ``dict``.
          )pbdoc");

  m.def("flatten",
        &Flatten,
        R"pbdoc(
            Returns a flat list from a given nested structure.
          )pbdoc");

  m.def("is_nested",
        &IsNested,
        R"pbdoc(
            Returns true if the input is one of: ``list``, ``unnamedtuple``,
            ``dict``, or ``namedtuple``. Note that this definition is different
            from tf's is_nested where all types that are ``collections.abc.Sequence``
            are defined to be nested.
          )pbdoc");

  m.def("is_namedtuple",
        &IsNamedtuple,
        R"pbdoc(
            Whether the value is a namedtuple instance.

            Args:
                value (Object):
            Returns:
                ``True`` if the value is a namedtuple instance.
          )pbdoc");

  m.def("is_unnamedtuple",
        &IsUnnamedtuple,
        R"pbdoc(
            Whether the value is an unnamedtuple instance.

            Args:
                value (Object):
            Returns:
                ``True`` if the value is an unnamedtuple instance.
          )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
