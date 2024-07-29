// Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights
// Reserved.
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

/*
Boost should be installed first: apt install libboost-all-dev
Compile with the following command:

PYTHON=python3.8
g++ -O3 -Wall -shared -std=c++17 -fPIC -fvisibility=hidden -pg \
 `$PYTHON -m pybind11 --includes` \
  parallel_environment.cpp \
  -o _penv`$PYTHON-config --extension-suffix` -lrt

After compilation, a python extension module _penv will be generated, which
can be imported using "import _penv" from python.
*/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

namespace bip = boost::interprocess;
namespace py = pybind11;

const size_t addr_alignment = 16;

// The size of job queue.
// Since we always wait for job to finish before sending the next job,
// we only need a short job queue.
const size_t kJobQueueSize = 3;

inline size_t align(size_t x) {
  return ((x + addr_alignment - 1) / addr_alignment) * addr_alignment;
}

// TODO(emailweixu): directly use cnest c++ function
py::list Flatten(py::object nest) {
  py::object flatten = py::module::import("cnest").attr("flatten");
  return py::list(flatten(nest));
}

// TODO(emailweixu): directly use cnest c++ function
py::object PackSequenceAs(py::object nest, const py::list& flat_seq) {
  py::object pack_sequence_as =
      py::module::import("cnest").attr("pack_sequence_as");
  return pack_sequence_as(nest, flat_seq);
}

class SharedDataBuffer {
 public:
  SharedDataBuffer(py::object data_spec,
                   size_t num_slices,
                   const std::string& name);
  ~SharedDataBuffer();
  void WriteSlice(py::object nested_array, size_t slice_id);
  void WriteBatch(py::object nested_array, size_t begin_slice_id);
  void WriteWhole(py::object nested_array);

  inline py::object ViewAsNestedArray(size_t slice_id, size_t num_slices);
  py::list ViewAsFlattenedArray(size_t slice_id, size_t num_slices);

  void* GetBufByFieldName(const char* field) {
    auto array = py::buffer(ViewAsNestedArray(0, num_slices_).attr(field));
    py::buffer_info info = array.request();
    return info.ptr;
  }

 private:
  std::string name_;
  size_t num_slices_;
  py::object data_spec_;
  std::unique_ptr<bip::shared_memory_object> shm_;
  std::unique_ptr<bip::mapped_region> region_;
  char* buf_;
  std::vector<size_t> offsets_;  // offset of the i-th array
  std::vector<size_t> sizes_;    // size of each slice of the i-th array
  std::vector<py::buffer_info> buffer_infos_;
  std::vector<std::vector<ssize_t>> slice_strides_;
  std::vector<std::vector<ssize_t>> slice_shapes_;
  std::vector<py::dtype> dtypes_;

  inline char* GetBuf(int slice_id, int array_id) const {
    return buf_ + offsets_[array_id] + sizes_[array_id] * slice_id;
  }
  inline void CheckSize(const py::list& arrays, const py::object& nested_array);
};

void CheckStrides(const py::buffer_info& info,
                  const py::dtype& dtype,
                  const std::vector<ssize_t>& strides,
                  const std::vector<ssize_t>& non_batch_shape,
                  const py::buffer& buffer) {
  if (info.strides != strides) {
    // Sometimes the stride mismatch is actually caused by dtype mismatch.
    if (!py::dtype(info).is(dtype)) {
      throw std::runtime_error(py::str("dtype mismatch. Expected: {}, Got {}."
                                       "The offending buffer is {}")
                                   .format(dtype, py::dtype(info), buffer));
    } else {
      throw std::runtime_error(
          py::str(
              "Strides mismatch. Expected: {} (non-batch shape is {}), "
              "Got: {} (shape is {}). It might be caused by non-continguous "
              "array, which can be fixed by copying the array. "
              "The offending buffer is {}")
              .format(
                  strides, non_batch_shape, info.strides, info.shape, buffer));
    }
  }
}

void SharedDataBuffer::CheckSize(const py::list& arrays,
                                 const py::object& nested_array) {
  if (arrays.size() != sizes_.size()) {
    py::object consistent_with_spec =
        py::module::import("alf.utils.spec_utils").attr("consistent_with_spec");
    consistent_with_spec(nested_array, data_spec_);
    throw std::runtime_error(py::str("array structure mismatch."));
  }
}

void SharedDataBuffer::WriteSlice(py::object nested_array, size_t slice_id) {
  auto arrays = Flatten(nested_array);
  CheckSize(arrays, nested_array);
  try {
    for (size_t j = 0; j < arrays.size(); ++j) {
      auto buffer = py::buffer(arrays[j]);
      const py::buffer_info& info = buffer.request();
      if (info.ndim == 0) {
        memcpy(GetBuf(slice_id, j), info.ptr, info.itemsize);
      } else {
        CheckStrides(
            info, dtypes_[j], slice_strides_[j], slice_shapes_[j], buffer);
        memcpy(GetBuf(slice_id, j), info.ptr, info.shape[0] * info.strides[0]);
      }
    }
  } catch (const py::type_error& e) {
    throw std::runtime_error(
        py::str("Caught '{}' while writing an object. It is likely "
                "that one of the "
                "field is not converted to numpy array or numpy scalar "
                "as required. The object itself is {}.")
            .format(e.what(), nested_array));
  }
}

void SharedDataBuffer::WriteBatch(py::object nested_array,
                                  size_t begin_slice_id) {
  auto arrays = Flatten(nested_array);
  CheckSize(arrays, nested_array);
  for (size_t j = 0; j < arrays.size(); ++j) {
    auto buffer = py::buffer(arrays[j]);
    const py::buffer_info& info = buffer.request();
    CheckStrides(
        info, dtypes_[j], buffer_infos_[j].strides, slice_shapes_[j], buffer);
    if (begin_slice_id + info.shape[0] > num_slices_) {
      throw std::runtime_error(
          py::str("batch size too big: begin_slice_id: {} batch_size: {} "
                  "num_slices: {} shape: {}")
              .format(begin_slice_id, info.shape[0], num_slices_, info.shape));
    }
    memcpy(
        GetBuf(begin_slice_id, j), info.ptr, info.shape[0] * info.strides[0]);
  }
}

void SharedDataBuffer::WriteWhole(py::object nested_array) {
  auto arrays = Flatten(nested_array);
  CheckSize(arrays, nested_array);
  for (size_t j = 0; j < arrays.size(); ++j) {
    auto buffer = py::buffer(arrays[j]);
    const py::buffer_info& info = buffer.request();
    CheckStrides(
        info, dtypes_[j], buffer_infos_[j].strides, slice_shapes_[j], buffer);
    if (info.shape[0] != (signed)num_slices_) {
      throw std::runtime_error(
          py::str("Incorrect batch size. Expected: {}, got {}, shape: {}.")
              .format(num_slices_, info.shape[0], info.shape));
    }
    memcpy(GetBuf(0, j), info.ptr, info.shape[0] * info.strides[0]);
  }
}

SharedDataBuffer::SharedDataBuffer(py::object data_spec,
                                   size_t num_slices,
                                   const std::string& name)
    : name_(name), num_slices_(num_slices), data_spec_(data_spec) {
  auto outer_dims = py::make_tuple(2);

  auto flattened_data_spec = Flatten(data_spec);
  size_t offset = 0;
  for (size_t i = 0; i < flattened_data_spec.size(); ++i) {
    offsets_.push_back(offset);
    const auto& spec = flattened_data_spec[i];
    py::buffer array = py::reinterpret_borrow<py::buffer>(
        spec.attr("numpy_zeros")(outer_dims));
    buffer_infos_.emplace_back(array.request());

    // Stores the strides and shapes (ignoring the first dimension which is a
    // dummy batch dimension) of each slice.
    auto& strides = buffer_infos_.back().strides;
    slice_strides_.emplace_back(strides.begin() + 1, strides.end());
    const std::vector<ssize_t>& shape = buffer_infos_.back().shape;
    slice_shapes_.emplace_back(shape.begin() + 1, shape.end());

    dtypes_.emplace_back(py::dtype(buffer_infos_.back()));
    size_t slice_size = buffer_infos_.back().strides[0];
    sizes_.push_back(slice_size);
    offset += align(num_slices * slice_size);
  }

  shm_ = std::make_unique<bip::shared_memory_object>(
      bip::open_or_create, name.c_str(), bip::read_write);
  shm_->truncate(offset);
  region_ = std::make_unique<bip::mapped_region>(*shm_, bip::read_write);
  buf_ = reinterpret_cast<char*>(region_->get_address());
}

SharedDataBuffer::~SharedDataBuffer() {
  bip::shared_memory_object::remove(name_.c_str());
}

py::list SharedDataBuffer::ViewAsFlattenedArray(size_t slice_id,
                                                size_t num_slices = 0) {
  py::list flattened_array;
  for (size_t i = 0; i < buffer_infos_.size(); ++i) {
    auto& inf = buffer_infos_[i];
    auto strides = inf.strides;
    auto shape = inf.shape;
    auto ndim = inf.ndim;
    if (num_slices == 0) {
      shape.erase(shape.begin());
      strides.erase(strides.begin());
      --ndim;
    } else {
      shape[0] = num_slices;
    }
    py::buffer_info info(
        GetBuf(slice_id, i), inf.itemsize, inf.format, ndim, shape, strides);
    flattened_array.append(py::array(info, data_spec_));
  }
  return flattened_array;
}

py::object SharedDataBuffer::ViewAsNestedArray(size_t slice_id,
                                               size_t num_slices = 0) {
  return PackSequenceAs(data_spec_, ViewAsFlattenedArray(slice_id, num_slices));
}

class EnvironmentBase {
 public:
  std::string name_;
  int num_envs_;
  int batch_size_per_env_;
  bool batched_;

 protected:
  SharedDataBuffer action_buffer_;
  SharedDataBuffer timestep_buffer_;

  EnvironmentBase(int num_envs,
                  int batch_size_per_env,
                  bool batched,
                  const py::object& action_spec,
                  const py::object& timestep_spec,
                  const std::string& name);
};

EnvironmentBase::EnvironmentBase(int num_envs,
                                 int batch_size_per_env,
                                 bool batched,
                                 const py::object& action_spec,
                                 const py::object& timestep_spec,
                                 const std::string& name)
    : name_(name),
      num_envs_(num_envs),
      batch_size_per_env_(batch_size_per_env),
      batched_(batched || batch_size_per_env > 1),
      action_buffer_(
          action_spec, num_envs * batch_size_per_env, name + ".action"),
      timestep_buffer_(
          timestep_spec, num_envs * batch_size_per_env, name + ".timestep") {}

class ParallelEnvironment : public EnvironmentBase {
 protected:
  int num_spare_envs_;
  std::vector<int> original_env_ids_;  // [num_envs]
  std::vector<uint8_t> reseted_;       // [num_envs]
  std::deque<int> reset_queue_;        // queue of the original env id
  py::object timestep_array_;
  int32_t* step_type_buf_;
  std::vector<std::unique_ptr<bip::message_queue>> job_queues_;
  bip::message_queue ready_queue_;

 public:
  ParallelEnvironment(int num_envs,
                      int num_spare_envs,
                      int batch_size_per_env,
                      bool batched,
                      py::object action_spec,
                      py::object timestep_spec,
                      const std::string& name);
  ~ParallelEnvironment();
  py::object Step(const py::object& action);
  py::object Reset();
};

enum class JobType { step, reset, close, call };
enum class StepType { first = 0, mid = 1, last = 2 };

struct Job {
  JobType type;
  int env_id;
};

ParallelEnvironment::ParallelEnvironment(int num_envs,
                                         int num_spare_envs,
                                         int batch_size_per_env,
                                         bool batched,
                                         py::object action_spec,
                                         py::object timestep_spec,
                                         const std::string& name)
    : EnvironmentBase(num_envs,
                      batch_size_per_env,
                      batched,
                      action_spec,
                      timestep_spec,
                      name),
      num_spare_envs_(num_spare_envs),
      original_env_ids_(num_envs),
      reseted_(num_envs, 0),
      timestep_array_(
          timestep_buffer_.ViewAsNestedArray(0, num_envs * batch_size_per_env)),
      ready_queue_(bip::open_or_create,
                   (name + ".ready_queue").c_str(),
                   3 * num_envs,  // we need it to be long enough to handle
                                  // ProcessEnvironment.Quit
                   sizeof(int)) {
  if (batch_size_per_env > 1 && num_spare_envs != 0) {
    throw std::runtime_error(py::str("num_spare_envs != 0 is not supported "
                                     "when batch_size_per_env > 0. "
                                     "batch_size_per_env={}, num_spare_envs={}")
                                 .format(batch_size_per_env, num_spare_envs));
  }
  for (int i = 0; i < num_envs; ++i) {
    original_env_ids_[i] = i;
  }
  for (int i = 0; i < num_spare_envs; ++i) {
    reset_queue_.push_back(num_envs + i);
  }
  step_type_buf_ = reinterpret_cast<int32_t*>(
      timestep_buffer_.GetBufByFieldName("step_type"));

  for (int i = 0; i < num_envs + num_spare_envs; ++i) {
    job_queues_.emplace_back(std::make_unique<bip::message_queue>(
        bip::open_or_create,
        (name + ".job_queue." + std::to_string(i)).c_str(),
        kJobQueueSize,
        sizeof(Job)));
  }
}

ParallelEnvironment::~ParallelEnvironment() {
  bip::message_queue::remove((name_ + ".ready_queue").c_str());
  for (int i = 0; i < num_envs_ + num_spare_envs_; ++i) {
    bip::message_queue::remove(
        (name_ + ".job_queue." + std::to_string(i)).c_str());
  }
}

py::object ParallelEnvironment::Step(const py::object& action) {
  action_buffer_.WriteWhole(action);
  for (int i = 0; i < num_envs_; ++i) {
    if (reseted_[i]) {
      original_env_ids_[i] = reset_queue_[0];
      reset_queue_.pop_front();
      reseted_[i] = false;
    }
  }
  for (int env_id = 0; env_id < num_envs_; ++env_id) {
    Job job{JobType::step, env_id};
    job_queues_[original_env_ids_[env_id]]->send(&job, sizeof(job), 0);
  }
  for (int i = 0; i < num_envs_; ++i) {
    bip::message_queue::size_type recvd_size;
    unsigned int priority;
    int env_id;
    ready_queue_.receive(&env_id, sizeof(env_id), recvd_size, priority);
    if (env_id == -1) {
      throw std::runtime_error("ProcessEnvironment is interrupted");
    }
    if (recvd_size != sizeof(env_id)) {
      throw std::runtime_error(py::str("Received unexpected size from "
                                       "ready_queue. Expected: {}, got: {}.")
                                   .format(sizeof(env_id), recvd_size));
    }
  }
  // It is important to send the reset job after the step job, so that the
  // order of reset_queue is deterministic.
  if (batch_size_per_env_ == 1) {
    for (int env_id = 0; env_id < num_envs_; ++env_id) {
      if (step_type_buf_[env_id] == static_cast<int32_t>(StepType::last)) {
        Job job{JobType::reset};
        job_queues_[original_env_ids_[env_id]]->send(&job, sizeof(job), 0);
        reseted_[env_id] = true;
        reset_queue_.push_back(original_env_ids_[env_id]);
      }
    }
  }
  return timestep_array_;
}

py::object ParallelEnvironment::Reset() {
  reset_queue_.clear();
  for (int env_id = 0; env_id < num_envs_ + num_spare_envs_; ++env_id) {
    Job job{JobType::reset};
    job_queues_[env_id]->send(&job, sizeof(job), 0);
    if (env_id < num_envs_) {
      original_env_ids_[env_id] = env_id;
      reseted_[env_id] = false;
    } else {
      reset_queue_.push_back(env_id);
    }
  }
  reseted_.assign(num_envs_, false);
  return Step(
      action_buffer_.ViewAsNestedArray(0, num_envs_ * batch_size_per_env_));
}

class ProcessEnvironment : public EnvironmentBase {
  std::string name_;
  int env_id_;
  py::object env_;
  py::function call_handler_;
  py::object py_step_;
  py::object py_reset_;
  bip::message_queue job_queue_;
  bip::message_queue ready_queue_;
  py::object reset_result_;
  bool just_reseted_;
  int32_t* env_id_buf_;

 public:
  ProcessEnvironment(py::object env,
                     py::function call_handler,
                     int env_id,
                     int num_envs,
                     int batch_size_per_env,
                     bool batched,
                     const py::object& action_spec,
                     const py::object& timestep_spec,
                     const std::string& name);
  ~ProcessEnvironment();
  void Worker();
  void Quit();

 protected:
  void Step(int env_id);
  void Reset();
  void SendTimestep(const py::object& timestep, int env_id);
};

ProcessEnvironment::ProcessEnvironment(py::object env,
                                       py::function call_handler,
                                       int env_id,
                                       int num_envs,
                                       int batch_size_per_env,
                                       bool batched,
                                       const py::object& action_spec,
                                       const py::object& timestep_spec,
                                       const std::string& name)
    : EnvironmentBase(num_envs,
                      batch_size_per_env,
                      batched,
                      action_spec,
                      timestep_spec,
                      name),
      name_(name),
      env_id_(env_id),
      env_(env),
      call_handler_(call_handler),
      py_step_(env.attr("step")),
      py_reset_(env.attr("reset")),
      job_queue_(bip::open_or_create,
                 (name + ".job_queue." + std::to_string(env_id)).c_str(),
                 kJobQueueSize,
                 sizeof(Job)),
      ready_queue_(bip::open_or_create,
                   (name + ".ready_queue").c_str(),
                   3 * num_envs,  // we need it to be long enough to handle
                                  // ProcessEnvironment.Quit
                   sizeof(int)),
      just_reseted_(false) {
  env_id_buf_ =
      reinterpret_cast<int32_t*>(timestep_buffer_.GetBufByFieldName("env_id"));
}

ProcessEnvironment::~ProcessEnvironment() {
  bip::message_queue::remove(
      (name_ + ".job_queue." + std::to_string(env_id_)).c_str());
  bip::message_queue::remove((name_ + ".ready_queue").c_str());
}

void ProcessEnvironment::Quit() {
  // Inform the main process to quit
  int env_id = -1;
  ready_queue_.send(&env_id, sizeof(env_id), 0);
}
class ProcessEnvironmentCaller {
  std::string name_;
  int env_id_;
  bip::message_queue job_queue_;

 public:
  ProcessEnvironmentCaller(int env_id, const std::string& name)
      : name_(name),
        env_id_(env_id),
        job_queue_(bip::open_or_create,
                   (name + ".job_queue." + std::to_string(env_id)).c_str(),
                   kJobQueueSize,
                   sizeof(Job)) {}
  ~ProcessEnvironmentCaller() {
    bip::message_queue::remove(
        (name_ + ".job_queue." + std::to_string(env_id_)).c_str());
  }

  void Call();
  void Close();
};

void ProcessEnvironmentCaller::Call() {
  Job job{JobType::call};
  job_queue_.send(&job, sizeof(job), 0);
}

void ProcessEnvironmentCaller::Close() {
  Job job{JobType::close};
  job_queue_.send(&job, sizeof(job), 0);
}

void ProcessEnvironment::Step(int env_id) {
  py::object timestep;
  if (just_reseted_) {
    timestep = reset_result_;
    just_reseted_ = false;
  } else {
    auto action = action_buffer_.ViewAsNestedArray(
        env_id * batch_size_per_env_, batched_ ? batch_size_per_env_ : 0);
    timestep = py_step_(action);
  }
  SendTimestep(timestep, env_id);
}

void ProcessEnvironment::Reset() {
  reset_result_ = py_reset_();
  just_reseted_ = true;
}

void ProcessEnvironment::SendTimestep(const py::object& timestep, int env_id) {
  if (!batched_) {
    timestep_buffer_.WriteSlice(timestep, env_id);
    env_id_buf_[env_id] = env_id;
  } else {
    int begin_slice_id = env_id * batch_size_per_env_;
    timestep_buffer_.WriteBatch(timestep, begin_slice_id);
    for (int i = 0; i < batch_size_per_env_; ++i) {
      env_id_buf_[begin_slice_id + i] = begin_slice_id + i;
    }
  }
  ready_queue_.send(&env_id, sizeof(env_id), 0);
}

void ProcessEnvironment::Worker() {
  while (true) {
    Job job;
    bip::message_queue::size_type recvd_size;
    unsigned int priority;
    job_queue_.receive(&job, sizeof(job), recvd_size, priority);
    if (recvd_size != sizeof(job)) {
      throw std::runtime_error(py::str("Received unexpected size from "
                                       "job_queue. Expected: {}, got: {}.")
                                   .format(sizeof(job), recvd_size));
    }
    if (job.type == JobType::step) {
      Step(job.env_id);
    } else if (job.type == JobType::reset) {
      Reset();
    } else if (job.type == JobType::close) {
      env_.attr("close")();
      break;
    } else if (job.type == JobType::call) {
      // Use call_handler to handle other types of communication with unknown
      // size.
      call_handler_();
    }
  }
}

PYBIND11_MODULE(_penv, m) {
  py::class_<ParallelEnvironment>(m, "ParallelEnvironment")
      .def(py::init<int,
                    int,
                    int,
                    int,
                    py::object,
                    py::object,
                    const std::string&>(),
           py::arg("num_envs"),
           py::arg("num_spare_envs"),
           py::arg("batch_size_per_env"),
           py::arg("batched"),
           py::arg("action_spec"),
           py::arg("timestep_spec"),
           py::arg("name"))
      .def("step",
           &ParallelEnvironment::Step,
           R"pbdoc(
                Step environment for one step.
            )pbdoc",
           py::arg("action"))
      .def("reset", &ParallelEnvironment::Reset)
      .def_readonly("name", &ParallelEnvironment::name_)
      .def_readonly("num_envs", &ParallelEnvironment::num_envs_);
  py::class_<ProcessEnvironment>(m, "ProcessEnvironment")
      .def(py::init<py::object,
                    py::function,
                    int,
                    int,
                    int,
                    int,
                    py::object,
                    py::object,
                    const std::string&>(),
           py::arg("env"),
           py::arg("call_handler"),
           py::arg("env_id"),
           py::arg("num_envs"),
           py::arg("batch_size_per_env"),
           py::arg("batched"),
           py::arg("action_spec"),
           py::arg("timestep_spec"),
           py::arg("name"))
      .def("worker", &ProcessEnvironment::Worker)
      .def("quit",
           &ProcessEnvironment::Quit,
           R"pbdoc(
              Inform the main process to quit.
           )pbdoc");
  py::class_<ProcessEnvironmentCaller>(m, "ProcessEnvironmentCaller")
      .def(py::init<int, const std::string&>())
      .def("close", &ProcessEnvironmentCaller::Close)
      .def("call",
           &ProcessEnvironmentCaller::Call,
           R"pbdoc(
                Call the environments.
            )pbdoc");
}
