#ifndef _generator_h
#define _generator_h

#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

int partial_strings_match
(
    py::buffer match,
    py::buffer offsets,
    py::buffer strings
);

#endif