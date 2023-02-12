#pragma once
#include <string>
#include <vector>
struct Connection;
struct Neuron {
  float activation = 0;
  float bias = 1;
  Connection *connections[];
};
struct Connection {
  float weight;
  Neuron connectedTo;
};
struct TrainingImage {
  int target;
  std::vector<std::vector<int>> image;
};
