#pragma once
#include <string>
#include <vector>
struct Connection;
struct Neuron {
  double activation = 0;
  float bias = 1;
  std::vector<Connection> connections;
};
struct Connection {
  float weight;
  Neuron connectedTo;
};
struct TrainingImage {
  int target;
  std::vector<int> rawData;
  std::vector<std::vector<int>> image;
};
