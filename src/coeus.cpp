#include "../include/coeus.h"

#include <math.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <ostream>

#include "../include/core.h"
#include "../include/csv.h"
#include "../include/neuron.h"
#include "Eigen/src/Core/Matrix.h"

std::vector<Neuron> inputLayer(784);
std::vector<Neuron> layer1(16);
std::vector<Neuron> layer2(16);
std::vector<Neuron> outputLayer(10);

float Sigmoid(float input) {
  float x = 1.0 / (1.0 + exp(-input));
  // std::cout << x << std::endl;
  return x;
}
float DyDxSigmoid(float input) {
  return Sigmoid(input) * (1.0 - Sigmoid(input));
}

double CalculateCost(int target) {
  double output = 0;
  for (int i = 0; i < outputLayer.size(); i++) {
    if (i == target) {
      output += std::sqrt(abs(outputLayer[i].activation - 1.0));
    } else {
      output += std::sqrt(abs(outputLayer[i].activation));
    }
  }
  return output;
}

void ConnectNeuronToLayer(Neuron* input, std::vector<Neuron> layer) {
  for (int i = 0; i < layer.size(); i++) {
    Connection con;
    con.connectedTo = layer[i];
    con.weight = 1;  // TODO change to random number
    input->connections.push_back(con);
  }
}
void GenerateNeurons(TrainingImage input) {
  for (int a = 0; a < inputLayer.size(); a++) {
    inputLayer[a].activation =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
    ;  //(1.0 / 255.0) * input.rawData[a];
    inputLayer[a].bias =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
  }
  for (int a = 0; a < layer1.size(); a++) {
    layer1[a].activation =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
    layer1[a].bias =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
  }
  for (int a = 0; a < layer2.size(); a++) {
    layer2[a].activation =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
    layer2[a].bias =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
  }
  for (int a = 0; a < outputLayer.size(); a++) {
    outputLayer[a].activation =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
    outputLayer[a].bias =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
  }
}

void ConnectNeurons() {
  // for (int a = 0; a < 784; a++) {
  // }
  //  inputlayer has no connections
  for (int a = 0; a < layer1.size(); a++) {
    ConnectNeuronToLayer(&layer1[a], inputLayer);
  }
  for (int a = 0; a < layer2.size(); a++) {
    ConnectNeuronToLayer(&layer2[a], layer1);
  }
  for (int a = 0; a < outputLayer.size(); a++) {
    ConnectNeuronToLayer(&outputLayer[a], layer2);
    // std::vector<Connection> cons;
    // outputLayer[a].connections = cons;
  }
}

float CalculateNeuronActivation(Neuron* input) {
  float output = 0;
  for (int i = 0; i < input->connections.size(); i++) {
    output += input->connections[i].weight *
              input->connections[i].connectedTo.activation;
  }
  output += input->bias;
  // std::cout << output << std::endl;
  return Sigmoid(output);
}

void FireNeurons() {
  for (int a = 0; a < layer1.size(); a++) {
    layer1[a].activation = CalculateNeuronActivation(&layer1[a]);
  }
  for (int a = 0; a < layer2.size(); a++) {
    layer2[a].activation = CalculateNeuronActivation(&layer2[a]);
  }
  for (int a = 0; a < outputLayer.size(); a++) {
    outputLayer[a].activation = CalculateNeuronActivation(&outputLayer[a]);
  }
}

void BackPropogate(int target) {
  // Output layer
  for (int n = 0; n < outputLayer.size(); n++) {
    int y = 0;
    if (n == y) {
      n = 1;
    }
    for (int x = 0; x < outputLayer[n].connections.size(); x++) {
      // Get deltaWeight
      float delta = outputLayer[n].connections[x].connectedTo.activation;
      delta *=
          DyDxSigmoid(outputLayer[n].connections[x].weight *
                          outputLayer[n].connections[x].connectedTo.activation +
                      outputLayer[n].bias);
      delta *= 2 * (outputLayer[n].activation - n);
      // Set deltaWeight
      outputLayer[n].connections[x].weight += delta;

      // Get deltaBias
      float bias = 1;
      bias *=
          DyDxSigmoid(outputLayer[n].connections[x].weight *
                          outputLayer[n].connections[x].connectedTo.activation +
                      outputLayer[n].bias);
      bias *= 2 * (outputLayer[n].activation - n);
      // Set deltaBias
      outputLayer[n].bias += bias;

      // Get deltaActivation
      float act = outputLayer[n].connections[x].weight;
      act *=
          DyDxSigmoid(outputLayer[n].connections[x].weight *
                          outputLayer[n].connections[x].connectedTo.activation +
                      outputLayer[n].bias);
      act *= 2 * (outputLayer[n].activation - n);
      // Set deltaActivation
      outputLayer[n].connections[x].connectedTo.activation += act;
    }
  }
  // Layer2
  for (int n = 0; n < layer2.size(); n++) {
    int y = layer2[n].activation;

    for (int x = 0; x < layer2[n].connections.size(); x++) {
      // Get deltaWeight
      float delta = layer2[n].connections[x].connectedTo.activation;
      delta *= DyDxSigmoid(layer2[n].connections[x].weight *
                               layer2[n].connections[x].connectedTo.activation +
                           layer2[n].bias);
      delta *= 2 * (layer2[n].activation - n);
      // Set deltaWeight
      layer2[n].connections[x].weight += delta;

      // Get deltaBias
      float bias = 1;
      bias *= DyDxSigmoid(layer2[n].connections[x].weight *
                              layer2[n].connections[x].connectedTo.activation +
                          layer2[n].bias);
      bias *= 2 * (layer2[n].activation - n);
      // Set deltaBias
      layer2[n].bias += bias;

      // Get deltaActivation
      float act = layer2[n].connections[x].weight;
      act *= DyDxSigmoid(layer2[n].connections[x].weight *
                             layer2[n].connections[x].connectedTo.activation +
                         layer2[n].bias);
      act *= 2 * (layer2[n].activation - n);
      // Set deltaActivation
      layer2[n].connections[x].connectedTo.activation += act;
    }
  }

  // Layer1
  for (int n = 0; n < layer1.size(); n++) {
    int y = layer1[n].activation;

    for (int x = 0; x < layer1[n].connections.size(); x++) {
      // Get deltaWeight
      float delta = layer1[n].connections[x].connectedTo.activation;
      delta *= DyDxSigmoid(layer1[n].connections[x].weight *
                               layer1[n].connections[x].connectedTo.activation +
                           layer1[n].bias);
      delta *= 2 * (layer1[n].activation - n);
      // Set deltaWeight
      layer1[n].connections[x].weight += delta;

      // Get deltaBias
      float bias = 1;
      bias *= DyDxSigmoid(layer1[n].connections[x].weight *
                              layer1[n].connections[x].connectedTo.activation +
                          layer1[n].bias);
      bias *= 2 * (layer1[n].activation - n);
      // Set deltaBias
      layer1[n].bias += bias;

      // Get deltaActivation
      float act = layer1[n].connections[x].weight;
      act *= DyDxSigmoid(layer1[n].connections[x].weight *
                             layer1[n].connections[x].connectedTo.activation +
                         layer1[n].bias);
      act *= 2 * (layer1[n].activation - n);
      // Set deltaActivation
      layer1[n].connections[x].connectedTo.activation += act;
    }
  }
}

int Entry(int argc, char* args[]) {
  // Initialise random number gen
  srand(static_cast<unsigned>(time(0)));
  // Get training image
  TrainingImage x = LoadImageFromFile(8);
  PrintNeuronLayer(x);
  GenerateNeurons(x);
  ConnectNeurons();

  FireNeurons();
  BackPropogate(x.target);
  std::cout << "Cost Before: " << CalculateCost(x.target) << std::endl;

  for (int i = 0; i < 10; i++) {
    FireNeurons();
    BackPropogate(x.target);
    std::cout << "Cost After [" << i << "]: " << CalculateCost(x.target)
              << std::endl;
  }
  std::cout << "Connection Count: " << layer1[0].connections.size()
            << std::endl;
  std::cout << "Connection activation Test: "
            << layer1[0].connections[12].connectedTo.activation << std::endl;
  std::cout << "Done!" << std::endl;
  return 0;
};