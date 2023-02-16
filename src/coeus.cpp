#include "../include/coeus.h"

#include <iostream>
#include <iterator>
#include <string>

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

float CalculateCost(int target) {
  float output = 0;
  for (int i = 0; i < outputLayer.size(); i++) {
    if (i == target) {
      output += std::sqrt(std::abs(outputLayer[i].activation - 1.0));
    } else {
      output += std::sqrt(std::abs(outputLayer[i].activation));
    }
  }
  return output;
}

void ConnectNeuronToLayer(Neuron* input, std::vector<Neuron> layer) {
  Neuron output = *input;
  for (int i = 0; i < layer.size(); i++) {
    Connection con;
    con.connectedTo = i;
    con.weight = 1;  // TODO change to random number
    output.connections.push_back(con);
  }
  *input = output;
}
void GenerateNeurons() {
  for (int a = 0; a < inputLayer.size(); a++) {
    inputLayer[a].activation =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
    ;  //(1.0 / 255.0) * input.rawData[a];
    inputLayer[a].bias =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
  }
  for (int a = 0; a < layer1.size(); a++) {
    layer1[a].activation =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
    layer1[a].bias =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
  }
  for (int a = 0; a < layer2.size(); a++) {
    layer2[a].activation =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
    layer2[a].bias =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
  }
  for (int a = 0; a < outputLayer.size(); a++) {
    outputLayer[a].activation =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
    outputLayer[a].bias =
        static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
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
    // std::cout << outputLayer[0].connections[0].connectedTo->activation
    //           << std::endl;
  }
}

float CalculateNeuronActivation(Neuron* input, std::vector<Neuron> upperLayer) {
  Neuron out = *input;
  float output = 0;
  for (int i = 0; i < input->connections.size(); i++) {
    output += input->connections[i].weight *
              upperLayer[input->connections[i].connectedTo].activation;
  }
  output += input->bias;
  // std::cout << output << std::endl;
  return Sigmoid(output);
}

void FireNeurons() {
  for (int a = 0; a < layer1.size(); a++) {
    layer1[a].activation = CalculateNeuronActivation(&layer1[a], layer1);
  }
  for (int a = 0; a < layer2.size(); a++) {
    layer2[a].activation = CalculateNeuronActivation(&layer2[a], layer2);
  }
  for (int a = 0; a < outputLayer.size(); a++) {
    outputLayer[a].activation =
        CalculateNeuronActivation(&outputLayer[a], outputLayer);
  }
}
#define alpha 1.0
void BackProp(Neuron* in, float y, std::vector<Neuron> upperLayer) {
  Neuron input = *in;
  for (int x = 0; x < input.connections.size(); x++) {
    // Get deltaWeight
    float delta = upperLayer[input.connections[x].connectedTo].activation;
    delta *= DyDxSigmoid(
        input.connections[x].weight *
            upperLayer[input.connections[x].connectedTo].activation +
        input.bias);
    delta *= 2 * (input.activation - y);
    // Set deltaWeight
    input.connections[x].weight -= delta * alpha;

    // Get deltaBias
    float bias = 1;
    bias *= DyDxSigmoid(
        input.connections[x].weight *
            upperLayer[input.connections[x].connectedTo].activation +
        input.bias);
    bias *= 2 * (input.activation - y);
    // Set deltaBias
    input.bias -= bias * alpha;

    // Get deltaActivation
    float act = input.connections[x].weight;
    act *= DyDxSigmoid(
        input.connections[x].weight *
            upperLayer[input.connections[x].connectedTo].activation +
        input.bias);
    act *= 2 * (input.activation - y);
    // Set deltaActivation
    upperLayer[input.connections[x].connectedTo].activation -=
        act * alpha;  // HERE
  }
  *in = input;
}
void BackPropogate(int target) {
  // Output layer
  for (int n = 0; n < outputLayer.size(); n++) {
    int y = 0;
    if (n == target) {
      y = 1;
    }
    BackProp(&outputLayer[n], y, outputLayer);
  }
  // Layer2
  for (int n = 0; n < layer2.size(); n++) {
    int y = layer2[n].activation;

    BackProp(&layer2[n], y, layer2);
  }

  // Layer1
  for (int n = 0; n < layer1.size(); n++) {
    int y = layer1[n].activation;

    BackProp(&layer1[n], y, layer1);
  }
}

void ReadNewData(TrainingImage input) {
  for (int a = 0; a < inputLayer.size(); a++) {
    inputLayer[a].activation = (1.0 / 255.0) * input.rawData[a];
  }
}

int BestGuess() {
  float max = 0.0;
  int guess = 0.0;
  for (int a = 0; a < outputLayer.size(); a++) {
    if (outputLayer[a].activation > max) {
      max = outputLayer[a].activation;
      guess = a;
    }
  }
  return guess;
}

int Entry(int argc, char* args[]) {
  // Initialise random number gen
  srand(static_cast<unsigned>(time(0)));
  // Setup neurons
  GenerateNeurons();
  ConnectNeurons();

  // Training
  for (int i = 1; i < 100; i++) {
    std::cout << NIP_BigLine << std::endl;
    int correct = 0;
    int total = 0;
    // Load input
    TrainingImage x = LoadImageFromFile(i);
    ReadNewData(x);
    double error = 100.0;
    // while (CalculateCost(x.target) > 1) {  // Percent
    total++;
    //  Fire neurons
    FireNeurons();
    //  Backprop
    BackPropogate(x.target);
    error = ((1.0 - (double)correct / (double)(total + 1)) * 100);

    std::cout << "Cost: " << CalculateCost(x.target)
              << " Guess: " << BestGuess() << " Real: " << x.target;
    if (x.target == BestGuess()) {
      std::cout << " TRUE";
      correct++;
    } else {
      std::cout << " FALSE";
    }
    std::cout << " Error: " << error << "%" << std::endl;
    // }
  }
  std::string in = "0";
  while (in != "exit") {
    std::cin >> in;
    TrainingImage x = LoadImageFromFile(std::stoi(in) + 1);
    ReadNewData(x);

    //  Fire neurons
    FireNeurons();
    //  Backprop
    // BackPropogate(x.target);

    std::cout << "Cost: " << CalculateCost(x.target)
              << " Guess: " << BestGuess() << " Real: " << x.target;
    if (x.target == BestGuess()) {
      std::cout << " TRUE";
    } else {
      std::cout << " FALSE";
    }
  }

  // Backprop test
  // for (int i = 0; i < 10; i++) {
  //   FireNeurons();
  //   BackPropogate(x.target);
  //   std::cout << "Cost After [" << i + 1 << "]: " << CalculateCost(x.target)
  //             << std::endl;
  // }

  // Connection test
  // std::cout << "Connection Count: " << layer1[0].connections.size()
  //           << std::endl;
  // std::cout << "Connection activation Test: "
  //           << layer1[0].connections[12].connectedTo.activation << std::endl;
  std::cout << "Done!" << std::endl;
  return 0;
};