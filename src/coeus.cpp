#include "../include/coeus.h"

#include <iostream>

#include "../include/core.h"
#include "../include/csv.h"
#include "../include/neuron.h"

Neuron inputLayer[784];

void InitiateInputNeuronLayer() {
  // for (csv::CSVRow& row : reader) {  // Input iterator
  //   for (csv::CSVField& field : row) {
  //  By default, get<>() produces a std::string.
  //  A more efficient get<string_view>() is also available, where the
  //  resulting string_view is valid as long as the parent CSVRow is alive
  //  std::cout << field.get<>() << std::endl;
  // }
  //}

  // for (int i = 0; i < sizeof(inputLayer); i++) {
  //   inputLayer->activation = 0;
  //   inputLayer->bias = 0;
  //  inputLayer->connections
  //}
}

int Entry(int argc, char* args[]) {
  InitiateInputNeuronLayer();

  TrainingImage x = LoadImageFromFile(3);

  // Print testimage
  for (int i = 0; i < MaterialSize - 1; i++) {
    for (int y = 0; y < MaterialSize - 1; y++) {
      std::string out = "\u25A1";
      if (x.image[i][y] > 0) {
        out = "\u25A0";
      }
      std::cout << out;  // imageOutput[i][y];  // out;
    }
    std::cout << std::endl;
  }
  // Print target
  std::cout << x.target << std::endl;

  std::cout << "Done!" << std::endl;
  return 0;
};