#include "../include/csv.h"

#include <stdio.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../include/core.h"
#include "../include/neuron.h"
using namespace std;
TrainingImage LoadImageFromFile(int id) {
  ifstream inFile("../data/mnist_train.csv");
  string line;
  int currentLine = -1;
  string foundLine;
  // Find wanted line in file
  while (getline(inFile, line)) {
    currentLine++;
    if (currentLine == id) {
      foundLine = line;
      break;
    }
  }
  stringstream foundStream;
  foundStream.str(foundLine);
  string segment;
  vector<string> seglist;
  // Add line contents to vector
  while (getline(foundStream, segment, ',')) {
    seglist.push_back(segment);
  }
  // cout << seglist.size() << endl;
  vector<vector<int>> imageOutput(MaterialSize, vector<int>(MaterialSize));
  int imageOutputTarget = -1;
  int segX = 0;
  int segY = 0;
  // Load line into testimage struct
  for (int x = 0; x < seglist.size(); x++) {
    if (x == 0) {
      imageOutputTarget = stoi(seglist[x]);
    } else {
      if (segX > MaterialSize - 1) {
        segY++;
        segX = 0;
      }
      imageOutput[segY][segX] = stoi(seglist[x]);
      // cout << stoi(seglist[x]) << " AT " << segX << "," << segY << endl;
      // cout << segX << endl;
      segX++;
    }
  }

  TrainingImage output;
  output.image = imageOutput;
  output.target = imageOutputTarget;
  return output;
};