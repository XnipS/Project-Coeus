#include "../include/csv.h"

#include <cmath>
using namespace std;
TrainingImage LoadImageFromFile(int id) {
  ifstream inFile(TrainingMaterialLocation);
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
  vector<int> rawData;
  vector<vector<int>> imageOutput(MaterialSize, vector<int>(MaterialSize));
  int imageOutputTarget = -1;
  int segX = 0;
  int segY = 0;
  // Load line into testimage struct
  for (int x = 0; x < seglist.size(); x++) {
    if (x == 0) {
      imageOutputTarget = stoi(seglist[x]);
    } else {
      rawData.push_back(stoi(seglist[x]));
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
  output.rawData = rawData;
  return output;
};

void PrintNeuronLayer(TrainingImage x) {
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
  std::cout << "Target: " << x.target << std::endl;
}