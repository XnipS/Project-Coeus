#pragma once

#include <fstream>
#include <iostream>
#include <sstream>

#include "core.h"
#include "neuron.h"

TrainingImage LoadImageFromFile(int id);

void PrintNeuronLayer(TrainingImage x);