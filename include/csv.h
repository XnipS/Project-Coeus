#pragma once
#include <stdio.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../include/core.h"
#include "../include/neuron.h"
#include "neuron.h"
TrainingImage LoadImageFromFile(int id);

void PrintNeuronLayer(TrainingImage x);