#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

#define CONST_E 2.7182818284590452353602874713527
#define LEARNING_CURVE 30000

//network with nodes 784-32-16-10
//  and with weights 25088-512-160
struct node{
    float value = 0.5;
    float bias = 0;
};

struct images{
    float pixels[784];
    unsigned char label;
};



void initialize_array(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    neurons[0].resize(784);
    neurons[1].resize(32);
    neurons[2].resize(16);
    neurons[3].resize(10);
    for(int j = 1; j < 4; j++)
        for(node & i : neurons[j])
            i.bias = ((float)(rand() % 64) - 31.5) / 16;

    weights[0].resize(25088);
    weights[1].resize(512);
    weights[2].resize(160);
    for(int j = 0; j < 3; j++)
        for(float & i : weights[j])
            i = ((float)(rand() % 64) - 31.5) / 16;
}

void loadImages(const std::string& image_path, const std::string& label_path, std::vector<images>& all_images){
    std::ifstream images_file, label_file;
    char* images_data;
    char* label_data;

    images_file.open(image_path, std::ios::in);
    label_file.open(label_path, std::ios::in);
    if(!images_file.is_open() || !label_file.is_open())
        throw;//don't have time to throw a meaningful exception so i just throw lol

    images_file.seekg(0, std::ios::end);
    long size = (int)images_file.tellg();
    images_data = new char[size];
    images_file.seekg(0, std::ios::beg);
    images_file.read(images_data, size);

    label_file.seekg(0, std::ios::end);
    size = (int)label_file.tellg();
    label_data = new char[size];
    label_file.seekg(0, std::ios::beg);
    label_file.read(label_data, size);

    images_file.close();
    label_file.close();

    size = ((unsigned char)images_data[6] << 8) + (unsigned char)images_data[7];


    for(int i = 0; i < size; i++){
        all_images.resize(size);
        all_images[i].label = label_data[i + 8];
        for(int j = 0; j < 784; j++){
            all_images[i].pixels[j] = ((float)(*(unsigned char*)(char*)&images_data[i * 784 + j + 16]) + 1) / 256;
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(all_images.begin(), all_images.end(), g);

    delete []images_data;
    delete []label_data;

}

float sigmoid_func(float input){
    return std::max(input, input / 100);
}

float sigmoid_derivative(float input){
    return input < 0 ? 0.01 : 1;
}

float inverse_sigmoid_func(float input){
    return std::min(input, input * 100);
}


void guess_image(images& image, std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    for(int i = 0; i < 784; i++){
        neurons[0][i].value = image.pixels[i];
    }
    for(int i = 1; i < 3; i++){
        for(int j = 0; j < neurons[i].size(); j++){
            neurons[i][j].value = 0;
        }
    }
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < neurons[i].size() * neurons[i + 1].size(); j++){
            neurons[i + 1][j / neurons[i].size()].value += neurons[i][j %  neurons[i].size()].value * weights[i][j];
            if((j + 1) %  neurons[i].size() == 0){
                neurons[i + 1][j / neurons[i].size()].value += neurons[i + 1][j / neurons[i].size()].bias;
                neurons[i + 1][j / neurons[i].size()].value = sigmoid_func(neurons[i + 1][j / neurons[i].size()].value);
            }
        }
    }

}

void loadWeights_biases(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, const std::string& path){
    char* data = new char[106408];

    std::ifstream wbfile;
    wbfile.open(path, std::ios::in);
    wbfile.read(data, 106408);
    wbfile.close();


    int index = 0;
    for(auto & neuron_layer : neurons){
        for(auto & neuron : neuron_layer){
            neuron.bias = *(float*)&data[index];
            index += 4;
        }
    }
    for(auto & weight_layer : weights){
        for(auto & weight : weight_layer){
            weight = *(float*)&data[index];
            index += 4;
        }
    }

    delete[] data;
}

void saveWeights_biases(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, const std::string& path){

    char* data = new char[106408];
    int index = 0;
    for(int i = 0; i < neurons.size(); i++){
        for(int j = 0; j < neurons[i].size(); j++){
            *(float*)&data[index] = neurons[i][j].bias;
            index += 4;
        }
    }
    for(auto & weight_layer : weights){
        for(auto & weight : weight_layer){
            *(float*)&data[index] = weight;
            index += 4;
        }
    }

    std::ofstream wbfile;
    wbfile.open(path, std::ios::out | std::ios::trunc);
    wbfile.write(data, 106408);
    wbfile.close();

    delete[] data;
}

void test_ai(std::vector<images>& all_images, std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    int image_number = 0;
    int correct_answers = 0;
    float average_cost = 0;
    for(auto& image : all_images){
        image_number++;
        guess_image(image, neurons, weights);
        int best_guess_number;
        float best_guess_value = 0;

        for(int i = 0; i < 10; i++){
            if(neurons[3][i].value > best_guess_value){
                best_guess_value = neurons[3][i].value;
                best_guess_number = i;
            }
        }
        for(int i = 0; i < 10; i++){
            if(image.label == i)
                average_cost += pow((neurons[3][i].value - 1), 2);
            else
                average_cost += pow(neurons[3][i].value, 2);
        }
        if(best_guess_number == image.label)
            correct_answers++;
        float correct_percentage = 0;
        if(correct_answers)
            correct_percentage = (float)correct_answers / (float)image_number;


        if(image_number % 100 == 0) {
            std::cout << image_number << std::endl;
        }
        if(image_number % 9999 == 0) {
            std::cout << std::to_string(correct_percentage).substr(0, 5);
        }
    }
    average_cost /= all_images.size();
    std::cout << std::endl << average_cost;
}

void backprop_layer(int layer, std::vector<std::vector<node>>& neurons, std::vector<float>& weights, std::vector<std::vector<node>>& new_neurons, std::vector<float>& new_weights){
    for(int i = 0; i < weights.size(); i++){//setting up weights
        float dc_dw = neurons[layer][i % neurons[layer].size()].value;
        dc_dw *= sigmoid_derivative(inverse_sigmoid_func(neurons[layer][i % neurons[layer].size()].value));
        dc_dw *= (2 * (neurons[layer + 1][i / neurons[layer].size()].value - new_neurons[layer + 1][i / neurons[layer].size()].value));
        new_weights[i] += dc_dw;
    }
    for(int i = 0; i < neurons[layer + 1].size(); i++){//setting up biases
        float dc_db = sigmoid_derivative(inverse_sigmoid_func(neurons[layer + 1][i].value));
        dc_db *= 2 * (neurons[layer + 1][i].value - new_neurons[layer + 1][i].value);
        new_neurons[layer + 1][i].bias += dc_db;
    }
    for(int i = 0; i < weights.size(); i++){//setting up next layer
        float dc_dv = weights[i];
        dc_dv *= sigmoid_derivative(inverse_sigmoid_func(neurons[layer][i % neurons[layer].size()].value));
        dc_dv *= (2 * (neurons[layer + 1][i / neurons[layer].size()].value - new_neurons[layer + 1][i / neurons[layer].size()].value));
        new_neurons[layer][i % neurons[layer].size()].value += dc_dv;
        if(i / neurons[layer].size() == neurons[layer].size() - 1)
            new_neurons[layer][i % neurons[layer].size()].value += neurons[layer][i % neurons[layer].size()].value;
    }
}


void backprop(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, std::vector<images>& all_images, int start_images, float& average_cost){
    std::vector<std::vector<node>> new_neurons(neurons.size());
    std::vector<std::vector<float>> new_weights(weights.size());
    for(int i = 0; i < neurons.size(); i++) {
        new_neurons[i].resize(neurons[i].size());
        for(auto & j : new_neurons[i]) {
            j.value = 0;
            j.bias = 0;
        }
    }
    for(int i = 0; i < weights.size(); i++) {
        new_weights[i].resize(weights[i].size());
        for(auto & j : new_weights[i]) {
            j = 0;
        }
    }
    float cost = 0;

    for(int i = 0 + start_images; i < 100 + start_images; i++) {

        guess_image(all_images[i], neurons, weights);

        for(int j = 0; j < 2; j++)
            for(int k = 0; k < new_neurons[j].size(); k++)
                new_neurons[j][k].value = 0;

        for (int j = 0; j < 10; j++) {
            new_neurons[3][j].value = j == all_images[i].label ? 1 : 0;//value is used to store what the values should be
            cost += pow(new_neurons[3][j].value - neurons[3][j].value, 2);
        }

        for (int j = 2; j >= 0; j--) {
            backprop_layer(j, neurons, weights[j], new_neurons, new_weights[j]);
        }
    }
    cost /= 100;
    average_cost += cost;
    for(int i = 1; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            neurons[i][j].bias += ((- cost / new_neurons[i][j].bias) / LEARNING_CURVE) / 100;
        }
    }
    for(int i = 0; i < weights.size(); i++) {
        for(int j = 0; j < weights[i].size(); j++) {
            weights[i][j] += ((-cost / new_weights[i][j]) / LEARNING_CURVE) / 100;
        }
    }
}

//                                                                                                                                                                        here i change the speed V
int main(int argc, char *argv[]) {//current accuracy of 10,41%, average cost of 600 is 5.23427, 5.23301, 5.23146, 5.22817, 5.22496, 5.22178, 5.21941,  5.22303, 5.21695, 5.20278, 5.1978, 5.19007, 5.03512, 4.6579, 4.53928,  4.81085, 4.72235, 4.72696,  4.73888, 4.71454, 4.61375, 4.87692, 4.67817, 4.47415, 4.33738, 4.32731, 4.20427, 4.23877, 4.02837, 3.90065, 4.05874
    srand(time(0));// delta C of:                                               0,00126, 0,00155, 0,00329, 0,00321, 0,00318, 0,00237, −0,00362, 0,00608, 0,01417, 0,00498, 0,00773, 0,15495, 0,37722, 0,11862, −0,27157, 0,0885, −0,00461, −0,01192, 0,02434
    std::vector<std::vector<node>> neurons(4);
    std::vector<std::vector<float>> weights(3);
    std::vector<images> all_images;
    std::string images_path = "/home/nejc/CLionProjects/Neural_network/AI_images_set";

    initialize_array(neurons, weights);

    if(argc != 4)
        throw;//don't have time to throw a meaningful exception so i just throw lol
    else{
        if(strcmp(argv[1], "true") == 0){
            loadWeights_biases(neurons, weights, images_path + "/neural_data.wbdata");
        }
        if(strcmp(argv[3], "test") == 0){
            loadImages(images_path + "/testing/test_images.data", images_path + "/testing/test_labels.data", all_images);//not loading values correctly i think
            test_ai(all_images, neurons, weights);
        }else {
            loadImages(images_path + "/training/train_images.data", images_path + "/training/train_labels.data",all_images);
            for(int j = 0; j < 20; j++) {
                float average_cost = 0;
                for (int i = 0; i < 600; i++) {
                    backprop(neurons, weights, all_images, i * 100, average_cost);
                    std::cout << j << " " << i << std::endl;
                }
                average_cost /= 600;
                std::cout << "average cost:" << average_cost << std::endl;
                saveWeights_biases(neurons, weights, images_path + "/neural_data.wbdata");
            }

        }
        if(strcmp(argv[2], "true") == 0){
            saveWeights_biases(neurons, weights, images_path + "/neural_data.wbdata");
        }


    }

    return 0;
}
